# Fine-tuning BART-base for Medical Question Answering

## Table of Contents

1. [Project Overview](#project-overview)
2. [Model Selection and Rationale](#model-selection-and-rationale)
3. [Dataset](#dataset)
4. [Environment and Dependencies](#environment-and-dependencies)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Configuration and Hyperparameters](#configuration-and-hyperparameters)
7. [Data Preprocessing and Split Strategy](#data-preprocessing-and-split-strategy)
8. [Tokenization](#tokenization)
9. [Training Procedure](#training-procedure)
10. [Generation Strategy](#generation-strategy)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Results and Analysis](#results-and-analysis)
13. [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

This project fine-tunes `facebook/bart-base` on a medical question-answering dataset in which each example consists of a clinical note, a question about that note, and a reference answer. The objective is to train the model to generate factual, coherent, and concise answers conditioned on both the clinical context and the specific question asked.

This work forms the abstractive summarization branch of a larger research project on automatic generation of medical document summaries, alongside extractive approaches (TextRank, TF-IDF) and prompt-engineering approaches (LLaMA 3.1 via llama-cpp-python).

The notebook was designed to run on Kaggle with a single NVIDIA Tesla T4 GPU and covers the full pipeline: dependency installation, data loading and splitting, tokenization, fine-tuning with `Seq2SeqTrainer`, inference, and multi-metric evaluation.

---

## Model Selection and Rationale

### Why BART

BART (Bidirectional and Auto-Regressive Transformers) is a denoising autoencoder built for sequence-to-sequence tasks. Unlike encoder-only models such as BERT, or decoder-only models such as GPT, BART has a full encoder-decoder architecture. The encoder processes the input bidirectionally (like BERT), and the decoder generates the output autoregressively (like GPT). This design makes it naturally suited for generation tasks such as abstractive summarization and question answering, because the encoder can build a rich contextual representation of the full input before the decoder generates each output token.

BART was pre-trained specifically with text generation in mind. The pre-training objective involved corrupting text with various noise functions (token masking, sentence permutation, document rotation, token deletion, and text infilling) and training the model to reconstruct the original text. This contrasts with T5, which was pre-trained on a more generic text-to-text framework. As a result, BART tends to perform well out of the box on summarization, which aligns closely with what is needed here.

### Why bart-base and Not a Larger Variant

The `facebook/bart-base` variant has approximately 139 million parameters. On a Kaggle T4 GPU with 16 GB of VRAM, this fits comfortably at batch size 8 with 16-bit mixed precision enabled. Larger variants such as `facebook/bart-large` (approximately 400 million parameters) would require either a smaller batch size or gradient checkpointing, and would take significantly longer to train, potentially exceeding the 12-hour Kaggle session limit.

The quality difference between `bart-base` and `bart-large` on domain-specific fine-tuning tasks is often smaller than it appears, especially when the fine-tuning corpus is large enough to provide sufficient task-specific signal. Given the dataset size and the training budget, `bart-base` is the appropriate choice.

### Considered Alternatives

| Model | Parameters | Fine-tuning time (T4) | Summarization quality | Medical domain |
|---|---|---|---|---|
| T5-small | 60 M | ~1 hour | Adequate | General |
| facebook/bart-base | 139 M | ~2-3 hours | Good | General |
| facebook/bart-large | 400 M | ~8+ hours, risk of OOM | Very good | General |
| GanjinZero/biobart-base | ~140 M | ~2-3 hours | Good | Biomedical |

`GanjinZero/biobart-base` is an architecturally identical variant of `bart-base` that was further pre-trained on PubMed and PMC biomedical literature. It is a drop-in replacement in this pipeline (changing only the `MODEL_NAME` configuration parameter) and would likely produce higher BERTScore-F1 on medical text due to its domain-specific vocabulary alignment. It was not selected as the primary model for this experiment in order to establish a general baseline first.

---

## Dataset

The dataset is `QA_dataset.csv`, sourced from the Kaggle dataset `mdamineelarbani/qa-datset-prombt-engineering`. Each row contains three relevant columns:

- `note`: a clinical document (discharge summary, hospital course report, or similar)
- `question`: a specific question about the clinical content of the note
- `answer`: a reference answer that the model should learn to generate

The dataset was originally constructed to support prompt engineering experiments with LLMs and contains examples covering multiple clinical NLP subtasks including summarization, relation extraction, coreference resolution, and paraphrasing. The `task` column encodes this information and is used during evaluation to break down performance by task type.

The input to the model is formed by concatenating the question and the note using BART's native separator token `</s>` as a delimiter. This produces inputs of the form:

```
{question} </s> {note}
```

The use of `</s>` as a separator is intentional. BART was pre-trained with this token as its end-of-sequence and segment boundary marker. By using it as a separator between the two input segments, the encoder receives an explicit structural signal that the input contains two distinct but related parts, which helps it attend to the question when encoding the note context.

---

## Environment and Dependencies

The notebook targets the Kaggle Notebook environment with the following setup:

- Platform: Kaggle Notebooks
- GPU: NVIDIA Tesla T4, 16 GB VRAM, CUDA architecture Turing (sm_75)
- Python: 3.12.13

The following package versions are pinned for reproducibility:

```
transformers==4.40.0
datasets==2.19.0
evaluate==0.4.1
rouge-score==0.1.2
bert-score==0.3.13
accelerate>=0.30.0
peft>=0.10.0
seaborn
```

The versions of `transformers` and `accelerate` matter here. The `Seq2SeqTrainer` requires `accelerate >= 0.30.0` because `clear_device_cache` was introduced in that release and is called internally during training. Mismatched versions between `peft` and `accelerate` can cause silent failures or deprecation warnings that affect training stability, which is why both are aligned explicitly.

A kernel restart is required after installation because the Kaggle base image ships with older versions of these libraries and the new versions must be loaded fresh before any import occurs.

---

## Pipeline Architecture

The complete pipeline is organized into nine sequential notebook cells:

1. **Installation** — Pin and install all dependencies, instruct the user to restart the kernel.
2. **Configuration** — Define a central `Config` dataclass holding every hyperparameter and file path.
3. **Data loading and splitting** — Load the CSV, clean the data, construct the input text, compute length statistics, visualize distributions, and produce stratified 80/10/10 train/validation/test splits.
4. **Tokenization** — Load the BART tokenizer, define the preprocessing function, map it over the HuggingFace `DatasetDict`, and verify a decoded example.
5. **Fine-tuning** — Load the model, configure the `Seq2SeqTrainer` with all training arguments, attach an early stopping callback, run training, and save the best checkpoint.
6. **Learning curves** — Extract the training log history and plot loss curves and ROUGE curves across training steps.
7. **Inference** — Load the saved best model, generate answers for every test example using beam search.
8. **Metric computation** — Calculate ROUGE-1, ROUGE-2, ROUGE-L, BERTScore-F1, and compression ratio line by line and aggregate results.
9. **Visualization and qualitative analysis** — Plot metric distributions, scatter ROUGE-1 against BERTScore-F1, and display the best and worst generated examples.

---

## Configuration and Hyperparameters

All hyperparameters are defined in a single `Config` dataclass in Cell 2. This design choice means that every knob in the pipeline can be adjusted in one place without hunting through multiple cells.

### Input Construction

```
QUESTION_COL = 'question'
NOTE_COL     = 'note'
TARGET_COL   = 'answer'
SEP_TOKEN    = ' </s> '
```

The separator `</s>` is BART's native end-of-sequence token. Using it between the question and the note serves as a structural boundary signal to the encoder without requiring any additional architectural modifications.

### Tokenization Lengths

```
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128
```

`MAX_SOURCE_LEN = 512` is BART's maximum encoder input length. Sequences exceeding 512 tokens are truncated from the right. From the length distribution analysis in Cell 3, a fraction of examples have concatenated inputs exceeding 512 tokens, meaning some note content is lost for long examples. This is a known limitation of the base model.

`MAX_TARGET_LEN = 128` is set to cover the observed distribution of reference answers. The vast majority of answers in the dataset fall below 128 tokens, so this limit results in minimal truncation of the target sequences.

### Training Hyperparameters

```
NUM_EPOCHS   = 4
BATCH_SIZE   = 8
GRAD_ACCUM   = 2      # effective batch size = 16
LR           = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
FP16         = True
```

**Learning rate (3e-5):** This value sits in the range commonly reported in the literature for fine-tuning BART-base on summarization tasks. A higher rate such as 5e-5 can cause instability in the early epochs because BART's pre-trained weights are sensitive to large updates. A lower rate such as 1e-5 converges more slowly and may not fully adapt the model to the medical domain within four epochs.

**Effective batch size of 16:** The physical batch size is set to 8 to stay within the T4's VRAM budget. Gradient accumulation over 2 steps produces a gradient update equivalent to a batch of 16 examples, which provides more stable gradient estimates than a batch of 8 while avoiding out-of-memory errors. In practice, BART-base at `MAX_SOURCE_LEN=512` with FP16 uses approximately 10-12 GB of VRAM at batch size 8, leaving headroom for the optimizer states.

**Weight decay (0.01):** A small L2 regularization term that penalizes large weight values and helps prevent overfitting, particularly important when fine-tuning on a dataset that may not be large enough to fully constrain all 139M parameters.

**Warmup ratio (0.06):** The learning rate is linearly increased from 0 to the target value over the first 6% of training steps. This prevents the model from making destructive updates at the start of training when the optimizer has not yet accumulated reliable gradient statistics.

**FP16 mixed precision:** The T4 natively supports FP16 tensor core operations. Training in mixed precision halves the memory footprint of activations and allows larger batch sizes or longer sequences. The `transformers` library handles the precision casting automatically through `accelerate`.

**4 epochs:** Given the dataset size, four passes over the training data are sufficient to achieve convergence while avoiding overfitting. The early stopping callback with patience 3 will terminate training earlier if the validation ROUGE-L stops improving.

### Evaluation and Checkpointing

```
EVAL_STEPS       = 500
SAVE_STEPS       = 500
SAVE_TOTAL       = 2
metric_for_best  = 'rougeL'
```

Evaluation is performed every 500 steps rather than every epoch. This provides more granular insight into training dynamics and allows the early stopping callback to trigger within an epoch if the model overfits. Only the 2 best checkpoints are kept on disk to limit storage usage. The best checkpoint is selected based on ROUGE-L because it captures both unigram overlap and structural coherence of the generated text, making it a more informative criterion than ROUGE-1 alone.

---

## Data Preprocessing and Split Strategy

### Cleaning Steps

After loading the CSV, the following cleaning operations are applied:

1. Rows with missing values in `question`, `note`, or `answer` are dropped.
2. All text columns are cast to string and stripped of leading and trailing whitespace.
3. Rows where the question has fewer than 3 words or the answer has fewer than 5 words are removed. These thresholds eliminate degenerate examples such as empty strings, single-word answers, or malformed rows that were introduced by the LLM generation process in earlier pipeline stages.

### Split Strategy

The dataset is split into train (80%), validation (10%), and test (10%) sets. The split is stratified on the `task` column when it is present, ensuring that the proportion of each task type (Summarization, Relation Extraction, Coreference Resolution, Paraphrasing) is preserved across all three sets. This is important because performance varies significantly across task types, and an unstratified split could produce misleading aggregate metrics if one split happens to contain an excess of easier examples.

The random seed is fixed at 42 for full reproducibility. The three splits are saved to CSV files at the end of Cell 3 for later use.

---

## Tokenization

The tokenizer used is `AutoTokenizer.from_pretrained('facebook/bart-base')`, which loads a `RobertaTokenizer` configured for the BART architecture.

The preprocessing function applies the following transformations:

**Source tokenization:**

```python
model_inputs = tokenizer(
    examples['input_text'],
    max_length=CFG.MAX_SOURCE_LEN,
    truncation=True,
    padding='max_length',
)
```

**Target tokenization:**

```python
labels = tokenizer(
    text_target=examples['target_text'],
    max_length=CFG.MAX_TARGET_LEN,
    truncation=True,
    padding='max_length',
)
```

The `text_target=` keyword argument is the correct way to tokenize decoder targets in `transformers >= 4.0`. The deprecated `as_target_tokenizer()` context manager was removed in modern versions because it relied on side-effectful global state, which caused issues in multi-process tokenization. Using `text_target=` is both semantically equivalent and compatible with the current API.

After tokenization, all padding token IDs in the label sequences are replaced with `-100`. The `CrossEntropyLoss` in PyTorch ignores positions labeled `-100` by default, meaning the model is not penalized for predicting pad tokens. Without this step, the model would learn to copy the pad token ID in the output, distorting the loss signal.

---

## Training Procedure

Training uses `Seq2SeqTrainer`, which extends `Trainer` with two capabilities specific to encoder-decoder models: it calls `model.generate()` during evaluation (when `predict_with_generate=True`) to produce actual decoded sequences for metric computation, and it handles the shift of decoder input IDs automatically.

The `compute_metrics` function decodes the generated sequences and reference labels using the tokenizer, strips whitespace, and computes ROUGE-1, ROUGE-2, ROUGE-L, and average generation length using the `evaluate` library. These metrics are reported in the training logs at each evaluation checkpoint and used to select the best model.

An `EarlyStoppingCallback` with `patience=3` is attached to the trainer. If the validation ROUGE-L does not improve across 3 consecutive evaluation checkpoints, training is stopped and the best checkpoint is restored. This prevents wasting compute on epochs where the model has stopped improving, and guards against overfitting in later epochs.

At the end of training, the best model and its tokenizer are saved to `CFG.OUTPUT_DIR` for use in inference.

---

## Generation Strategy

During inference, text is generated with beam search using the following settings:

```
num_beams            = 4
max_length           = 128
min_length           = 20
length_penalty       = 1.0
no_repeat_ngram_size = 3
```

**Beam search with 4 beams:** Greedy decoding (1 beam) selects the single most probable token at each step, which tends to produce repetitive and generic text. Beam search maintains 4 candidate sequences simultaneously and selects the one with the highest overall log-probability, producing more fluent and coherent outputs. Increasing beyond 4 beams offers diminishing returns in quality while increasing inference time significantly.

**min_length = 20:** A minimum output length prevents the model from collapsing to very short outputs such as a single sentence when the reference answer is several sentences long. This is particularly relevant for questions expecting a multi-step clinical explanation.

**length_penalty = 1.0:** A value of 1.0 applies no bias toward shorter or longer sequences, letting the model output a length that reflects its learned distribution. Values greater than 1.0 would encourage longer outputs; values less than 1.0 would encourage shorter outputs.

**no_repeat_ngram_size = 3:** Any trigram that has already appeared in the output is blocked from appearing again. This prevents the common failure mode in which beam search produces repetitive phrases, which is especially frequent in medical text where the same clinical terms can dominate the probability distribution.

---

## Evaluation Metrics

Performance is assessed using five metrics computed on the held-out test set.

### ROUGE-1

ROUGE-1 measures the F-measure of unigram overlap between the generated answer and the reference answer. It captures the fraction of reference words that appear in the generated output (recall) and the fraction of generated words that appear in the reference (precision). It is a measure of vocabulary coverage and is sensitive to exact word matches, which makes it less tolerant of paraphrase.

### ROUGE-2

ROUGE-2 extends this to bigrams, measuring the overlap of consecutive two-word sequences. It is more sensitive to local fluency and phrasing coherence than ROUGE-1, but also more sensitive to the choice of words, since a synonymous paraphrase that preserves meaning but not surface form will score zero on bigram overlap.

### ROUGE-L

ROUGE-L uses the longest common subsequence (LCS) between the generated output and the reference. Unlike ROUGE-1 and ROUGE-2, it does not require the matching tokens to be contiguous, which makes it more tolerant of word reordering while still capturing structural coherence. It tends to be the most informative of the three ROUGE variants for summarization tasks.

### BERTScore-F1

BERTScore computes the cosine similarity between contextual token embeddings of the generated text and the reference text, using a pre-trained transformer as the embedding model. In this pipeline, `distilbert-base-uncased` is used as the embedding backbone. BERTScore-F1 is particularly relevant for medical text evaluation because it measures semantic similarity rather than lexical overlap. A generated answer that uses different but medically equivalent terminology (for example, "myocardial infarction" versus "heart attack") would score very low on ROUGE metrics but would score high on BERTScore because the embeddings of these phrases are similar in the contextual embedding space.

### Compression Ratio

Compression ratio is defined as:

```
compression (%) = (number of words in generated answer / number of words in source note) * 100
```

It quantifies the degree to which the model condenses the source material. A compression ratio of 10% means the generated answer contains one word for every ten words in the source note. This metric is useful for diagnosing generation behavior: very low compression may indicate that the model is producing minimal answers, while very high compression (close to 100%) may indicate that the model is copying large portions of the note rather than abstracting.

---

## Results and Analysis

The evaluation results below are drawn directly from the notebook outputs.

### Overall Test Set Performance

| Metric | Value |
|---|---|
| ROUGE-1 | 0.461 |
| ROUGE-2 | 0.307 |
| ROUGE-L | 0.393 |
| BERTScore-F1 | 0.893 |

These numbers reflect the aggregate average across all examples in the test set, including all task types.

### Interpretation

**ROUGE-1 of 0.461** indicates that roughly 46% of unigrams in the generated answers overlap with the reference answers. For a medical QA task where the model must locate and summarize specific clinical facts from a long document, this is a reasonable result for a base-sized model fine-tuned from scratch on a single GPU session. The score is not high enough to suggest near-perfect factual alignment, but it is well above the range expected from a zero-shot baseline (typically 0.20 to 0.30 on this type of task).

**ROUGE-2 of 0.307** shows a meaningful drop from ROUGE-1, which is expected: bigram matching is stricter than unigram matching, and the model does not always reproduce exact two-word sequences from the reference even when it captures the same content. The gap between ROUGE-1 and ROUGE-2 reflects the degree of paraphrase in the generated outputs.

**ROUGE-L of 0.393** falls between ROUGE-1 and ROUGE-2, which is the expected ordering. The LCS-based metric confirms that the model produces outputs with reasonable sequential overlap with the references, meaning it tends to follow the same general structure of answer presentation as the references.

**BERTScore-F1 of 0.893** is the most informative single number here. A score near 0.90 indicates strong semantic alignment between the generated answers and the references. The gap between the ROUGE scores and BERTScore-F1 reflects the fact that the model often generates paraphrases that are semantically equivalent to the reference but use different surface forms. This is precisely the behavior expected from a generative model and confirms that ROUGE alone would underestimate the true quality of the generated outputs.

### Qualitative Analysis

The best-performing examples show near-identical content to the references with only minor surface-level variation. For example:

- Reference: "The patient's response to the CBD+ solution included notable changes in anxiety levels, avoidance behaviors, and sociability, with sustained improvement for two years."
- Generated: "The patient's response to the CBD+ solution was notable changes in anxiety levels, avoidance behaviors, and sociability, and indicated a therapeutic effect."

The scores for this example (ROUGE-1: 0.827, ROUGE-L: 0.827, BERTScore-F1: 0.970) confirm that the generated text is almost identical to the reference.

The worst-performing examples reveal a consistent failure pattern: the model generates factually plausible content that is not grounded in the specific details of the reference answer. For example, for a question about a patient presenting with xerostomia and hypercalcemia:

- Reference mentions the diagnosis of undifferentiated connective tissue disease and monoclonal gammopathy.
- Generated output describes the clinical presentation (dry mouth, weight loss, lip biopsy) but does not name the specific diagnoses.

This failure mode is not a hallucination in the strict sense (the generated content is consistent with the note), but rather an under-specification of the required answer. The model correctly identifies the relevant clinical entities but does not extract the specific diagnostic conclusion that the reference focuses on. This suggests that for complex multi-step clinical reasoning questions, additional training signal or a larger model may be needed.

---

## Limitations and Future Work

**Context window truncation.** The maximum encoder length of 512 tokens means that long clinical notes are truncated. The portion of the note most relevant to the answer may appear in the truncated section. A sliding window approach or a long-context model such as LongT5 or LED would address this limitation.

**Domain specificity.** `facebook/bart-base` was pre-trained on general English text (CNN/DailyMail, BooksCorpus, English Wikipedia). Fine-tuning on a medical dataset partially addresses the domain gap, but a model pre-trained on biomedical literature such as `GanjinZero/biobart-base` would start with better representations of medical terminology. Replacing the base model with BioBART is a one-line change in this pipeline.

**Hallucination analysis.** The current evaluation does not explicitly measure factual consistency between the generated answer and the source note. A faithfulness metric such as FactCC or QuestEval would be required to quantify the rate at which the model generates claims not supported by the source.

**Task-specific fine-tuning.** All four task types (Summarization, Relation Extraction, Coreference Resolution, Paraphrasing) are trained jointly. Task-specific fine-tuning, or the use of task prefixes as in T5, might improve performance on individual tasks at the cost of requiring separate training runs.

**Comparison with extractive baselines.** This notebook produces only abstractive outputs. A complete evaluation would compare the ROUGE and BERTScore results of this fine-tuned BART model against the extractive baselines (TextRank, TF-IDF) and the LLM prompt-engineering approach on the same test set, using the same evaluation code.