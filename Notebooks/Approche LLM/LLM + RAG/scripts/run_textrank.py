"""
Phase 3 runner — apply TextRank to the unified evaluation sample.

Reads:   data/processed/pubmed_eval_sample.jsonl
Writes:  data/processed/predictions_textrank_<method>.jsonl

The output format is the SHARED prediction format that all three approaches
(extractive, abstractive, LLM+RAG) will produce. Phase 6 (evaluation) will
load any file matching this format and compute the metrics.

Shared prediction format (one JSON per line):
    {
      "id": "pubmed_000",
      "approach": "textrank_tfidf" | "textrank_embeddings" | "llm" | ...,
      "prediction": "...generated summary...",
      "reference": "...gold abstract from PubMed...",
      "elapsed_seconds": 0.42
    }

Run with:
    python scripts/run_textrank.py --method tfidf
    python scripts/run_textrank.py --method embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Make `src` importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractive.textrank import textrank_summarize


DEFAULT_INPUT = "data/processed/pubmed_eval_sample.jsonl"
DEFAULT_NUM_SENTENCES = 7


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TextRank on the eval sample.")
    parser.add_argument(
        "--method",
        choices=["tfidf", "embeddings"],
        default="tfidf",
        help="Similarity backbone for TextRank.",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=DEFAULT_NUM_SENTENCES,
        help="Number of sentences to keep in each summary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(f"data/processed/predictions_textrank_{args.method}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load eval sample
    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} documents from {input_path}")
    print(f"Method: TextRank + {args.method}, keeping {args.num_sentences} sentences each")

    approach_tag = f"textrank_{args.method}"

    with output_path.open("w", encoding="utf-8") as out:
        for rec in tqdm(records, desc=approach_tag):
            t0 = time.perf_counter()
            result = textrank_summarize(
                rec["article"],
                num_sentences=args.num_sentences,
                method=args.method,
            )
            elapsed = time.perf_counter() - t0

            out_rec = {
                "id": rec["id"],
                "approach": approach_tag,
                "prediction": result.summary,
                "reference": rec["reference_summary"],
                "elapsed_seconds": round(elapsed, 3),
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} predictions to {output_path}")
    print("Phase 3 done for this method. Re-run with --method embeddings for the other variant.")


if __name__ == "__main__":
    main()
