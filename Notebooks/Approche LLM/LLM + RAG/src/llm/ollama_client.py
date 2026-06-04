"""
Thin wrapper around the Ollama Python client.

Adds three things on top of the raw client:

1. Article truncation. Mistral-7B has an 8k-token context window. PubMed
   articles can blow past that. We truncate by word count with a conservative
   budget so the article + prompt + generated tokens all fit.

2. Retry on transient errors. The local daemon sometimes hiccups; we retry
   once with a short delay.

3. Deterministic generation. We pin temperature=0.2 and seed=42. We are not
   trying to get creative outputs — we want reproducible summaries that an
   evaluator can re-run and get the same numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import ollama

from src.llm.prompts import Prompt

# Rough budget. 1 word ~= 1.3 tokens for English, so 4000 words ~= 5200 tokens.
# That leaves ~2800 tokens for the prompt scaffolding + system message + output,
# which is comfortable for Mistral-7B (8k window).
MAX_ARTICLE_WORDS = 4000


@dataclass
class LLMResponse:
    text: str
    elapsed_seconds: float
    truncated: bool


def _truncate_article(article: str, max_words: int = MAX_ARTICLE_WORDS) -> tuple[str, bool]:
    words = article.split()
    if len(words) <= max_words:
        return article, False
    return " ".join(words[:max_words]), True


def generate_summary(
    article: str,
    prompt: Prompt,
    model: str = "mistral",
    temperature: float = 0.2,
    seed: int = 42,
    num_predict: int = 400,
    max_retries: int = 1,
) -> LLMResponse:
    """
    Generate a summary for one article using a local Ollama model.

    Parameters
    ----------
    article : str
        Raw input document.
    prompt : Prompt
        Prompt template to use (see src.llm.prompts).
    model : str
        Ollama model tag (e.g. "mistral", "mistral:7b-instruct-q4_0", "phi3:mini").
    temperature : float
        0.0 = greedy, 1.0 = creative. 0.2 = nearly-deterministic with a
        small relief valve for repetitive outputs.
    seed : int
        Sampling seed for reproducibility.
    num_predict : int
        Hard cap on generated tokens. 400 is enough for ~250-word summaries.
    max_retries : int
        Retry on transient Ollama errors before giving up.
    """
    article_trimmed, was_truncated = _truncate_article(article)

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.render_user(article_trimmed)},
    ]
    options = {
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            response = ollama.chat(model=model, messages=messages, options=options)
            elapsed = time.perf_counter() - t0
            return LLMResponse(
                text=response["message"]["content"].strip(),
                elapsed_seconds=elapsed,
                truncated=was_truncated,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2.0)
            else:
                raise

    raise RuntimeError(f"Unreachable: {last_error}")
