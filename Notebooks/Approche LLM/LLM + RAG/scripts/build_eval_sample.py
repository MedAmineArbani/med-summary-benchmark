"""
Phase 2 — Build the unified evaluation sample.

This is the single most important script of Phase 2. It produces ONE file
(pubmed_eval_sample.jsonl) that will be the input for ALL THREE summarization
approaches (extractive, abstractive, LLM+RAG). This way the benchmark is
strictly fair: every approach sees exactly the same 100 documents.

We apply minimal filtering:
    - drop documents with empty article or abstract
    - drop documents that are way too long (would crash the LLM context window)
    - drop documents that are too short (abstract is meaningless)

Then we take a fixed random sample with a fixed seed for reproducibility.

Run with:
    python scripts/build_eval_sample.py data/raw/PubMed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src` importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.load_pubmed import load_pubmed_test

# --- Configuration ---
SAMPLE_SIZE = 30         # number of documents in the evaluation set
                         # (reduced from 100 to fit a CPU-only setup;
                         # justified in the report as a compute constraint)
MIN_ARTICLE_WORDS = 200  # discard very short articles
MAX_ARTICLE_WORDS = 4000 # discard very long articles (Mistral context ≈ 8k tokens)
MIN_ABSTRACT_WORDS = 50
RANDOM_SEED = 42         # reproducibility


def main() -> None:
    pubmed_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/PubMed"
    out_path = Path("data/processed/pubmed_eval_sample.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading PubMed test split from {pubmed_dir}...")
    df = load_pubmed_test(pubmed_dir)
    print(f"Loaded {len(df):,} documents.")

    # Length filtering
    art_words = df["article"].str.split().str.len()
    abs_words = df["abstract"].str.split().str.len()
    mask = (
        (art_words >= MIN_ARTICLE_WORDS)
        & (art_words <= MAX_ARTICLE_WORDS)
        & (abs_words >= MIN_ABSTRACT_WORDS)
    )
    df = df[mask].reset_index(drop=True)
    print(f"After length filtering: {len(df):,} documents.")

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(df)} documents pass filters, need at least {SAMPLE_SIZE}."
        )

    # Fixed random sample
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    # Write JSONL, one document per line. Each record has a stable id.
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in sample.iterrows():
            record = {
                "id": f"pubmed_{i:03d}",
                "article": row["article"],
                "reference_summary": row["abstract"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {SAMPLE_SIZE} documents to {out_path}")
    print(f"Seed = {RANDOM_SEED}, so re-running gives the exact same sample.")
    print("\nThis file is the SINGLE input for all 3 approaches in Phases 3, 4, 5.")


if __name__ == "__main__":
    main()
