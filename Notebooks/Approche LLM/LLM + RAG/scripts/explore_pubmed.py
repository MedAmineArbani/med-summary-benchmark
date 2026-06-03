"""
Phase 2 — Dataset exploration.

This script computes the corpus statistics that should appear in the final
report (length distributions, vocabulary, presence of medical terms).

Run with:
    python scripts/explore_pubmed.py data/raw/PubMed

It prints stats to stdout AND saves a JSON summary to data/processed/pubmed_stats.json
so you can quote exact numbers in the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.load_pubmed import load_pubmed_test


def word_count(text: str) -> int:
    return len(text.split())


def compute_stats(df: pd.DataFrame) -> dict:
    art_lens = df["article"].map(word_count)
    abs_lens = df["abstract"].map(word_count)

    return {
        "n_documents": int(len(df)),
        "article_words": {
            "mean": float(art_lens.mean()),
            "median": float(art_lens.median()),
            "min": int(art_lens.min()),
            "max": int(art_lens.max()),
            "p95": float(np.percentile(art_lens, 95)),
        },
        "abstract_words": {
            "mean": float(abs_lens.mean()),
            "median": float(abs_lens.median()),
            "min": int(abs_lens.min()),
            "max": int(abs_lens.max()),
            "p95": float(np.percentile(abs_lens, 95)),
        },
        "compression_ratio": {
            "mean": float((abs_lens / art_lens).mean()),
            "median": float((abs_lens / art_lens).median()),
        },
    }


def main() -> None:
    pubmed_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/PubMed"
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PubMed test split from {pubmed_dir}...")
    df = load_pubmed_test(pubmed_dir)
    print(f"Loaded {len(df):,} documents.\n")

    stats = compute_stats(df)
    print("=== Corpus statistics ===")
    print(json.dumps(stats, indent=2))

    out_path = out_dir / "pubmed_stats.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"\nSaved to {out_path}")

    # Visualize length distributions (saved as PNG for the report)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df["article"].map(word_count).hist(bins=50, ax=axes[0])
        axes[0].set_title("Article length (words)")
        axes[0].set_xlabel("words")
        df["abstract"].map(word_count).hist(bins=50, ax=axes[1])
        axes[1].set_title("Abstract length (words)")
        axes[1].set_xlabel("words")
        plt.tight_layout()
        fig_path = out_dir / "pubmed_length_distribution.png"
        plt.savefig(fig_path, dpi=120)
        print(f"Saved histogram to {fig_path}")
    except ImportError:
        print("matplotlib not installed — skipping plots. `pip install matplotlib` to enable.")


if __name__ == "__main__":
    main()
