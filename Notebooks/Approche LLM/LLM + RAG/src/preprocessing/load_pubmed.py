"""
Loader for the PubMed summarization dataset (ccdv/pubmed-summarization).

The dataset is distributed as parquet files: train-0..train-4, validation, test.
Each row is a dict with at least:
    - "article"  : the full PubMed article text
    - "abstract" : the gold summary (used as ground truth)

We don't load the train splits — we're not fine-tuning here. We only need
the test split for evaluation.

Usage:
    from src.preprocessing.load_pubmed import load_pubmed_test
    df = load_pubmed_test("data/raw/PubMed")
    print(df.head())
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_pubmed_test(pubmed_dir: str | Path) -> pd.DataFrame:
    """
    Load the PubMed test split from local parquet files.

    Parameters
    ----------
    pubmed_dir : str or Path
        Path to the folder containing `test.parquet`.

    Returns
    -------
    pd.DataFrame with columns ["article", "abstract"].
    """
    pubmed_dir = Path(pubmed_dir)
    test_path = pubmed_dir / "test.parquet"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Expected test.parquet at {test_path}. "
            f"Found in {pubmed_dir}: {list(pubmed_dir.iterdir())}"
        )

    df = pd.read_parquet(test_path)

    # Some HF mirrors use different column names — normalise them.
    rename_map = {}
    for col in df.columns:
        if col.lower() in {"article", "document", "text"}:
            rename_map[col] = "article"
        elif col.lower() in {"abstract", "summary", "highlights"}:
            rename_map[col] = "abstract"
    df = df.rename(columns=rename_map)

    if "article" not in df.columns or "abstract" not in df.columns:
        raise ValueError(
            f"PubMed parquet must have 'article' and 'abstract' columns. "
            f"Got: {list(df.columns)}"
        )

    return df[["article", "abstract"]].copy()


if __name__ == "__main__":
    # Quick smoke test — adjust the path to where you put the data
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/PubMed"
    df = load_pubmed_test(path)
    print(f"Loaded {len(df)} test examples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst article (truncated to 500 chars):")
    print(df.iloc[0]["article"][:500])
    print(f"\nFirst abstract:")
    print(df.iloc[0]["abstract"][:500])
