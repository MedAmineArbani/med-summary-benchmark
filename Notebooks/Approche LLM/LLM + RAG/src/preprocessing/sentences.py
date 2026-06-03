"""
Sentence splitting for medical text.

We use NLTK's Punkt tokenizer. It's not biomedical-specific but works well
enough on PubMed abstracts and clinical notes. We add a light post-processing
step: very short fragments (< 4 words) are merged with the previous sentence
because Punkt sometimes splits on abbreviations like "Fig. 2" or "e.g.".
"""

from __future__ import annotations

import nltk


def _ensure_punkt() -> None:
    """Download Punkt models if missing (no-op if already present)."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """
    Split a document into sentences.

    Parameters
    ----------
    text : str
        Raw article text.
    min_words : int
        Sentences shorter than this get merged with the previous one. Helps
        with abbreviation false-splits (e.g. "Fig.", "et al.").

    Returns
    -------
    list of sentence strings, in document order.
    """
    _ensure_punkt()
    raw = nltk.sent_tokenize(text)

    merged: list[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if merged and len(s.split()) < min_words:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged
