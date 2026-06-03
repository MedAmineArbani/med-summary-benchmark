"""
TextRank extractive summarizer.

Two variants:
    - tfidf: classic sentence similarity via TF-IDF + cosine.
    - embeddings: sentence similarity via sentence-transformers.

Both build a sentence graph, run PageRank, and return the top-N sentences
in original document order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.sentences import split_sentences


# Lazy global so we don't reload the model for every document
_EMBEDDING_MODEL = None


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load sentence-transformers model once, reuse afterwards."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


@dataclass
class TextRankResult:
    summary: str
    selected_sentences: list[str]
    selected_indices: list[int]
    all_scores: list[float]  # PageRank score per original sentence


def _build_similarity_matrix_tfidf(sentences: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)  # no self-loops
    return sim


def _build_similarity_matrix_embeddings(sentences: list[str]) -> np.ndarray:
    model = _get_embedding_model()
    embeddings = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0.0)
    # Clamp tiny negatives that come out of float math — PageRank expects non-negative weights
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def textrank_summarize(
    text: str,
    num_sentences: int = 7,
    method: Literal["tfidf", "embeddings"] = "tfidf",
) -> TextRankResult:
    """
    Extractive summarization via TextRank.

    Parameters
    ----------
    text : str
        Input document.
    num_sentences : int
        How many sentences to keep in the final summary.
    method : "tfidf" or "embeddings"
        Sentence-similarity backbone.

    Returns
    -------
    TextRankResult
    """
    sentences = split_sentences(text)

    # Edge case: if the document has fewer sentences than we asked for, return all.
    if len(sentences) <= num_sentences:
        return TextRankResult(
            summary=" ".join(sentences),
            selected_sentences=sentences,
            selected_indices=list(range(len(sentences))),
            all_scores=[1.0] * len(sentences),
        )

    if method == "tfidf":
        sim = _build_similarity_matrix_tfidf(sentences)
    elif method == "embeddings":
        sim = _build_similarity_matrix_embeddings(sentences)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build graph and run PageRank
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph, max_iter=200, tol=1e-6)

    # Pick top-N by score, then sort them back into document order for readability
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_indices = sorted(idx for idx, _ in ranked[:num_sentences])

    return TextRankResult(
        summary=" ".join(sentences[i] for i in top_indices),
        selected_sentences=[sentences[i] for i in top_indices],
        selected_indices=top_indices,
        all_scores=[scores[i] for i in range(len(sentences))],
    )
