"""
Retriever for the medical knowledge base.

Two query strategies are exposed:

1. retrieve_for_article(article, top_k):
   The simplest approach. The article text is used directly as the query;
   the most semantically similar KB entries are returned. Robust but coarse —
   for a 4000-word article the embedding is noisy.

2. retrieve_for_entities(entities, top_k_per_entity):
   The smart approach. We pre-extract medical entities from the article
   (drugs, diseases, lab terms) and run one query per entity. This produces
   targeted retrievals tied to actual content of the article.

We use strategy 1 by default for simplicity, then add strategy 2 as an
ablation in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.rag.build_vector_store import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


@dataclass
class RetrievedEntry:
    term: str
    category: str
    content: str
    distance: float  # lower = more similar (cosine distance)


_client = None
_collection = None


def _get_collection():
    """Open the persistent Chroma collection. Cached on first call."""
    global _client, _collection
    if _collection is None:
        if not Path(CHROMA_DIR).exists():
            raise FileNotFoundError(
                f"Vector store not found at {CHROMA_DIR}. "
                f"Run: python -m src.rag.build_vector_store"
            )
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    return _collection


def retrieve_for_article(article: str, top_k: int = 5) -> list[RetrievedEntry]:
    """
    Retrieve the top-k KB entries most relevant to a whole article.

    For long articles we truncate to the first ~1500 chars so the embedding
    isn't washed out by the article tail (usually references / discussion).
    """
    collection = _get_collection()
    query = article[:1500]
    res = collection.query(query_texts=[query], n_results=top_k)

    out: list[RetrievedEntry] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        # Strip the "term. " prefix we added at indexing time
        content = doc.split(". ", 1)[1] if ". " in doc else doc
        out.append(
            RetrievedEntry(
                term=meta["term"],
                category=meta["category"],
                content=content,
                distance=float(dist),
            )
        )
    return out


def format_retrieved(entries: list[RetrievedEntry]) -> str:
    """Format retrieved entries as a clean text block to inject into a prompt."""
    if not entries:
        return "(no relevant reference knowledge found)"
    lines = []
    for e in entries:
        lines.append(f"- {e.term} ({e.category}): {e.content}")
    return "\n".join(lines)
