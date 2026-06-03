"""
Builds a vector store from the medical knowledge base.

Strategy
--------
We use ChromaDB in persistent local mode (no server needed). Each KB entry
becomes one document; the embedding is computed on `term + content` so the
retriever finds both exact-term hits and semantic matches.

The embedding model is a general-purpose `all-MiniLM-L6-v2`. For a research
project that wants to go further, you'd swap this for a biomedical model
like `pritamdeka/S-PubMedBert-MS-MARCO`. We stick to MiniLM here to keep
CPU usage low.

Run once:
    python -m src.rag.build_vector_store

This creates ./data/knowledge_base/chroma/ which is then loaded by the
retriever at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

KB_PATH = Path("data/knowledge_base/medical_kb.json")
CHROMA_DIR = Path("data/knowledge_base/chroma")
COLLECTION_NAME = "medical_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_vector_store() -> None:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Expected knowledge base at {KB_PATH}")

    with KB_PATH.open(encoding="utf-8") as f:
        kb_entries = json.load(f)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset to keep the script idempotent — re-running cleanly rebuilds
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"kb_{i:03d}" for i in range(len(kb_entries))]
    documents = [f"{e['term']}. {e['content']}" for e in kb_entries]
    metadatas = [
        {"term": e["term"], "category": e["category"]} for e in kb_entries
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Indexed {len(kb_entries)} entries into {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}, embedding model: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    build_vector_store()
