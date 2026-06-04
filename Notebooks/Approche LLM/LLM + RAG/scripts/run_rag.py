"""
Phase 5 runner — LLM + RAG on the eval sample.

For each article we:
  1. Retrieve the top-k relevant entries from the medical knowledge base.
  2. Build a RAG prompt that injects them next to the article.
  3. Generate the summary with the same Ollama model used in Phase 4.

The output file has the same shared format as Phases 3 and 4, so Phase 6
treats it identically.

Run:
    python -m src.rag.build_vector_store   # one-time setup
    python scripts/run_rag.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.ollama_client import generate_summary
from src.rag.rag_prompt import make_rag_prompt
from src.rag.retriever import format_retrieved, retrieve_for_article


DEFAULT_INPUT = "data/processed/pubmed_eval_sample.jsonl"


def load_already_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM+RAG on the eval sample.")
    parser.add_argument("--model", default="mistral", help="Ollama model tag.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many KB entries to retrieve and inject per article.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents (smoke test).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    approach_tag = f"rag_{args.model.split(':')[0]}_top{args.top_k}"
    output_path = Path(f"data/processed/predictions_{approach_tag}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sidecar file logging WHAT was retrieved for each article (for error analysis)
    retrieval_log_path = Path(f"data/processed/retrieval_log_{approach_tag}.jsonl")

    # Load eval sample
    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if args.limit is not None:
        records = records[: args.limit]

    already_done = load_already_done(output_path)
    todo = [r for r in records if r["id"] not in already_done]

    print(f"Total in eval sample : {len(records)}")
    print(f"Already done         : {len(already_done)}")
    print(f"To process now       : {len(todo)}")
    print(f"Approach tag         : {approach_tag}")
    print(f"Output file          : {output_path}")
    print(f"Retrieval log        : {retrieval_log_path}\n")

    if not todo:
        print("Nothing to do.")
        return

    truncation_count = 0

    with output_path.open("a", encoding="utf-8") as out, retrieval_log_path.open(
        "a", encoding="utf-8"
    ) as rlog:
        for rec in tqdm(todo, desc=approach_tag):
            # 1. Retrieve
            retrieved = retrieve_for_article(rec["article"], top_k=args.top_k)
            retrieved_block = format_retrieved(retrieved)

            # Log retrieval for error analysis later
            rlog.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "retrieved": [
                            {
                                "term": e.term,
                                "category": e.category,
                                "distance": round(e.distance, 4),
                            }
                            for e in retrieved
                        ],
                    }
                )
                + "\n"
            )
            rlog.flush()

            # 2. Build prompt and generate
            prompt = make_rag_prompt(retrieved_block)
            try:
                resp = generate_summary(rec["article"], prompt=prompt, model=args.model)
            except Exception as exc:
                tqdm.write(f"  [error] doc {rec['id']}: {exc}")
                continue

            if resp.truncated:
                truncation_count += 1

            out_rec = {
                "id": rec["id"],
                "approach": approach_tag,
                "prediction": resp.text,
                "reference": rec["reference_summary"],
                "elapsed_seconds": round(resp.elapsed_seconds, 3),
                "input_truncated": resp.truncated,
                "num_retrieved": len(retrieved),
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out.flush()

    print(f"\nDone. {truncation_count} articles were truncated.")


if __name__ == "__main__":
    main()
