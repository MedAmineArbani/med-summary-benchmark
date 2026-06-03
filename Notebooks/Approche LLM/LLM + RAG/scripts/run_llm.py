"""
Phase 4 runner — apply an LLM + prompt configuration to the eval sample.

Produces predictions in the shared format used by all approaches.

Why "resume support": at ~30s per document on CPU, 100 documents take
~50 minutes. If you Ctrl-C or your machine sleeps, we don't want to redo
everything. The script appends to the output file and skips IDs already
processed.

Run:
    python scripts/run_llm.py --prompt basic
    python scripts/run_llm.py --prompt structured

Optional flags:
    --model mistral              (default; or e.g. "phi3:mini" for small machines)
    --limit 10                   (process only the first 10 docs — useful for a smoke test)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.ollama_client import generate_summary
from src.llm.prompts import get_prompt


DEFAULT_INPUT = "data/processed/pubmed_eval_sample.jsonl"


def load_already_done(output_path: Path) -> set[str]:
    """Return the set of IDs already present in the output file (resume support)."""
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
    parser = argparse.ArgumentParser(description="Run LLM+prompt on the eval sample.")
    parser.add_argument(
        "--prompt",
        choices=["basic", "structured"],
        required=True,
        help="Which prompt variant to use.",
    )
    parser.add_argument("--model", default="mistral", help="Ollama model tag.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents (smoke test).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    approach_tag = f"llm_{args.model.split(':')[0]}_{args.prompt}"
    output_path = Path(f"data/processed/predictions_{approach_tag}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load eval sample
    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if args.limit is not None:
        records = records[: args.limit]

    # Resume support
    already_done = load_already_done(output_path)
    todo = [r for r in records if r["id"] not in already_done]

    print(f"Total in eval sample : {len(records)}")
    print(f"Already done         : {len(already_done)}")
    print(f"To process now       : {len(todo)}")
    print(f"Approach tag         : {approach_tag}")
    print(f"Output file          : {output_path}\n")

    if not todo:
        print("Nothing to do. Output file already contains everything.")
        return

    prompt = get_prompt(args.prompt)
    truncation_count = 0

    # Open in append mode so resume works correctly
    with output_path.open("a", encoding="utf-8") as out:
        for rec in tqdm(todo, desc=approach_tag):
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
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out.flush()  # commit each prediction to disk immediately

    print(f"\nDone. {truncation_count} articles were truncated to fit the context window.")


if __name__ == "__main__":
    main()
