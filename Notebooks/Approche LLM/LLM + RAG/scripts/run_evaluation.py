"""
Phase 6 — Master evaluation script.

Computes the full benchmark across all prediction files:
    - ROUGE-1, ROUGE-2, ROUGE-L (vs gold reference)
    - BERTScore F1                (vs gold reference)
    - Faithfulness                (vs SOURCE article — needs eval sample)

Outputs:
    data/processed/evaluation_results.json  (raw numbers)
    data/processed/evaluation_results.csv   (table for the report)

Usage:
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.faithfulness import compute_faithfulness


PREDICTION_FILES = [
    "predictions_textrank_tfidf.jsonl",
    "predictions_textrank_embeddings.jsonl",
    "predictions_llm_mistral_basic.jsonl",
    "predictions_llm_mistral_structured.jsonl",
    "predictions_llm_phi3_structured.jsonl",
    "predictions_rag_mistral_top5.jsonl",
    "predictions_rag_phi3_top5.jsonl",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_eval_sample(path: Path) -> dict[str, str]:
    """Returns {id: article} from the eval sample for faithfulness scoring."""
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {r["id"]: r["article"] for r in rows}


def compute_rouge(predictions: list[dict]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for rec in predictions:
        s = scorer.score(rec["reference"], rec["prediction"])
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": sum(r1) / len(r1),
        "rouge2": sum(r2) / len(r2),
        "rougeL": sum(rL) / len(rL),
    }


def compute_bertscore(predictions: list[dict]) -> dict:
    from bert_score import score

    refs = [r["reference"] for r in predictions]
    cands = [r["prediction"] for r in predictions]
    P, R, F = score(cands, refs, lang="en", verbose=False, rescale_with_baseline=False)
    return {
        "bertscore_p": P.mean().item(),
        "bertscore_r": R.mean().item(),
        "bertscore_f": F.mean().item(),
    }


def compute_faithfulness_metric(predictions: list[dict], articles: dict[str, str]) -> dict:
    if not articles:
        return {"faithfulness": None, "note": "eval sample not found"}
    scores = []
    n_unsupported_total = 0
    for rec in predictions:
        src = articles.get(rec["id"])
        if src is None:
            continue
        report = compute_faithfulness(rec["prediction"], src)
        scores.append(report.score)
        n_unsupported_total += len(report.unsupported_entities)
    if not scores:
        return {"faithfulness": None, "note": "no matching ids"}
    return {
        "faithfulness": sum(scores) / len(scores),
        "avg_unsupported_per_doc": n_unsupported_total / len(scores),
    }


def evaluate_one_file(pred_path: Path, articles: dict[str, str], common_ids: set[str] | None = None) -> dict:
    preds = load_jsonl(pred_path)
    if common_ids is not None:
        preds = [p for p in preds if p["id"] in common_ids]

    rouge_scores = compute_rouge(preds)
    try:
        bert_scores = compute_bertscore(preds)
    except Exception as exc:
        print(f"  [warning] BERTScore failed: {exc}")
        bert_scores = {"bertscore_f": None}

    faith_scores = compute_faithfulness_metric(preds, articles)

    elapsed = [p.get("elapsed_seconds", 0) for p in preds]

    return {
        "approach": preds[0]["approach"] if preds else "unknown",
        "n_documents": len(preds),
        "avg_elapsed_s": sum(elapsed) / len(elapsed) if elapsed else 0,
        **rouge_scores,
        **bert_scores,
        **faith_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-dir",
        default="data/processed",
        help="Directory containing the predictions_*.jsonl files.",
    )
    parser.add_argument(
        "--eval-sample",
        default="data/processed/pubmed_eval_sample.jsonl",
        help="Path to the eval sample (needed for faithfulness).",
    )
    parser.add_argument(
        "--fair-comparison",
        action="store_true",
        help="Restrict to document IDs present in ALL prediction files.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    articles = load_eval_sample(Path(args.eval_sample))
    if not articles:
        print(f"WARNING: eval sample not found at {args.eval_sample}.")
        print("Faithfulness scores will be skipped.")
    else:
        print(f"Loaded {len(articles)} source articles for faithfulness scoring.")

    # Find all available prediction files
    found = [pred_dir / f for f in PREDICTION_FILES if (pred_dir / f).exists()]
    if not found:
        print(f"No prediction files found in {pred_dir}.")
        sys.exit(1)

    print(f"\nFound {len(found)} prediction file(s):")
    for f in found:
        print(f"  - {f.name}")

    # For fair comparison, restrict to IDs in all files
    common_ids = None
    if args.fair_comparison and len(found) > 1:
        all_id_sets = [set(p["id"] for p in load_jsonl(f)) for f in found]
        common_ids = set.intersection(*all_id_sets)
        print(f"\nFair-comparison mode: restricting to {len(common_ids)} common IDs.")

    # Evaluate each file
    results = []
    for f in found:
        print(f"\nEvaluating {f.name}...")
        res = evaluate_one_file(f, articles, common_ids=common_ids)
        results.append(res)

    # Print table
    print("\n" + "=" * 110)
    print(
        f"{'Approach':<25} {'N':>4} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'BERT-F':>8} {'Faith':>7} {'Unsupp':>8} {'Time(s)':>9}"
    )
    print("-" * 110)
    for r in results:
        bf = f"{r['bertscore_f'] * 100:>7.2f}" if r.get("bertscore_f") else "    n/a"
        faith = f"{r['faithfulness'] * 100:>6.2f}" if r.get("faithfulness") is not None else "   n/a"
        unsupp = f"{r.get('avg_unsupported_per_doc', 0):>7.2f}" if r.get("faithfulness") is not None else "    n/a"
        print(
            f"{r['approach']:<25} {r['n_documents']:>4} "
            f"{r['rouge1'] * 100:>5.2f}  {r['rouge2'] * 100:>5.2f}  {r['rougeL'] * 100:>5.2f}  "
            f"{bf} {faith} {unsupp} {r['avg_elapsed_s']:>8.1f}"
        )
    print("=" * 110)

    # Save JSON
    out_json = Path("data/processed/evaluation_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Save CSV
    out_csv = Path("data/processed/evaluation_results.csv")
    headers = ["approach", "n_documents", "rouge1", "rouge2", "rougeL",
               "bertscore_f", "faithfulness", "avg_unsupported_per_doc", "avg_elapsed_s"]
    with out_csv.open("w") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [str(r.get(h, "")) for h in headers]
            f.write(",".join(row) + "\n")
    print(f"CSV table saved to {out_csv}")


if __name__ == "__main__":
    main()
