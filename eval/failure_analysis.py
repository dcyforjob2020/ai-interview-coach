"""Failure analysis for baseline vs. new model feedback.

Goes beyond the average score to understand *where* and *why* the new
(SFT+DPO) model wins or loses against the baseline on the fixed 200-example
test set. This is the qualitative half of the comparison: the statistical
tests (eval/statistical_tests.py) show the average difference is not
significant, and this script explains why by surfacing the offsetting
regressions and gains.

For every matched test example it joins:
- the score for each model (baseline_scores.jsonl / new_model_scores.jsonl),
- the generated feedback text (baseline_outputs.jsonl / new_model_outputs.jsonl),
- the judge's free-text "reason" for the score.

It then:
1. Labels each example a regression / gain / tie based on overall_score.
2. Heuristically categorizes the *reason* for each regression and gain
   (e.g. technical_error / hallucination / vagueness / verbosity / other),
   using keyword matching over the judge's reason text.
3. Breaks down win/loss/tie by question category, difficulty, and
   student_answer_type to find where training helped or hurt.
4. Writes machine-readable JSON plus two human-readable CSVs of the
   biggest regressions and biggest gains (with feedback + judge reason)
   for direct quotation in the report.

Usage:
    python eval/failure_analysis.py
    python eval/failure_analysis.py --top-k 15
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_BASELINE_SCORES = ROOT_DIR / "baseline" / "baseline_scores.jsonl"
DEFAULT_NEW_SCORES = ROOT_DIR / "newModel" / "new_model_scores.jsonl"
DEFAULT_BASELINE_OUTPUTS = ROOT_DIR / "baseline" / "baseline_outputs.jsonl"
DEFAULT_NEW_OUTPUTS = ROOT_DIR / "newModel" / "new_model_outputs.jsonl"

DEFAULT_JSON_OUT = ROOT_DIR / "eval" / "failure_analysis.json"
DEFAULT_REGRESSIONS_CSV = ROOT_DIR / "eval" / "failure_regressions.csv"
DEFAULT_GAINS_CSV = ROOT_DIR / "eval" / "failure_gains.csv"

# Keyword heuristics applied to the judge's reason text, in priority order.
# The first matching category wins, so order from most to least severe.
REASON_CATEGORIES = [
    ("technical_error", [
        "technical inaccuracy", "technically inaccurate", "technically incorrect",
        "technical error", "factual inaccuracy", "factually incorrect", "incorrect",
        "false", "inaccurate", "wrong", "misleading", "flawed",
    ]),
    ("hallucination", [
        "hallucinat", "claims the student missed", "claimed the student",
        "did not actually", "actually present", "already covered", "fabricat",
    ]),
    ("vagueness", [
        "vague", "generic", "lacks specificity", "not specific", "too general",
        "superficial", "shallow",
    ]),
    ("verbosity", [
        "verbose", "too long", "repetitive", "redundant", "wordy", "unfocused",
    ]),
    ("missing_actionability", [
        "no clear next step", "lacks actionable", "not actionable",
        "no concrete", "lacks concrete",
    ]),
    ("positive", [
        "flawless", "exceptional", "excellent", "no flaws", "zero room",
        "highly specific", "technically accurate", "master-class", "no inaccuracies",
    ]),
]


def read_jsonl(path):
    records = {}
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["id"]] = record
    return records


def categorize_reason(reason):
    """Return the first heuristic category whose keywords appear in the reason."""
    if not reason:
        return "unknown"
    text = reason.lower()
    for category, keywords in REASON_CATEGORIES:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def truncate(text, limit=400):
    if text is None:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def breakdown_by(field, matched_ids, base_scores, new_scores, meta):
    """Win/loss/tie counts grouped by a metadata field (category/difficulty/...)."""
    groups = defaultdict(lambda: {"new_wins": 0, "baseline_wins": 0, "ties": 0, "n": 0})
    for i in matched_ids:
        key = meta[i].get(field, "unknown")
        delta = float(new_scores[i]["overall_score"]) - float(base_scores[i]["overall_score"])
        groups[key]["n"] += 1
        if delta > 0:
            groups[key]["new_wins"] += 1
        elif delta < 0:
            groups[key]["baseline_wins"] += 1
        else:
            groups[key]["ties"] += 1
    # Sort by net (new_wins - baseline_wins) ascending so worst groups surface first.
    return dict(
        sorted(groups.items(), key=lambda kv: kv[1]["new_wins"] - kv[1]["baseline_wins"])
    )


def main():
    parser = argparse.ArgumentParser(description="Failure analysis for baseline vs. new model.")
    parser.add_argument("--baseline-scores", type=Path, default=DEFAULT_BASELINE_SCORES)
    parser.add_argument("--new-scores", type=Path, default=DEFAULT_NEW_SCORES)
    parser.add_argument("--baseline-outputs", type=Path, default=DEFAULT_BASELINE_OUTPUTS)
    parser.add_argument("--new-outputs", type=Path, default=DEFAULT_NEW_OUTPUTS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--regressions-csv", type=Path, default=DEFAULT_REGRESSIONS_CSV)
    parser.add_argument("--gains-csv", type=Path, default=DEFAULT_GAINS_CSV)
    parser.add_argument("--top-k", type=int, default=20, help="rows per CSV")
    args = parser.parse_args()

    base_scores = read_jsonl(args.baseline_scores)
    new_scores = read_jsonl(args.new_scores)
    base_outputs = read_jsonl(args.baseline_outputs)
    new_outputs = read_jsonl(args.new_outputs)

    matched_ids = sorted(set(base_scores) & set(new_scores))
    if not matched_ids:
        raise ValueError("No matching ids between baseline and new model scores.")

    # Metadata (question category/difficulty/student type) comes from outputs files.
    meta = {i: new_outputs.get(i, base_outputs.get(i, {})) for i in matched_ids}

    regressions = []  # new < baseline
    gains = []        # new > baseline
    for i in matched_ids:
        b = float(base_scores[i]["overall_score"])
        n = float(new_scores[i]["overall_score"])
        delta = n - b
        row = {
            "id": i,
            "category": meta[i].get("category", ""),
            "difficulty": meta[i].get("difficulty", ""),
            "student_answer_type": meta[i].get("student_answer_type", ""),
            "baseline_score": b,
            "new_score": n,
            "delta": round(delta, 4),
            "baseline_reason_category": categorize_reason(base_scores[i].get("reason")),
            "new_reason_category": categorize_reason(new_scores[i].get("reason")),
            "new_judge_reason": truncate(new_scores[i].get("reason")),
            "baseline_judge_reason": truncate(base_scores[i].get("reason")),
            "new_feedback": truncate(new_outputs.get(i, {}).get("new_model_feedback"), 600),
            "baseline_feedback": truncate(base_outputs.get(i, {}).get("baseline_feedback"), 600),
        }
        if delta < 0:
            regressions.append(row)
        elif delta > 0:
            gains.append(row)

    regressions.sort(key=lambda r: r["delta"])          # most negative first
    gains.sort(key=lambda r: r["delta"], reverse=True)  # most positive first

    # Why did the new model lose? Categorize the judge's reason on regressions.
    regression_reason_counts = Counter(r["new_reason_category"] for r in regressions)
    gain_reason_counts = Counter(r["new_reason_category"] for r in gains)

    summary = {
        "n_matched": len(matched_ids),
        "counts": {
            "regressions": len(regressions),
            "gains": len(gains),
            "ties": len(matched_ids) - len(regressions) - len(gains),
        },
        "regression_reason_categories": dict(regression_reason_counts.most_common()),
        "gain_reason_categories": dict(gain_reason_counts.most_common()),
        "breakdown_by_category": breakdown_by("category", matched_ids, base_scores, new_scores, meta),
        "breakdown_by_difficulty": breakdown_by("difficulty", matched_ids, base_scores, new_scores, meta),
        "breakdown_by_student_answer_type": breakdown_by(
            "student_answer_type", matched_ids, base_scores, new_scores, meta
        ),
        "top_regressions": [
            {k: r[k] for k in ("id", "category", "difficulty", "baseline_score", "new_score", "delta", "new_reason_category")}
            for r in regressions[: args.top_k]
        ],
        "top_gains": [
            {k: r[k] for k in ("id", "category", "difficulty", "baseline_score", "new_score", "delta", "new_reason_category")}
            for r in gains[: args.top_k]
        ],
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    csv_fields = [
        "id", "category", "difficulty", "student_answer_type",
        "baseline_score", "new_score", "delta",
        "new_reason_category", "new_judge_reason",
        "baseline_feedback", "new_feedback",
    ]

    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows[: args.top_k])

    write_csv(args.regressions_csv, regressions)
    write_csv(args.gains_csv, gains)

    # Console report.
    print(f"Matched examples: {len(matched_ids)}")
    print(
        f"Regressions (new<base): {len(regressions)} | "
        f"Gains (new>base): {len(gains)} | "
        f"Ties: {summary['counts']['ties']}"
    )
    print("\nWhy the new model LOST (judge reason category on regressions):")
    for cat, count in regression_reason_counts.most_common():
        print(f"  {cat:<24}{count}")
    print("\nWhy the new model WON (judge reason category on gains):")
    for cat, count in gain_reason_counts.most_common():
        print(f"  {cat:<24}{count}")

    print("\nWin/loss/tie by student_answer_type (sorted worst-first for new model):")
    print(f"  {'type':<26}{'new':>5}{'base':>6}{'tie':>5}{'n':>5}")
    for key, g in summary["breakdown_by_student_answer_type"].items():
        print(f"  {str(key):<26}{g['new_wins']:>5}{g['baseline_wins']:>6}{g['ties']:>5}{g['n']:>5}")

    print("\nTop 5 regressions (biggest score drops):")
    for r in regressions[:5]:
        print(f"  {r['id']}  {r['baseline_score']:.1f} -> {r['new_score']:.1f}  "
              f"({r['new_reason_category']})  [{r['category']}]")

    print(f"\nWrote {args.json_output}")
    print(f"Wrote {args.regressions_csv}")
    print(f"Wrote {args.gains_csv}")


if __name__ == "__main__":
    main()
