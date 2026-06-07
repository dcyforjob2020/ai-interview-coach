"""Statistical significance tests for baseline vs. new model feedback scores.

Runs paired statistical tests on the 200-example fixed test set to check
whether the new (SFT+DPO) model's scores differ meaningfully from the baseline.

For every score dimension it reports:
- Baseline / new-model mean and standard deviation.
- Mean paired difference (new - baseline).
- Paired t-test (parametric).
- Wilcoxon signed-rank test (non-parametric; robust to the bimodal,
  ceiling-bound score distribution produced by the judge).
- Cohen's d for paired samples (effect size).
- Win/loss/tie counts and an exact binomial sign test on the non-tie pairs.

Results are printed and written to eval/statistical_tests.json.

Usage:
    python eval/statistical_tests.py
    python eval/statistical_tests.py --baseline ... --new-model ... --output ...

Example output (overall_score, N=200):
    baseline mean=16.861 sd=2.372 | new mean=17.032 sd=1.847 | delta=+0.171
    paired t-test:        t=0.806  p=0.4210
    Wilcoxon signed-rank: W=1868.5 p=0.7042
    Cohen's d (paired):   0.057  (negligible)
    => The difference is NOT statistically significant.
"""

import argparse
import json
import statistics
from pathlib import Path

from scipy import stats


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_BASELINE_PATH = ROOT_DIR / "baseline" / "baseline_scores.jsonl"
DEFAULT_NEW_MODEL_PATH = ROOT_DIR / "newModel" / "new_model_scores.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "eval" / "statistical_tests.json"

SCORE_FIELDS = [
    "technical_correctness",
    "specificity",
    "helpfulness",
    "actionability",
    "interview_coaching_quality",
    "overall_score",
]

ALPHA = 0.05


def read_jsonl(path):
    records = {}
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            record_id = record.get("id")
            if not record_id:
                raise ValueError(f"{path} contains a record without an id at line {line_number}")
            records[record_id] = record
    return records


def cohens_d_paired(diffs):
    """Cohen's d for paired samples: mean(diff) / sd(diff)."""
    sd = statistics.stdev(diffs)
    if sd == 0:
        return 0.0
    return statistics.mean(diffs) / sd


def interpret_d(d):
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def analyze_field(baseline_by_id, new_by_id, matched_ids, field):
    baseline_vals = [float(baseline_by_id[i][field]) for i in matched_ids]
    new_vals = [float(new_by_id[i][field]) for i in matched_ids]
    diffs = [n - b for n, b in zip(new_vals, baseline_vals)]

    t_stat, t_p = stats.ttest_rel(new_vals, baseline_vals)

    # Wilcoxon is undefined when every paired difference is zero.
    if all(d == 0 for d in diffs):
        w_stat, w_p = float("nan"), 1.0
    else:
        w_stat, w_p = stats.wilcoxon(new_vals, baseline_vals)

    d = cohens_d_paired(diffs)

    return {
        "metric": field,
        "n": len(matched_ids),
        "baseline_mean": round(statistics.mean(baseline_vals), 4),
        "baseline_sd": round(statistics.stdev(baseline_vals), 4),
        "new_model_mean": round(statistics.mean(new_vals), 4),
        "new_model_sd": round(statistics.stdev(new_vals), 4),
        "mean_diff": round(statistics.mean(diffs), 4),
        "paired_t": round(float(t_stat), 4),
        "paired_t_p": round(float(t_p), 4),
        "wilcoxon_W": None if w_stat != w_stat else round(float(w_stat), 4),  # NaN check
        "wilcoxon_p": round(float(w_p), 4),
        "cohens_d": round(d, 4),
        "effect_size": interpret_d(d),
        "significant_at_0.05": bool(t_p < ALPHA),
    }


def win_loss_tie(baseline_by_id, new_by_id, matched_ids, field="overall_score"):
    new_wins = baseline_wins = ties = 0
    for i in matched_ids:
        delta = float(new_by_id[i][field]) - float(baseline_by_id[i][field])
        if delta > 0:
            new_wins += 1
        elif delta < 0:
            baseline_wins += 1
        else:
            ties += 1

    non_tie = new_wins + baseline_wins
    # Exact binomial sign test: under H0 wins/losses are 50/50.
    if non_tie > 0:
        sign_p = float(stats.binomtest(new_wins, non_tie, 0.5).pvalue)
    else:
        sign_p = 1.0

    return {
        "new_model_wins": new_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "new_model_win_rate_all": round(new_wins / len(matched_ids), 4),
        "new_model_win_rate_excluding_ties": round(new_wins / non_tie, 4) if non_tie else None,
        "sign_test_p": round(sign_p, 4),
        "sign_test_significant_at_0.05": bool(sign_p < ALPHA),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical tests for baseline vs. new model scores."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--new-model", type=Path, default=DEFAULT_NEW_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    baseline_by_id = read_jsonl(args.baseline)
    new_by_id = read_jsonl(args.new_model)
    matched_ids = sorted(set(baseline_by_id) & set(new_by_id))
    if not matched_ids:
        raise ValueError("No matching ids between baseline and new model scores.")

    field_results = [
        analyze_field(baseline_by_id, new_by_id, matched_ids, field) for field in SCORE_FIELDS
    ]
    wlt = win_loss_tie(baseline_by_id, new_by_id, matched_ids)

    summary = {
        "n_matched": len(matched_ids),
        "alpha": ALPHA,
        "win_loss_tie": wlt,
        "per_dimension": field_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    # Console report.
    print(f"Matched examples: {len(matched_ids)}\n")
    header = (
        f"{'metric':<28}{'base mean(sd)':<16}{'new mean(sd)':<16}"
        f"{'delta':<9}{'t':<8}{'t_p':<9}{'wilcox_p':<10}{'d':<8}sig?"
    )
    print(header)
    print("-" * len(header))
    for r in field_results:
        print(
            f"{r['metric']:<28}"
            f"{r['baseline_mean']:.2f}({r['baseline_sd']:.2f})".ljust(16)
            + f"{r['new_model_mean']:.2f}({r['new_model_sd']:.2f})".ljust(16)
            + f"{r['mean_diff']:+.3f}".ljust(9)
            + f"{r['paired_t']:.2f}".ljust(8)
            + f"{r['paired_t_p']:.4f}".ljust(9)
            + f"{r['wilcoxon_p']:.4f}".ljust(10)
            + f"{r['cohens_d']:.3f}".ljust(8)
            + ("YES" if r["significant_at_0.05"] else "no")
        )

    print(
        f"\nWin/loss/tie (overall_score): "
        f"new={wlt['new_model_wins']}, baseline={wlt['baseline_wins']}, ties={wlt['ties']}"
    )
    print(
        f"Sign test (non-tie pairs): p={wlt['sign_test_p']:.4f} "
        f"({'significant' if wlt['sign_test_significant_at_0.05'] else 'not significant'} at a={ALPHA})"
    )

    overall = next(r for r in field_results if r["metric"] == "overall_score")
    verdict = (
        "IS statistically significant"
        if overall["significant_at_0.05"]
        else "is NOT statistically significant"
    )
    print(
        f"\nVerdict (overall_score): the difference {verdict} "
        f"(paired t p={overall['paired_t_p']:.4f}, Wilcoxon p={overall['wilcoxon_p']:.4f}, "
        f"Cohen's d={overall['cohens_d']:.3f} = {overall['effect_size']})."
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
