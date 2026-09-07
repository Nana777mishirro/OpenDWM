"""Summarize paired unavailable-vs-absent object ROI metrics."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--baseline-report", type=Path,
        help="Optional full A/B/C report supplying the C baseline.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values)


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def group_summary(rows):
    baseline = mean([row["baseline_roi_mse"] for row in rows])
    trained = mean([row["trained_roi_mse"] for row in rows])
    improvements = [row["improvement"] for row in rows]
    return {
        "count": len(rows),
        "baseline_mean_roi_mse": baseline,
        "trained_mean_roi_mse": trained,
        "mean_absolute_improvement": mean(improvements),
        "relative_improvement_of_macro_mean_percent":
            100 * (baseline - trained) / baseline,
        "mean_per_record_relative_improvement_percent": mean([
            100 * row["improvement"] / row["baseline_roi_mse"]
            for row in rows
        ]),
        "improved_count": sum(row["improvement"] > 0 for row in rows),
        "worsened_count": sum(row["improvement"] < 0 for row in rows),
        "tie_count": sum(row["improvement"] == 0 for row in rows),
    }


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        group: group_summary(group_rows)
        for group, group_rows in sorted(groups.items())
    }


def two_sided_sign_test(improved, worsened):
    count = improved + worsened
    if count == 0:
        return None
    tail = min(improved, worsened)
    probability = sum(
        math.comb(count, k) for k in range(tail + 1)) / (2 ** count)
    return min(1.0, 2 * probability)


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    source = json.loads(args.report.read_text(encoding="utf-8"))
    baseline_source = source if args.baseline_report is None else json.loads(
        args.baseline_report.read_text(encoding="utf-8"))
    baseline_samples = {
        sample["sample_index"]: sample
        for sample in baseline_source["samples"]
    }
    rows = []
    for sample in source["samples"]:
        baseline_sample = baseline_samples.get(sample["sample_index"])
        if baseline_sample is None:
            raise KeyError(
                f"Missing baseline sample {sample['sample_index']}")
        for key in ("sample_seed", "noise_sha256", "common_input_hashes"):
            if sample[key] != baseline_sample[key]:
                raise AssertionError(
                    f"Candidate/baseline mismatch for {key} at sample "
                    f"{sample['sample_index']}")
        metrics = sample["mode_metrics_vs_gt_in_target_interval"]
        baseline_metrics = baseline_sample[
            "mode_metrics_vs_gt_in_target_interval"]
        pairs = sample.get(
            "pairwise_generation_metrics_in_target_interval", {}).get(
                "B_vs_C")
        baseline = baseline_metrics["C_explicit_absent"]["roi_mse"]
        trained = metrics["B_temporary_unavailable"]["roi_mse"]
        rows.append({
            "sample_index": sample["sample_index"],
            "scene_token": sample["record"].get("scene_token", "unknown"),
            "base_index": sample["record"]["base_index"],
            "duration": sample["record"]["missing_duration"],
            "category": sample["record"]["target_category"],
            "baseline_roi_mse": baseline,
            "trained_roi_mse": trained,
            "improvement": baseline - trained,
            "relative_improvement_percent":
                100 * (baseline - trained) / baseline,
            "B_vs_C_roi_mse": pairs["roi_mse"] if pairs else None,
            "B_vs_C_background_mse": pairs["background_mse"]
            if pairs else None,
        })

    overall = group_summary(rows)
    rng = random.Random(args.seed)
    absolute_bootstrap = []
    relative_bootstrap = []
    for _ in range(args.bootstrap_samples):
        resample = [rows[rng.randrange(len(rows))] for _ in rows]
        summary = group_summary(resample)
        absolute_bootstrap.append(summary["mean_absolute_improvement"])
        relative_bootstrap.append(
            summary["relative_improvement_of_macro_mean_percent"])
    absolute_bootstrap.sort()
    relative_bootstrap.sort()
    overall["window_bootstrap_95_percent_ci"] = {
        "resamples": args.bootstrap_samples,
        "seed": args.seed,
        "mean_absolute_improvement": [
            percentile(absolute_bootstrap, 0.025),
            percentile(absolute_bootstrap, 0.975),
        ],
        "relative_improvement_of_macro_mean_percent": [
            percentile(relative_bootstrap, 0.025),
            percentile(relative_bootstrap, 0.975),
        ],
        "caveat": (
            "Windows within a scene are correlated; this interval is "
            "descriptive and is not a scene-level confidence interval."),
    }
    overall["two_sided_sign_test_p"] = two_sided_sign_test(
        overall["improved_count"], overall["worsened_count"])
    if all(row["B_vs_C_roi_mse"] is not None for row in rows):
        roi_change = mean([row["B_vs_C_roi_mse"] for row in rows])
        background_change = mean([
            row["B_vs_C_background_mse"] for row in rows])
        overall["mean_B_vs_C_roi_mse"] = roi_change
        overall["mean_B_vs_C_background_mse"] = background_change
        overall["roi_to_background_change_ratio"] = (
            roi_change / background_change if background_change else None)

    report = {
        "status": "passed",
        "metric_direction": "positive improvement means B ROI MSE < C ROI MSE",
        "baseline_equivalence": (
            "C is a hard adapter no-op. B and C use identical box rasters, "
            "so C is the paired original-model baseline for B."),
        "record_count": len(rows),
        "independent_scene_count": len({
            row["scene_token"] for row in rows}),
        "overall": overall,
        "by_scene": grouped(rows, "scene_token"),
        "by_duration": grouped(rows, "duration"),
        "by_category": grouped(rows, "category"),
        "records": rows,
        "source_report": str(args.report.resolve()),
        "baseline_source_report": str(
            (args.baseline_report or args.report).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "record_count": report["record_count"],
        "independent_scene_count": report["independent_scene_count"],
        "overall": overall,
        "by_scene": report["by_scene"],
        "by_duration": report["by_duration"],
        "by_category": report["by_category"],
    }, indent=2))


if __name__ == "__main__":
    main()
