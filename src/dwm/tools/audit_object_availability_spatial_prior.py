"""Audit causal spatial-prior overlap with held-out missing-frame GT."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dwm.common
from dwm.datasets.availability import MODE_UNAVAILABLE
from dwm.datasets.common import DatasetAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset-key", choices=("training_dataset", "validation_dataset"),
        default="validation_dataset")
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=28)
    parser.add_argument(
        "--dilation-tokens", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--motion-model",
        choices=("constant_velocity", "constant_acceleration"))
    parser.add_argument("--acceleration-scale", type=float)
    parser.add_argument(
        "--extrapolate-rotation", action=argparse.BooleanOptionalAction,
        default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def downsample(mask, height, width):
    flat = mask.float().reshape(-1, 1, *mask.shape[-2:])
    return F.adaptive_max_pool2d(flat, (height, width)).view(
        *mask.shape[:2], height, width).bool()


def dilate(mask, radius):
    if radius == 0:
        return mask
    flat = mask.float().reshape(-1, 1, *mask.shape[-2:])
    return F.max_pool2d(
        flat, kernel_size=2 * radius + 1, stride=1, padding=radius
    ).view_as(mask).bool()


def overlap(prediction, target):
    intersection = int(torch.count_nonzero(prediction & target))
    prediction_count = int(torch.count_nonzero(prediction))
    target_count = int(torch.count_nonzero(target))
    union = int(torch.count_nonzero(prediction | target))
    return {
        "intersection": intersection,
        "prediction_count": prediction_count,
        "target_count": target_count,
        "union": union,
        "recall": intersection / target_count if target_count else None,
        "precision": intersection / prediction_count
        if prediction_count else None,
        "iou": intersection / union if union else None,
    }


def mean_present(records, key):
    values = [record[key] for record in records if record[key] is not None]
    return sum(values) / len(values) if values else None


def main():
    args = parse_args()
    if any(radius < 0 for radius in args.dilation_tokens):
        raise ValueError("Dilation radii must be non-negative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in config.get("global_state", {}).items():
        dwm.common.global_state[key] = \
            dwm.common.create_instance_from_config(value)
    dataset = dwm.common.create_instance_from_config(config[args.dataset_key])
    if not isinstance(dataset, DatasetAdapter):
        raise TypeError(f"{args.dataset_key} must be a DatasetAdapter")
    wrapper = dataset.base_dataset
    if wrapper.manifest_records is None:
        raise ValueError("Spatial-prior audit requires a manifest")
    if args.motion_model is not None:
        wrapper.motion_model = args.motion_model
    if args.acceleration_scale is not None:
        if not np.isfinite(args.acceleration_scale) or \
                args.acceleration_scale < 0:
            raise ValueError("acceleration-scale must be finite and non-negative")
        wrapper.acceleration_scale = args.acceleration_scale
    if args.extrapolate_rotation is not None:
        wrapper.extrapolate_rotation = args.extrapolate_rotation

    records = []
    leakage_checks = 0
    for index, record in enumerate(wrapper.manifest_records):
        clean, unavailable, absent, record = wrapper.make_paired_items(index)
        start = int(record["missing_start"])
        stop = int(record["missing_end_exclusive"])
        prior = unavailable["object_spatial_prior"]
        target_roi = unavailable["loss_target_roi"]
        if torch.count_nonzero(clean["object_spatial_prior"]) or \
                torch.count_nonzero(absent["object_spatial_prior"]):
            raise AssertionError("Clean/absent priors must be empty")
        outside = torch.cat((prior[:start], prior[stop:]), dim=0)
        if torch.count_nonzero(outside):
            raise AssertionError("Unavailable prior leaked outside interval")

        # Counterfactual leakage check: delete every missing-frame target GT
        # annotation, then reconstruct model-visible inputs.  They must remain
        # byte-identical because only pre-interval observations are legal.
        segment = wrapper._segment(int(record["base_index"]))
        annotations = wrapper._frame_annotations(segment)
        counterfactual = copy.deepcopy(annotations)
        target = record["target_instance_token"]
        for frame in range(start, stop):
            counterfactual[frame].pop(target, None)
        counterfactual_prior = wrapper._build_spatial_prior(
            segment, counterfactual, record, MODE_UNAVAILABLE)
        counterfactual_inputs = wrapper._build_model_inputs(
            segment, counterfactual, record, MODE_UNAVAILABLE)
        if not torch.equal(prior, counterfactual_prior):
            raise AssertionError("Spatial prior depends on missing-frame GT")
        if not torch.equal(
                unavailable["object_box_states"],
                counterfactual_inputs["object_box_states"]):
            raise AssertionError("Box states depend on missing-frame GT")
        leakage_checks += 2

        pixel = overlap(prior[start:stop].bool(), target_roi[start:stop].bool())
        prior_tokens = downsample(
            prior, args.latent_height, args.latent_width)
        target_tokens = downsample(
            target_roi, args.latent_height, args.latent_width)
        token_metrics = {
            str(radius): overlap(
                dilate(prior_tokens, radius)[start:stop],
                target_tokens[start:stop])
            for radius in args.dilation_tokens
        }

        center_errors = []
        for frame in range(start, stop):
            predicted = wrapper._causal_missing_annotation(
                annotations, target, start, frame)
            actual = annotations[frame][target]
            center_errors.append(float(np.linalg.norm(
                np.asarray(predicted["translation"], dtype=np.float64) -
                np.asarray(actual["translation"], dtype=np.float64))))
        records.append({
            "index": index,
            "base_index": int(record["base_index"]),
            "missing_duration": int(record["missing_duration"]),
            "target_category": record["target_category"],
            "causal_sources": list(range(
                start - (3 if wrapper.motion_model ==
                         "constant_acceleration" else 2), start)),
            "interval": [start, stop],
            "center_error_m": {
                "mean": sum(center_errors) / len(center_errors),
                "max": max(center_errors),
                "per_frame": center_errors,
            },
            "pixel": pixel,
            "token": token_metrics,
        })

    summary = {"pixel": {
        metric: mean_present([record["pixel"] for record in records], metric)
        for metric in ("recall", "precision", "iou")
    }, "token": {}}
    for radius in args.dilation_tokens:
        key = str(radius)
        radius_records = [record["token"][key] for record in records]
        summary["token"][key] = {
            metric: mean_present(radius_records, metric)
            for metric in ("recall", "precision", "iou")
        }
        summary["token"][key]["per_duration_recall"] = {
            str(duration): mean_present([
                record["token"][key] for record in records
                if record["missing_duration"] == duration
            ], "recall")
            for duration in sorted({
                record["missing_duration"] for record in records})
        }

    report = {
        "status": "passed",
        "method": {
            "motion_model": wrapper.motion_model,
            "acceleration_scale": wrapper.acceleration_scale,
            "extrapolate_rotation": wrapper.extrapolate_rotation,
        },
        "missing_or_future_gt_is_model_input": False,
        "counterfactual_leakage_checks_passed": leakage_checks,
        "config": str(args.config.resolve()),
        "dataset_key": args.dataset_key,
        "latent_size": [args.latent_height, args.latent_width],
        "record_count": len(records),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "counterfactual_leakage_checks_passed": leakage_checks,
        "record_count": len(records),
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
