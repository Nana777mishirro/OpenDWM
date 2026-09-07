"""Audit target-ROI survival at the diffusion latent resolution."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import dwm.common
from dwm.datasets.common import DatasetAdapter
from dwm.datasets.nuscenes import MotionDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset-key",
        choices=("training_dataset", "validation_dataset"),
        default="training_dataset")
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=28)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in config.get("global_state", {}).items():
        dwm.common.global_state[key] = \
            dwm.common.create_instance_from_config(value)

    dataset = dwm.common.create_instance_from_config(config[args.dataset_key])
    if not isinstance(dataset, DatasetAdapter):
        raise TypeError(f"{args.dataset_key} must be a DatasetAdapter")
    wrapper = dataset.base_dataset
    if wrapper.manifest_records is None:
        raise ValueError("ROI audit requires a manifest-backed dataset")

    records = []
    for index, record in enumerate(wrapper.manifest_records):
        segment = wrapper._segment(int(record["base_index"]))
        annotations = wrapper._frame_annotations(segment)
        target = record["target_instance_token"]
        roi = torch.stack([
            torch.stack([
                wrapper._make_loss_roi(
                    sample_data, annotations[frame].get(target))
                for sample_data in segment[frame]
                if MotionDataset.check_sensor(
                    wrapper.base_dataset.tables,
                    wrapper.base_dataset.indices,
                    sample_data,
                    modality="camera")
            ])
            for frame in range(len(segment))
        ]).float()
        flat_roi = roi.reshape(-1, 1, *roi.shape[-2:])
        latent_roi_nearest = F.interpolate(
            flat_roi,
            size=(args.latent_height, args.latent_width),
            mode="nearest").view(
                *roi.shape[:2], 1, args.latent_height, args.latent_width)
        latent_roi_max = F.adaptive_max_pool2d(
            flat_roi,
            output_size=(args.latent_height, args.latent_width)).view(
                *roi.shape[:2], 1, args.latent_height, args.latent_width)
        start = int(record["missing_start"])
        stop = int(record["missing_end_exclusive"])
        interval_nearest = latent_roi_nearest[start:stop]
        interval_max = latent_roi_max[start:stop]
        nearest_nonzero = int(torch.count_nonzero(interval_nearest))
        max_nonzero = int(torch.count_nonzero(interval_max))
        total = interval_max.numel()
        records.append({
            "index": index,
            "base_index": int(record["base_index"]),
            "missing_duration": int(record["missing_duration"]),
            "target_category": record["target_category"],
            "nearest_nonzero": nearest_nonzero,
            "max_pool_nonzero": max_nonzero,
            "latent_roi_total": total,
            "nearest_fraction": nearest_nonzero / total,
            "max_pool_fraction": max_nonzero / total,
        })

    def summarize(prefix):
        counts = [record[f"{prefix}_nonzero"] for record in records]
        fractions = [record[f"{prefix}_fraction"] for record in records]
        return {
            "empty_record_indices": [
                record["index"] for record in records
                if record[f"{prefix}_nonzero"] == 0
            ],
            "nonzero": {
                "min": min(counts),
                "median": sorted(counts)[len(counts) // 2],
                "max": max(counts),
            },
            "fraction": {
                "mean": sum(fractions) / len(fractions),
                "min": min(fractions),
                "max": max(fractions),
            },
        }

    report = {
        "status": "passed",
        "config": str(args.config.resolve()),
        "dataset_key": args.dataset_key,
        "latent_size": [args.latent_height, args.latent_width],
        "record_count": len(records),
        "nearest": summarize("nearest"),
        "adaptive_max_pool": summarize("max_pool"),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "status", "record_count", "nearest", "adaptive_max_pool")
    }, indent=2))


if __name__ == "__main__":
    main()
