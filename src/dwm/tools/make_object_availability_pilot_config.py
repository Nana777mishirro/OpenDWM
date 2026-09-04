"""Apply the validity-aware prototype profile to a normal CTSD config."""

import argparse
import copy
import json
from pathlib import Path

from dwm.datasets.availability import (
    MODE_ABSENT, MODE_CLEAN, MODE_MIXED, MODE_UNAVAILABLE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an adapter-only CTSD pilot config")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument(
        "--mode", choices=(
            MODE_CLEAN, MODE_UNAVAILABLE, MODE_ABSENT, MODE_MIXED),
        required=True)
    parser.add_argument(
        "--validation-mode", choices=(
            MODE_CLEAN, MODE_UNAVAILABLE, MODE_ABSENT, MODE_MIXED),
        default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--pretrained-model", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)

    def wrap_dataset(dataset_config, mode, manifest):
        dataset_config = copy.deepcopy(dataset_config)
        if dataset_config.get("_class_name") != \
                "dwm.datasets.common.DatasetAdapter":
            raise ValueError(
                "Pilot generator expects DatasetAdapter datasets")
        base_dataset = dataset_config["base_dataset"]
        if base_dataset.get("_class_name") != \
                "dwm.datasets.nuscenes.MotionDataset":
            raise ValueError(
                "Pilot generator currently supports one nuScenes dataset")
        wrapper_config = {
            "_class_name":
                "dwm.datasets.availability.ObjectAvailabilityDataset",
            "base_dataset": base_dataset,
            **profile["dataset_wrapper"],
            "mode": mode,
        }
        if manifest is not None:
            wrapper_config["manifest_path"] = str(manifest.resolve())
        dataset_config["base_dataset"] = wrapper_config
        return dataset_config

    config["training_dataset"] = wrap_dataset(
        config["training_dataset"], args.mode, args.manifest)
    validation_mode = args.validation_mode or args.mode
    config["validation_dataset"] = wrap_dataset(
        config["validation_dataset"], validation_mode,
        args.validation_manifest)

    dataset_config = config["training_dataset"]
    if dataset_config.get("_class_name") != \
            "dwm.datasets.common.DatasetAdapter":
        raise ValueError("Pilot generator expects a DatasetAdapter at training_dataset")

    config["pipeline"]["common_config"].update(profile["common_config"])
    config["pipeline"]["model"][
        "object_availability_adapter_config"] = profile["model_adapter"]
    training_config = config["pipeline"]["training_config"]
    training_config.pop("freezing_pattern", None)
    training_config.update(profile["training_config"])
    config["pipeline"].setdefault("model_load_state_args", {})[
        "strict"] = False

    if args.output_path is not None:
        config["output_path"] = args.output_path
    if args.pretrained_model is not None:
        config["pipeline"]["pretrained_model_name_or_path"] = \
            args.pretrained_model
    if args.checkpoint is not None:
        config["pipeline"]["model_checkpoint_path"] = args.checkpoint
    config["availability_experiment"] = {
        "mode": args.mode,
        "validation_mode": validation_mode,
        "manifest": None if args.manifest is None else str(args.manifest.resolve()),
        "validation_manifest": None if args.validation_manifest is None else
            str(args.validation_manifest.resolve()),
        "semantic_contract": {
            "clean": "current box observed; persistent slot active",
            "temporary_unavailable": (
                "target absent from raster; slot active with last observation"),
            "explicit_absent": "target absent from raster; slot removed/reset",
        },
    }

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_config": str(args.output_config),
        "mode": args.mode,
        "validation_mode": validation_mode,
        "manifest": config["availability_experiment"]["manifest"],
        "adapter_only": training_config[
            "train_object_availability_adapter_only"],
    }, indent=2))


if __name__ == "__main__":
    main()
