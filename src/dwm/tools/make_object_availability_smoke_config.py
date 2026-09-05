"""Convert an availability pilot config into a one-GPU nuScenes smoke config.

The published CTSD recipe targets a private 12 Hz dataset and a 16-GPU FSDP
mesh.  This tool changes only deployment/data settings needed for a bounded
full-model check.  It expects the semantic availability wrapper to have
already been added by ``make_object_availability_pilot_config``.
"""

import argparse
import copy
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a single-GPU raw-nuScenes availability config")
    parser.add_argument("--input-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="v1.0-mini")
    parser.add_argument("--training-split", default="mini_train")
    parser.add_argument("--validation-split", default="mini_val")
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument(
        "--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
        default=(128, 224))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def _availability_wrapper(dataset_config):
    base_dataset = dataset_config["base_dataset"]
    if base_dataset.get("_class_name") != \
            "dwm.datasets.availability.ObjectAvailabilityDataset":
        raise ValueError(
            "Input must first pass through "
            "make_object_availability_pilot_config")
    motion_dataset = base_dataset["base_dataset"]
    if motion_dataset.get("_class_name") != \
            "dwm.datasets.nuscenes.MotionDataset":
        raise ValueError("Smoke overlay currently supports raw nuScenes only")
    return base_dataset, motion_dataset


def _configure_dataset(
    dataset_config, split, args, manifest, image_size
):
    dataset_config = copy.deepcopy(dataset_config)
    wrapper, motion = _availability_wrapper(dataset_config)
    motion.update({
        "fs": {
            "_class_name": "dwm.common.get_state",
            "key": "nuscenes_fs",
        },
        "dataset_name": args.dataset_name,
        "split": split,
        "sequence_length": args.sequence_length,
        "fps_stride_tuples": [[0, 1]],
        "keyframe_only": True,
        "enable_synchronization_check": False,
        "enable_scene_description": True,
        "enable_camera_transforms": True,
    })
    motion.pop("image_description_settings", None)
    wrapper["missing_durations"] = [1]
    wrapper["roi_size"] = image_size
    if manifest is None:
        wrapper.pop("manifest_path", None)
    else:
        wrapper["manifest_path"] = str(manifest.resolve())

    for transform in dataset_config["transform_list"]:
        old_key = transform["old_key"]
        if old_key in ("images", "3dbox_images", "hdmap_images"):
            # Remove stochastic augmentation from this reproducibility smoke.
            transform["transform"] = {
                "_class_name": "torchvision.transforms.Compose",
                "transforms": [
                    {
                        "_class_name": "torchvision.transforms.Resize",
                        "size": image_size,
                    },
                    {"_class_name": "torchvision.transforms.ToTensor"},
                ],
            }
        elif old_key == "image_description":
            transform["old_key"] = "scene_description"

    dataset_config["pop_list"] = ["images", "scene_description"]
    return dataset_config


def main():
    args = parse_args()
    if args.sequence_length < 6:
        raise ValueError(
            "sequence_length must be >= 6 for 3-before + 1-missing + 2-after")
    image_size = [int(i) for i in args.image_size]
    config = json.loads(args.input_config.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)

    config["global_state"] = {
        "nuscenes_fs": {
            "_class_name": "dwm.fs.dirfs.DirFileSystem",
            "path": str(args.nuscenes_root.resolve()),
        }
    }
    config["training_dataset"] = _configure_dataset(
        config["training_dataset"], args.training_split, args,
        args.manifest, image_size)
    config["validation_dataset"] = _configure_dataset(
        config["validation_dataset"], args.validation_split, args,
        args.validation_manifest, image_size)

    pipeline = config["pipeline"]
    common = pipeline["common_config"]
    common["distribution_framework"] = "ddp"
    common["autocast"] = {
        "device_type": "cuda",
        "dtype": {
            "_class_name": "get_class",
            "class_name": "torch.float16",
        },
    }
    common["memory_efficient_batch"] = 6
    common["print_load_state_info"] = True
    common.pop("ddp_wrapper_settings", None)
    common.pop("t5_fsdp_wrapper_settings", None)

    training = pipeline["training_config"]
    training.update({
        "text_prompt_condition_ratio": 1.0,
        "3dbox_condition_ratio": 1.0,
        "hdmap_condition_ratio": 1.0,
        "reference_frame_count": 3,
        "generation_task_ratio": 0.0,
        "image_generation_ratio": 0.0,
        "all_reference_visible_ratio": 1.0,
        "enable_grad_scaler": False,
        # One train_step performs backward but intentionally does not step or
        # zero the optimizer, so the smoke runner can audit adapter gradients.
        "gradient_accumulation_steps": 2,
        "target_roi_loss_weight": 0.0,
    })
    pipeline["metrics"] = {}

    config["training_dataloader"] = {
        "batch_size": 1,
        "num_workers": 0,
        "collate_fn": {
            "_class_name": "dwm.datasets.common.CollateFnIgnoring",
            "keys": ["clip_text"],
        },
    }
    config["validation_dataloader"] = copy.deepcopy(
        config["training_dataloader"])
    config.pop("preview_dataloader", None)
    config["data_shuffle"] = False
    config["train_epochs"] = 1
    if args.output_path is not None:
        config["output_path"] = str(args.output_path.resolve())

    config.setdefault("availability_experiment", {})["full_smoke"] = {
        "single_process": True,
        "batch_size": 1,
        "sequence_length": args.sequence_length,
        "image_size": image_size,
        "missing_durations": [1],
        "optimizer_step": False,
        "raw_dataset": args.dataset_name,
        "training_split": args.training_split,
        "validation_split": args.validation_split,
    }

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_config": str(args.output_config),
        "nuscenes_root": str(args.nuscenes_root.resolve()),
        "sequence_length": args.sequence_length,
        "image_size": image_size,
        "manifest": None if args.manifest is None else
            str(args.manifest.resolve()),
        "single_gpu": True,
        "optimizer_step": False,
    }, indent=2))


if __name__ == "__main__":
    main()
