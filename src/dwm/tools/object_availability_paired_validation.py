"""Run fixed-seed paired A/B/C generation for object availability.

This is a deliberately small integration evaluator.  It loads one trained
checkpoint, derives clean/unavailable/absent items from the same raw nuScenes
sample, resets the diffusion generator before every mode, and uses the target
ROI only after generation for metric computation.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

import dwm.common
from dwm.datasets.availability import (
    MODE_ABSENT, MODE_CLEAN, MODE_UNAVAILABLE,
)
from dwm.datasets.common import DatasetAdapter
import dwm.utils.preview


MODES = (MODE_CLEAN, MODE_UNAVAILABLE, MODE_ABSENT)
MODE_LABELS = {
    MODE_CLEAN: "A_clean",
    MODE_UNAVAILABLE: "B_temporary_unavailable",
    MODE_ABSENT: "C_explicit_absent",
}
COMMON_KEYS = (
    "vae_images", "hdmap_images", "clip_text", "camera_intrinsics",
    "camera_transforms", "pts", "fps", "crossview_mask",
)
OBJECT_KEYS = (
    "object_class_ids", "object_box_states", "object_availability",
    "object_slot_mask", "object_source_frame_indices", "object_spatial_prior",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-seed paired object-availability generation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-kind", choices=("auto", "full", "adapter"),
        default="auto",
        help="Treat checkpoint as a full model or adapter-only checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-key",
        choices=("training_dataset", "validation_dataset"),
        default="validation_dataset",
        help="Configured dataset split to evaluate (default: validation_dataset)")
    parser.add_argument(
        "--sample-indices", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument(
        "--all-samples", action="store_true",
        help="Evaluate every record in the selected manifest.")
    parser.add_argument(
        "--max-samples", type=int,
        help="Optionally cap the selected records after index expansion.")
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes", choices=MODES, nargs="+", default=list(MODES),
        help="Generation branches to run; input invariants still audit A/B/C.")
    return parser.parse_args()


def resolve_checkpoint_kind(path, requested_kind):
    if requested_kind != "auto":
        return requested_kind
    if path.suffix == ".safetensors":
        return "full"

    payload = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = payload.get("state_dict", payload)
    keys = list(state_dict)
    if keys and all(
        key.startswith("object_availability_adapter.") for key in keys
    ):
        return "adapter"
    return "full"


def tensor_hash(value):
    if isinstance(value, torch.Tensor):
        payload = value.detach().cpu().contiguous().numpy().tobytes()
    else:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def combined_tensor_hash(item, keys):
    digest = hashlib.sha256()
    for key in keys:
        value = item[key]
        digest.update(key.encode("utf-8"))
        digest.update(tensor_hash(value).encode("ascii"))
    return digest.hexdigest()


def apply_dataset_adapter(dataset, item):
    item = copy.deepcopy(item)
    for transform_config in dataset.transform_list:
        if transform_config.get("is_dynamic_transform", False):
            item = transform_config["transform"](item)
        else:
            item[transform_config["new_key"]] = \
                DatasetAdapter.apply_transform(
                    transform_config["transform"],
                    item[transform_config["old_key"]],
                    transform_config.get("stack", True))
    if dataset.pop_list is not None:
        for key in dataset.pop_list:
            item.pop(key, None)
    return item


def masked_mse(prediction, target, mask):
    mask = mask.to(dtype=prediction.dtype).unsqueeze(3)
    denominator = mask.sum() * prediction.shape[3]
    if denominator.item() == 0:
        return None
    return float(
        ((prediction - target).square() * mask).sum().item() /
        denominator.item())


def psnr(mse):
    if mse is None:
        return None
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def interval_metrics(prediction, target, roi, start, stop):
    prediction = prediction[:, start:stop].float()
    target = target[:, start:stop].float()
    roi = roi[:, start:stop].bool()
    roi_mse = masked_mse(prediction, target, roi)
    background_mse = masked_mse(prediction, target, ~roi)
    return {
        "roi_mse": roi_mse,
        "roi_psnr": psnr(roi_mse),
        "background_mse": background_mse,
        "global_mse": float((prediction - target).square().mean()),
    }


def pair_metrics(first, second, roi, start, stop):
    first = first[:, start:stop].float()
    second = second[:, start:stop].float()
    roi = roi[:, start:stop].bool()
    return {
        "roi_mse": masked_mse(first, second, roi),
        "background_mse": masked_mse(first, second, ~roi),
        "global_mse": float((first - second).square().mean()),
        "global_mean_abs": float((first - second).abs().mean()),
    }


def make_aligned_preview_batch(batch, generated_start):
    output = dict(batch)
    for key in ("vae_images", "3dbox_images", "hdmap_images"):
        if key in output:
            output[key] = output[key][:, generated_start:]
    return output


def main():
    args = parse_args()
    if args.inference_steps <= 0:
        raise ValueError("--inference-steps must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    checkpoint_path = str(args.checkpoint.resolve())
    checkpoint_kind = resolve_checkpoint_kind(
        args.checkpoint, args.checkpoint_kind)
    if checkpoint_kind == "adapter":
        config["pipeline"][
            "object_availability_adapter_checkpoint_path"] = checkpoint_path
    else:
        config["pipeline"]["model_checkpoint_path"] = checkpoint_path
    config["pipeline"].setdefault("model_load_state_args", {})["strict"] = False
    inference_config = config["pipeline"]["inference_config"]
    inference_config.update({
        "inference_steps": args.inference_steps,
        "sequence_length_per_iteration": 12,
        "reference_frame_count": 3,
        "generate_frames_for_reference": False,
        "preview_image_size": [224, 128],
    })
    # The raw-Mini overlay uses one static scene-description string instead
    # of a time-indexed prompt list.  Object audit fields below are likewise
    # sample metadata, not temporal tensors.  Autoregressive clipping must
    # therefore leave all of them untouched.
    clip_exceptions = set(inference_config.get(
        "autoregression_data_exception_for_take_sequence", []))
    clip_exceptions.update({
        "crossview_mask", "clip_text", "object_track_hash",
        "object_availability_mode_id", "object_missing_interval",
        "object_pair_seed",
    })
    inference_config[
        "autoregression_data_exception_for_take_sequence"] = \
        sorted(clip_exceptions)

    torch.manual_seed(args.seed)
    device = torch.device(config["device"])
    for key, value in config.get("global_state", {}).items():
        dwm.common.global_state[key] = \
            dwm.common.create_instance_from_config(value)

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=str(args.output), config=config,
        device=device)
    pipeline.model_wrapper.eval()
    dataset = dwm.common.create_instance_from_config(config[args.dataset_key])
    if not isinstance(dataset, DatasetAdapter):
        raise TypeError(f"{args.dataset_key} must be a DatasetAdapter")
    wrapper = dataset.base_dataset
    if not hasattr(wrapper, "make_paired_items"):
        raise TypeError("DatasetAdapter must wrap ObjectAvailabilityDataset")
    dataloader_key = args.dataset_key.replace("dataset", "dataloader")
    dataloader_args = dwm.common.instantiate_config(config[dataloader_key])
    collate_fn = dataloader_args["collate_fn"]
    sample_indices = list(range(len(dataset))) if args.all_samples else \
        list(args.sample_indices)
    if args.max_samples is not None:
        sample_indices = sample_indices[:args.max_samples]

    report = {
        "status": "running",
        "scope": (
            f"{len(sample_indices)} fixed-seed integration records; "
            f"generated modes: {args.modes}"),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_kind": checkpoint_kind,
        "dataset_key": args.dataset_key,
        "sample_indices": sample_indices,
        "base_seed": args.seed,
        "inference_steps": args.inference_steps,
        "reference_frame_count": 3,
        "roi_is_model_input": False,
        "samples": [],
    }

    for sample_index in sample_indices:
        raw_clean, raw_unavailable, raw_absent, record = \
            wrapper.make_paired_items(sample_index)
        items = {
            MODE_CLEAN: apply_dataset_adapter(dataset, raw_clean),
            MODE_UNAVAILABLE: apply_dataset_adapter(dataset, raw_unavailable),
            MODE_ABSENT: apply_dataset_adapter(dataset, raw_absent),
        }
        start = int(record["missing_start"])
        stop = int(record["missing_end_exclusive"])

        common_hashes = {
            key: {mode: tensor_hash(items[mode][key]) for mode in MODES}
            for key in COMMON_KEYS if key in items[MODE_CLEAN]
        }
        failed_common = [
            key for key, hashes in common_hashes.items()
            if len(set(hashes.values())) != 1
        ]
        if failed_common:
            raise AssertionError(
                f"A/B/C common inputs differ for keys: {failed_common}")
        if not torch.equal(
                items[MODE_UNAVAILABLE]["3dbox_images"],
                items[MODE_ABSENT]["3dbox_images"]):
            raise AssertionError("B/C 3D-box rasters must be identical")
        if torch.count_nonzero(
                items[MODE_CLEAN]["object_spatial_prior"]) != 0 or \
                torch.count_nonzero(
                    items[MODE_ABSENT]["object_spatial_prior"]) != 0:
            raise AssertionError("A/C spatial priors must be empty")
        if torch.count_nonzero(
                items[MODE_UNAVAILABLE]["object_spatial_prior"][start:stop]) == 0:
            raise AssertionError("B spatial prior is empty in missing interval")
        raster_changed = int(torch.count_nonzero(
            items[MODE_CLEAN]["3dbox_images"][start:stop] !=
            items[MODE_UNAVAILABLE]["3dbox_images"][start:stop]))
        if raster_changed == 0:
            raise AssertionError("A/B target raster deletion is a no-op")

        sample_seed = args.seed + sample_index
        batches = {
            mode: collate_fn([items[mode]]) for mode in MODES
        }
        sequence_length = batches[MODE_CLEAN]["vae_images"].shape[1]
        view_count = batches[MODE_CLEAN]["vae_images"].shape[2]
        latent_scale = 2 ** (len(pipeline.vae.config.down_block_types) - 1)
        latent_shape = (
            1, pipeline.get_latent_sequence_length(sequence_length), view_count,
            pipeline.vae.config.latent_channels,
            batches[MODE_CLEAN]["vae_images"].shape[-2] // latent_scale,
            batches[MODE_CLEAN]["vae_images"].shape[-1] // latent_scale,
        )
        noise_generator = torch.Generator().manual_seed(sample_seed)
        shared_noise = torch.randn(latent_shape, generator=noise_generator)
        noise_hash = tensor_hash(shared_noise)

        outputs = {}
        cuda_peaks = {}
        for mode in args.modes:
            pipeline.generator.manual_seed(sample_seed)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                generation = pipeline.autoregressive_inference_pipeline(
                    latent_shape, batches[mode], "pt")
            images = generation["images"].detach().cpu()
            output_time = images.shape[0] // view_count
            outputs[mode] = images.unflatten(0, (1, output_time, view_count))
            cuda_peaks[mode] = int(torch.cuda.max_memory_allocated(device))
            del generation
            torch.cuda.empty_cache()
            print(json.dumps({
                "sample_index": sample_index,
                "duration": record["missing_duration"],
                "mode": mode,
                "output_shape": list(outputs[mode].shape),
                "cuda_peak_allocated_bytes": cuda_peaks[mode],
            }), flush=True)

        output_time = outputs[args.modes[0]].shape[1]
        generated_start = sequence_length - output_time
        if generated_start != inference_config["reference_frame_count"]:
            raise AssertionError(
                f"Expected generated output after 3 references, got "
                f"start={generated_start}")
        relative_start = start - generated_start
        relative_stop = stop - generated_start
        if relative_start < 0 or relative_stop > output_time:
            raise AssertionError("Missing interval is outside generated frames")

        target = batches[MODE_CLEAN]["vae_images"][:, generated_start:].cpu()
        roi = batches[MODE_CLEAN]["loss_target_roi"][:, generated_start:].cpu()
        mode_metrics = {
            mode: interval_metrics(
                outputs[mode], target, roi, relative_start, relative_stop)
            for mode in args.modes
        }
        pairwise_metrics = {}
        pair_definitions = {
            "A_vs_B": (MODE_CLEAN, MODE_UNAVAILABLE),
            "A_vs_C": (MODE_CLEAN, MODE_ABSENT),
            "B_vs_C": (MODE_UNAVAILABLE, MODE_ABSENT),
        }
        for name, (first, second) in pair_definitions.items():
            if first in outputs and second in outputs:
                pairwise_metrics[name] = pair_metrics(
                    outputs[first], outputs[second], roi,
                    relative_start, relative_stop)

        sample_directory = args.output / f"sample_{sample_index:03d}"
        sample_directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {MODE_LABELS[k]: v.half() for k, v in outputs.items()},
            sample_directory / "generated_fp16.pt")
        for mode in args.modes:
            aligned_batch = make_aligned_preview_batch(
                batches[mode], generated_start)
            preview = dwm.utils.preview.make_ctsd_preview_tensor(
                outputs[mode].flatten(0, 2), aligned_batch,
                inference_config)
            dwm.utils.preview.save_tensor_to_video(
                str(sample_directory / f"{MODE_LABELS[mode]}.mp4"),
                "libx264", float(batches[mode]["fps"][0]), preview)

        combined_video_name = None
        if len(args.modes) > 1:
            stacked = torch.stack([outputs[mode][0] for mode in args.modes])
            combined_video = stacked.permute(1, 3, 0, 4, 2, 5)\
                .flatten(2, 3).flatten(3, 4)
            combined_video_name = "combined_generated.mp4"
            dwm.utils.preview.save_tensor_to_video(
                str(sample_directory / combined_video_name), "libx264",
                float(batches[MODE_CLEAN]["fps"][0]), combined_video)

        sample_report = {
            "sample_index": sample_index,
            "sample_seed": sample_seed,
            "record": record,
            "generated_frame_interval": [generated_start, sequence_length],
            "noise_sha256": noise_hash,
            "common_input_hashes": {
                key: next(iter(hashes.values()))
                for key, hashes in common_hashes.items()
            },
            "common_inputs_identical": True,
            "B_C_3dbox_raster_identical": True,
            "A_C_spatial_priors_empty": True,
            "B_spatial_prior_nonempty_in_interval": True,
            "A_B_interval_raster_changed_values": raster_changed,
            "object_condition_hashes": {
                MODE_LABELS[mode]: combined_tensor_hash(items[mode], OBJECT_KEYS)
                for mode in MODES
            },
            "generated_output_hashes": {
                MODE_LABELS[mode]: tensor_hash(outputs[mode])
                for mode in args.modes
            },
            "mode_metrics_vs_gt_in_target_interval": {
                MODE_LABELS[mode]: metrics
                for mode, metrics in mode_metrics.items()
            },
            "pairwise_generation_metrics_in_target_interval":
                pairwise_metrics,
            "cuda_peak_allocated_bytes": {
                MODE_LABELS[mode]: value
                for mode, value in cuda_peaks.items()
            },
            "artifacts": {
                "directory": str(sample_directory),
                "generated_tensor": "generated_fp16.pt",
                "videos": [
                    f"{MODE_LABELS[mode]}.mp4" for mode in args.modes
                ] + ([combined_video_name] if combined_video_name else []),
            },
        }
        report["samples"].append(sample_report)
        (sample_directory / "report.json").write_text(
            json.dumps(sample_report, indent=2) + "\n", encoding="utf-8")

    report["status"] = "passed"
    report["checks"] = {
        "all_common_inputs_identical": True,
        "all_B_C_rasters_identical": True,
        "all_A_B_rasters_different_in_interval": True,
        "fixed_seed_reset_per_pair": True,
        "three_object_condition_encodings_distinct": all(
            len(set(sample["object_condition_hashes"].values())) == 3
            for sample in report["samples"]),
        "generated_modes": args.modes,
        "generated_outputs_distinct": all(
            len(set(sample["generated_output_hashes"].values())) ==
            len(args.modes)
            for sample in report["samples"]),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(args.output / "report.json"),
        "checks": report["checks"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
