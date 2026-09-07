"""Run exactly one real CTSD batch and audit the availability adapter.

Unlike ``object_availability_smoke`` this loads the configured SD/CTSD
checkpoint, VAE, and text encoders.  It still is not a training launcher: the
configuration must use gradient accumulation so no optimizer update occurs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import dwm.common


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-batch full-checkpoint object availability smoke")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _digest(tensor):
    array = tensor.detach().float().cpu().contiguous().numpy()
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def _find_availability_wrapper(dataset):
    current = dataset
    while hasattr(current, "base_dataset"):
        current = current.base_dataset
        if current.__class__.__name__ == "ObjectAvailabilityDataset":
            return current
    raise RuntimeError("No ObjectAvailabilityDataset in training dataset")


def _slot_encoding(adapter, item, hidden_dim, device):
    class_ids = item["object_class_ids"].unsqueeze(0).to(device)
    box_states = item["object_box_states"].unsqueeze(0).to(
        device=device, dtype=next(adapter.parameters()).dtype)
    scene_context = torch.zeros(
        1, class_ids.shape[1], hidden_dim,
        device=device, dtype=box_states.dtype)
    kwargs = {
        "scene_context": scene_context,
        "class_ids": class_ids,
        "box_states": box_states,
        "availability": item["object_availability"].unsqueeze(0).to(device),
        "slot_mask": item["object_slot_mask"].unsqueeze(0).to(device),
        "source_frame_indices": item[
            "object_source_frame_indices"].unsqueeze(0).to(device),
    }
    with torch.no_grad():
        return adapter.encode_slots(**kwargs).detach()


def _clean_residual(adapter, item, device):
    """Probe the enabled full-size branch on a clean slot at initialization."""
    time = item["object_class_ids"].shape[0]
    dtype = next(adapter.parameters()).dtype
    hidden_states = torch.zeros(
        time, 4, adapter.hidden_dim, device=device, dtype=dtype)
    with torch.no_grad():
        residual = adapter(
            hidden_states,
            batch_size=1,
            sequence_length=time,
            view_count=1,
            class_ids=item["object_class_ids"].unsqueeze(0).to(device),
            box_states=item["object_box_states"].unsqueeze(0).to(
                device=device, dtype=dtype),
            availability=item["object_availability"].unsqueeze(0).to(device),
            slot_mask=item["object_slot_mask"].unsqueeze(0).to(device),
            layer_index=adapter.injection_layers[0],
            source_frame_indices=item[
                "object_source_frame_indices"].unsqueeze(0).to(device),
            spatial_prior=item["object_spatial_prior"][:, :1].unsqueeze(0).to(
                device=device, dtype=dtype),
            spatial_height=2,
            spatial_width=2,
        )
    return float(residual.abs().max())


def main():
    args = parse_args()
    if "LOCAL_RANK" in os.environ:
        raise RuntimeError("Full smoke must run as one process, not torchrun")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["pipeline"]["training_config"]
    if not training.get("train_object_availability_adapter_only", False):
        raise AssertionError("Full smoke requires adapter-only freezing")
    if training.get("gradient_accumulation_steps", 1) <= 1:
        raise AssertionError(
            "Full smoke requires gradient_accumulation_steps > 1 so its "
            "single backward cannot update or clear parameters")
    if training.get("target_roi_loss_weight", 0.0) != 0.0:
        raise AssertionError("ROI weighting must remain disabled in this smoke")
    if config.get("availability_experiment", {}).get("mode") != \
            "temporary_unavailable":
        raise AssertionError(
            "Full smoke is intentionally pinned to temporary_unavailable")

    torch.manual_seed(config["generator_seed"])
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full-checkpoint smoke requires an available CUDA GPU")
    torch.cuda.reset_peak_memory_stats(device)

    for key, value in config.get("global_state", {}).items():
        dwm.common.global_state[key] = \
            dwm.common.create_instance_from_config(value)

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=config.get("output_path"),
        config=config, device=device, resume_from=None)
    trainable = {
        name: parameter for name, parameter in pipeline.model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any(
            not name.startswith("object_availability_adapter.")
            for name in trainable):
        raise AssertionError(
            "Trainable parameters must belong only to the availability adapter")
    adapter = pipeline.model.object_availability_adapter
    projection_max_abs_before = max(
        float(parameter.detach().abs().max())
        for projection in adapter.residual_projections.values()
        for parameter in projection.parameters())
    if projection_max_abs_before != 0.0:
        raise AssertionError("Availability residual projection is not zero-init")

    dataset = dwm.common.create_instance_from_config(config["training_dataset"])
    wrapper = _find_availability_wrapper(dataset)
    clean, unavailable, absent, record = wrapper.make_paired_items(0)
    clean_residual_max_abs = _clean_residual(adapter, clean, device)
    if clean_residual_max_abs != 0.0:
        raise AssertionError("Zero-init adapter changed the clean hidden state")
    hidden_dim = adapter.hidden_dim
    unavailable_encoding = _slot_encoding(
        adapter, unavailable, hidden_dim, device)
    absent_encoding = _slot_encoding(adapter, absent, hidden_dim, device)
    start = record["missing_start"]
    stop = record["missing_end_exclusive"]
    unavailable_interval = unavailable_encoding[:, start:stop]
    absent_interval = absent_encoding[:, start:stop]
    encoding_l2 = float(torch.linalg.vector_norm(
        unavailable_interval.float() - absent_interval.float()))
    if encoding_l2 <= 0 or torch.count_nonzero(absent_interval) != 0:
        raise AssertionError(
            "Temporary-unavailable and explicit-absent encodings collapsed")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        **dwm.common.instantiate_config(config["training_dataloader"]),
        shuffle=False)
    batch = next(iter(dataloader))
    if not torch.all(batch["object_availability_mode_id"] == 1):
        raise AssertionError("Training batch is not temporary-unavailable mode B")
    pipeline.train_step(batch, global_step=0)

    grad_tensors = {}
    for name, parameter in trainable.items():
        if parameter.grad is not None:
            grad_tensors[name] = {
                "nonzero_count": int(torch.count_nonzero(parameter.grad)),
                "l2_norm": float(torch.linalg.vector_norm(
                    parameter.grad.detach().float())),
            }
    nonzero_grad_names = [
        name for name, value in grad_tensors.items()
        if value["nonzero_count"] > 0]
    if not nonzero_grad_names:
        raise AssertionError("No availability-adapter parameter received a gradient")

    projection_max_abs_after = max(
        float(parameter.detach().abs().max())
        for projection in adapter.residual_projections.values()
        for parameter in projection.parameters())
    if projection_max_abs_after != projection_max_abs_before:
        raise AssertionError("One-batch smoke unexpectedly changed model weights")

    report = {
        "status": "passed",
        "scope": "one full CTSD forward/backward; no optimizer step",
        "config": str(args.config.resolve()),
        "pretrained_model": config["pipeline"][
            "pretrained_model_name_or_path"],
        "checkpoint": config["pipeline"].get("model_checkpoint_path"),
        "selected_record": record,
        "batch": {
            "vae_images_shape": list(batch["vae_images"].shape),
            "3dbox_images_shape": list(batch["3dbox_images"].shape),
            "mode_ids": batch["object_availability_mode_id"].tolist(),
        },
        "loss": pipeline.loss_report_list[-1],
        "trainable_parameter_count": sum(i.numel() for i in trainable.values()),
        "trainable_tensor_count": len(trainable),
        "gradient_tensor_count": len(grad_tensors),
        "nonzero_gradient_tensor_count": len(nonzero_grad_names),
        "nonzero_gradient_names": nonzero_grad_names,
        "gradient_details": grad_tensors,
        "zero_init": {
            "residual_projection_max_abs_before": projection_max_abs_before,
            "residual_projection_max_abs_after": projection_max_abs_after,
            "clean_residual_max_abs": clean_residual_max_abs,
            "weights_unchanged_without_optimizer_step": True,
        },
        "unavailable_vs_absent": {
            "interval": [start, stop],
            "l2_difference": encoding_l2,
            "temporary_unavailable_hash": _digest(unavailable_interval),
            "explicit_absent_hash": _digest(absent_interval),
            "temporary_unavailable_nonzero_count": int(
                torch.count_nonzero(unavailable_interval)),
            "explicit_absent_nonzero_count": int(
                torch.count_nonzero(absent_interval)),
        },
        "cuda": {
            "device_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
