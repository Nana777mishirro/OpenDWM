"""CPU smoke/audit for the validity-aware selected-object prototype.

This deliberately uses a tiny DiT with the real OpenDWM forward path.  It does
not load the 17 GB CTSD checkpoint and never starts a training loop.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import dwm.datasets.common
from dwm.datasets.availability import ObjectAvailabilityDataset
from dwm.datasets.nuscenes import MotionDataset
from dwm.fs.dirfs import DirFileSystem
from dwm.models.crossview_temporal_dit import \
    DiTCrossviewTemporalConditionModel
from dwm.models.object_availability_adapter import ObjectAvailabilityAdapter
from dwm.pipelines.ctsd import CrossviewTemporalSD


CAMERAS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")
MODEL_OBJECT_KEYS = (
    "object_class_ids", "object_box_states", "object_availability",
    "object_slot_mask", "object_source_frame_indices",
    "object_spatial_prior")
CONTROLLED_KEYS = {
    "3dbox_images", "loss_target_roi", "object_class_ids",
    "object_box_states", "object_availability", "object_slot_mask",
    "object_source_frame_indices", "object_spatial_prior",
    "object_state_codes",
    "object_track_hash", "object_availability_mode_id",
    "object_missing_interval", "object_pair_seed",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the object-availability CPU smoke test")
    parser.add_argument(
        "--nuscenes-root", type=Path,
        default=Path("/home/nvidia/Datasets/nuscenes"))
    parser.add_argument("--dataset-name", default="v1.0-trainval")
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--sequence-length", type=int, default=13)
    parser.add_argument("--max-scan", type=int, default=500)
    parser.add_argument(
        "--profile", type=Path,
        default=Path(
            "configs/experimental/object_availability/pilot_profile.json"))
    parser.add_argument(
        "--report-output", type=Path,
        default=Path("reports/object_availability_smoke_report.json"))
    parser.add_argument(
        "--manifest-output", type=Path,
        default=Path("reports/object_availability_smoke_manifest.json"))
    return parser.parse_args()


def update_hash(hasher, value):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    elif isinstance(value, Image.Image):
        hasher.update(value.mode.encode())
        hasher.update(str(value.size).encode())
        hasher.update(value.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            hasher.update(str(key).encode())
            update_hash(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            update_hash(hasher, item)
    else:
        hasher.update(repr(value).encode())


def digest(value):
    hasher = hashlib.sha256()
    update_hash(hasher, value)
    return hasher.hexdigest()


def frame_digests(frames):
    return [digest(frame) for frame in frames]


def tiny_model(adapter=False):
    kwargs = dict(
        patch_size=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        in_channels=4,
        out_channels=4,
        sample_size=8,
        pos_embed_max_size=8,
        joint_attention_dim=32,
        caption_projection_dim=16,
        pooled_projection_dim=16,
        dual_attention_layers=(),
        qk_norm=None,
        enable_crossview=False,
        enable_temporal=False,
        crossview_block_layers=[],
        temporal_block_layers=[],
    )
    if adapter:
        kwargs["object_availability_adapter_config"] = {
            "num_classes": 12,
            "class_embedding_dim": 4,
            "slot_dim": 8,
            "injection_layers": [0],
            "use_velocity": False,
            "validate_inputs": True,
        }
    return DiTCrossviewTemporalConditionModel(**kwargs)


def object_kwargs(item):
    return {key: item[key].unsqueeze(0) for key in MODEL_OBJECT_KEYS}


def main():
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    expected_modes = {
        "A": "clean",
        "B": "temporary_unavailable",
        "C": "explicit_absent",
    }
    assert profile["experiment_modes"] == expected_modes

    dataset = MotionDataset(
        fs=DirFileSystem(path=str(args.nuscenes_root)),
        dataset_name=args.dataset_name,
        sequence_length=args.sequence_length,
        fps_stride_tuples=[(0, 1)],
        split=args.split,
        sensor_channels=list(CAMERAS),
        keyframe_only=True,
        enable_synchronization_check=False,
        enable_scene_description=True,
        enable_camera_transforms=True,
        enable_ego_transforms=True,
        _3dbox_image_settings={},
    )
    wrapper = ObjectAvailabilityDataset(
        dataset,
        mode="temporary_unavailable",
        missing_durations=tuple(profile["dataset_wrapper"][
            "missing_durations"]),
        min_observed_before=profile["dataset_wrapper"][
            "min_observed_before"],
        min_observed_after=profile["dataset_wrapper"][
            "min_observed_after"],
        seed=profile["dataset_wrapper"]["seed"],
        roi_size=(64, 112),
        strict=True,
        debug_assertions=True,
    )

    record = None
    selected_index = None
    for base_index in range(min(len(dataset), args.max_scan)):
        record = wrapper.find_record(base_index)
        if record is not None:
            selected_index = base_index
            break
    if record is None:
        raise RuntimeError("No eligible real nuScenes track found in scan range")

    base_item = dataset[selected_index]
    # Exercise pair invariance for HDMap and per-view text even though the
    # local smoke dataset intentionally skips expensive map/caption loading.
    base_item["hdmap_images"] = [[
        Image.new("RGB", (8, 8), color=(frame, view, 7))
        for view in range(len(CAMERAS))]
        for frame in range(args.sequence_length)]
    base_item["image_description"] = [[
        f"paired invariant text view {view}"
        for view in range(len(CAMERAS))]
        for _ in range(args.sequence_length)]
    clean, unavailable, absent, paired_record = wrapper.make_paired_items(
        selected_index, base_item=base_item)
    assert paired_record == record
    assert digest(base_item["3dbox_images"]) == digest(clean["3dbox_images"])
    start = record["missing_start"]
    stop = record["missing_end_exclusive"]

    invariant_hashes = {}
    for key in sorted(set(clean).difference(CONTROLLED_KEYS)):
        hashes = [digest(item[key]) for item in (clean, unavailable, absent)]
        if len(set(hashes)) != 1:
            raise AssertionError(f"Paired invariant changed: {key}")
        invariant_hashes[key] = hashes[0]

    clean_layout = frame_digests(clean["3dbox_images"])
    unavailable_layout = frame_digests(unavailable["3dbox_images"])
    absent_layout = frame_digests(absent["3dbox_images"])
    assert clean_layout[:start] == unavailable_layout[:start]
    assert clean_layout[stop:] == unavailable_layout[stop:]
    assert unavailable_layout == absent_layout
    changed_interval_frames = [
        frame for frame in range(start, stop)
        if clean_layout[frame] != unavailable_layout[frame]]
    if len(changed_interval_frames) != stop - start:
        raise AssertionError(
            "Target raster was not removed in every unavailable frame")

    torch.manual_seed(record["pair_seed"])
    noise = torch.randn(1, args.sequence_length, 1, 4, 8, 8)
    noise_hash = digest(noise)
    timestep = torch.ones(
        1, args.sequence_length, 1, dtype=torch.long)
    encoder_hidden_states = torch.randn(
        1, args.sequence_length, 1, 5, 32)
    pooled_projections = torch.randn(
        1, args.sequence_length, 1, 16)

    torch.manual_seed(101)
    baseline_model = tiny_model(adapter=False)
    torch.manual_seed(102)
    adapter_model = tiny_model(adapter=True)
    missing_keys, unexpected_keys = adapter_model.load_state_dict(
        baseline_model.state_dict(), strict=False)
    if unexpected_keys or not missing_keys:
        raise AssertionError("Unexpected checkpoint compatibility result")

    # Exercise CTSD's condition assembly and prove the loss-only ROI is not
    # forwarded, while all five object-state tensors are.
    condition_probe_batch = {
        key: unavailable[key].unsqueeze(0) for key in MODEL_OBJECT_KEYS}
    condition_probe_batch["pts"] = torch.zeros(
        1, args.sequence_length, 1)
    condition_probe_batch["loss_target_roi"] = unavailable[
        "loss_target_roi"].unsqueeze(0)
    assembled_conditions = CrossviewTemporalSD.get_conditions(
        adapter_model, None, None,
        {"object_availability_enabled": True},
        noise.shape, condition_probe_batch, torch.device("cpu"),
        torch.float32)
    assert "loss_target_roi" not in assembled_conditions
    assert all(key in assembled_conditions for key in MODEL_OBJECT_KEYS)

    baseline_model.eval()
    adapter_model.eval()
    with torch.no_grad():
        baseline_output = baseline_model(
            noise, timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections)[0][0]
        clean_output = adapter_model(
            noise, timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            **object_kwargs(clean))[0][0]
        initialization_max_abs_diff = float(
            (baseline_output - clean_output).abs().max())
        if initialization_max_abs_diff != 0:
            raise AssertionError("Zero-init adapter changed the clean output")

        adapter_model(
            noise, timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            **object_kwargs(unavailable))
        unavailable_encoding = adapter_model.object_availability_adapter\
            .last_slot_encoding.clone()
        adapter_model(
            noise, timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            **object_kwargs(absent))
        absent_encoding = adapter_model.object_availability_adapter\
            .last_slot_encoding.clone()

    interval_unavailable_encoding = unavailable_encoding[:, start:stop]
    interval_absent_encoding = absent_encoding[:, start:stop]
    encoding_l2_difference = float(torch.linalg.vector_norm(
        interval_unavailable_encoding - interval_absent_encoding))
    if encoding_l2_difference <= 0:
        raise AssertionError("Unavailable and absent encodings are identical")
    assert torch.count_nonzero(interval_absent_encoding) == 0

    adapter_model.requires_grad_(False)
    adapter_model.object_availability_adapter.requires_grad_(True)
    adapter_model.train()
    prediction = adapter_model(
        noise, timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        **object_kwargs(unavailable))[0][0]
    loss = torch.nn.functional.mse_loss(prediction, torch.zeros_like(prediction))
    loss.backward()
    trainable_parameters = [
        (name, parameter)
        for name, parameter in adapter_model.named_parameters()
        if parameter.requires_grad]
    gradient_parameters = [
        name for name, parameter in trainable_parameters
        if parameter.grad is not None]
    nonzero_gradient_parameters = [
        name for name, parameter in trainable_parameters
        if parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0]
    if not nonzero_gradient_parameters:
        raise AssertionError("No new parameter received a non-zero gradient")

    # With a strict zero output projection, the first step opens that
    # projection; the following backward must reach the upstream memory/token
    # parameters as well.
    optimizer = torch.optim.SGD(
        [parameter for _, parameter in trainable_parameters], lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()
    second_prediction = adapter_model(
        noise, timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        **object_kwargs(unavailable))[0][0]
    second_loss = torch.nn.functional.mse_loss(
        second_prediction, torch.zeros_like(second_prediction))
    second_loss.backward()
    post_open_nonzero_gradient_parameters = [
        name for name, parameter in trainable_parameters
        if parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0]
    if len(post_open_nonzero_gradient_parameters) <= len(
            nonzero_gradient_parameters):
        raise AssertionError("Zero projection did not open upstream gradients")

    pilot_adapter = ObjectAvailabilityAdapter(
        hidden_dim=1536, **profile["model_adapter"])
    pilot_trainable_parameter_count = sum(
        parameter.numel() for parameter in pilot_adapter.parameters()
        if parameter.requires_grad)

    source = unavailable["object_source_frame_indices"][start:stop, 0]
    segment = wrapper._segment(selected_index)
    annotations = wrapper._frame_annotations(segment)
    reference_world_from_ego = dwm.datasets.common.get_transform(
        segment[0][0]["rotation"], segment[0][0]["translation"])
    reference_from_world = np.linalg.inv(reference_world_from_ego)
    expected_missing_boxes = torch.stack([
        wrapper._normalized_box_state(
            wrapper._causal_missing_annotation(
                annotations, record["target_instance_token"], start, frame),
            reference_from_world)
        for frame in range(start, stop)
    ]).unsqueeze(1)
    leakage_assertions = {
        "unavailable_uses_strictly_past_source": bool(torch.all(
            source < torch.arange(start, stop))),
        "unavailable_source_is_last_observed": bool(torch.all(source == start - 1)),
        "unavailable_box_is_causal_extrapolation": bool(torch.equal(
            unavailable["object_box_states"][start:stop],
            expected_missing_boxes)),
        "absent_slot_removed": bool(torch.all(
            ~absent["object_slot_mask"][start:stop])),
        "absent_source_removed": bool(torch.all(
            absent["object_source_frame_indices"][start:stop] == -1)),
        "clean_spatial_prior_empty": bool(
            torch.count_nonzero(clean["object_spatial_prior"]) == 0),
        "absent_spatial_prior_empty": bool(
            torch.count_nonzero(absent["object_spatial_prior"]) == 0),
        "unavailable_spatial_prior_nonempty": bool(torch.count_nonzero(
            unavailable["object_spatial_prior"][start:stop]) > 0),
        "roi_not_in_model_kwargs": "loss_target_roi" not in MODEL_OBJECT_KEYS,
        "spatial_prior_is_model_input":
            "object_spatial_prior" in MODEL_OBJECT_KEYS,
        "velocity_not_available_or_used": "object_velocities" not in unavailable,
    }
    if not all(leakage_assertions.values()):
        raise AssertionError(f"Leakage assertion failed: {leakage_assertions}")

    manifest = {
        "schema_version": 1,
        "records": [record],
        "constraints": {
            "single_selected_object": True,
            "missing_durations_supported": [1, 3, 5, 7],
            "min_observed_before": 3,
            "min_observed_after": 2,
            "velocity_input": False,
            "motion_prior": "constant_velocity_from_last_two_observations",
        },
    }
    report = {
        "status": "passed",
        "scope": "real nuScenes dataset + tiny integrated CPU DiT; no checkpoint load",
        "config_parse": {
            "profile": str(args.profile),
            "experiment_modes": expected_modes,
        },
        "dataset": {
            "root": str(args.nuscenes_root),
            "dataset_name": args.dataset_name,
            "split": args.split,
            "dataset_item_count": len(dataset),
            "selected_record": record,
            "changed_interval_frames": changed_interval_frames,
            "default_clean_layout_unchanged": True,
            "clean_layout_hashes": clean_layout,
            "temporary_unavailable_layout_hashes": unavailable_layout,
            "explicit_absent_layout_hashes": absent_layout,
        },
        "pair_invariance": {
            "invariant_keys": sorted(invariant_hashes),
            "hashes": invariant_hashes,
            "shared_noise_seed": record["pair_seed"],
            "shared_noise_hash": noise_hash,
        },
        "leakage_assertions": leakage_assertions,
        "model_smoke": {
            "forward_output_shape": list(clean_output.shape),
            "loss": float(loss.detach()),
            "clean_initialization_max_abs_diff": initialization_max_abs_diff,
            "trainable_parameter_count": sum(
                parameter.numel() for _, parameter in trainable_parameters),
            "pilot_hidden_dim_1536_trainable_parameter_count":
                pilot_trainable_parameter_count,
            "trainable_parameter_tensor_count": len(trainable_parameters),
            "gradient_parameter_tensor_count": len(gradient_parameters),
            "nonzero_gradient_parameter_tensor_count": len(
                nonzero_gradient_parameters),
            "nonzero_gradient_parameters": nonzero_gradient_parameters,
            "post_zero_projection_step_nonzero_gradient_parameter_tensor_count":
                len(post_open_nonzero_gradient_parameters),
            "post_zero_projection_step_nonzero_gradient_parameters":
                post_open_nonzero_gradient_parameters,
            "checkpoint_missing_keys_are_adapter_only": all(
                key.startswith("object_availability_adapter.")
                for key in missing_keys),
        },
        "encoding_separation": {
            "temporary_unavailable_hash": digest(
                interval_unavailable_encoding),
            "explicit_absent_hash": digest(interval_absent_encoding),
            "l2_difference": encoding_l2_difference,
            "temporary_unavailable_nonzero_count": int(torch.count_nonzero(
                interval_unavailable_encoding)),
            "explicit_absent_nonzero_count": int(torch.count_nonzero(
                interval_absent_encoding)),
        },
        "condition_assembly": {
            "model_condition_keys": sorted(assembled_conditions),
            "loss_target_roi_forwarded": False,
        },
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(args.report_output),
        "manifest": str(args.manifest_output),
        "selected_record": record,
        "clean_initialization_max_abs_diff": initialization_max_abs_diff,
        "trainable_parameter_count": report["model_smoke"][
            "trainable_parameter_count"],
        "pilot_hidden_dim_1536_trainable_parameter_count":
            pilot_trainable_parameter_count,
        "encoding_l2_difference": encoding_l2_difference,
    }, indent=2))


if __name__ == "__main__":
    main()
