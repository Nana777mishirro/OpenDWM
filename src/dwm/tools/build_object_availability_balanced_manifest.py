"""Build a scene/duration-balanced object-availability manifest."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import dwm.common
from dwm.datasets.common import DatasetAdapter
from dwm.datasets.availability import _stable_int64
from dwm.datasets.nuscenes import MotionDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset-key", choices=("training_dataset", "validation_dataset"),
        default="validation_dataset")
    parser.add_argument(
        "--records-per-duration-per-scene", type=int, default=2)
    parser.add_argument(
        "--category-prefix", action="append", default=None,
        help=("Restrict targets to one or more category prefixes; repeat the "
              "argument for multiple prefixes."))
    parser.add_argument(
        "--min-source-displacement-m", type=float, default=0.0,
        help=("Require at least this displacement between the final two "
              "observed target states."))
    parser.add_argument(
        "--scene-partition-count", type=int, default=1,
        help=("Deterministically divide sorted scene tokens into this many "
              "disjoint partitions."))
    parser.add_argument(
        "--scene-partition-index", type=int, default=0,
        help="Zero-based scene partition to include.")
    parser.add_argument(
        "--skip-insufficient-scenes", action="store_true",
        help=("Skip scenes without enough candidates for every duration "
              "instead of failing the entire manifest build."))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scene_token(wrapper, base_index):
    first_sample_data = wrapper._segment(base_index)[0][0]
    sample = MotionDataset.query(
        wrapper.base_dataset.tables, wrapper.base_dataset.indices,
        "sample", first_sample_data["sample_token"])
    return sample["scene_token"]


def fast_candidates_for_base(
    wrapper, base_index, durations, category_prefixes=None,
    min_source_displacement_m=0.0,
):
    """Find candidates while caching target/camera projection visibility."""
    segment = wrapper._segment(base_index)
    annotations = wrapper._frame_annotations(segment)
    frame_count = len(segment)
    visibility = {}
    category_cache = {}

    def category(target):
        if target not in category_cache:
            category_cache[target] = wrapper._category_name(target)
        return category_cache[target]

    def visible(frame, target):
        key = (frame, target)
        if key not in visibility:
            annotation = annotations[frame][target]
            visibility[key] = any(
                bool(wrapper._make_loss_roi(sample_data, annotation).any())
                for sample_data in segment[frame]
                if MotionDataset.check_sensor(
                    wrapper.base_dataset.tables,
                    wrapper.base_dataset.indices,
                    sample_data,
                    modality="camera")
            )
        return visibility[key]

    output = []
    for duration in durations:
        seen_targets = set()
        last_start = frame_count - duration - wrapper.min_observed_after
        starts = list(range(wrapper.min_observed_before, last_start + 1))
        # Avoid systematically selecting the earliest legal interval for
        # shorter durations while retaining a deterministic manifest.
        offset = (base_index * 17 + duration * 13) % len(starts)
        starts = starts[offset:] + starts[:offset]
        for start in starts:
            required = range(
                start - wrapper.min_observed_before,
                start + duration + wrapper.min_observed_after)
            candidates = set(annotations[next(iter(required))])
            for frame in required:
                candidates.intersection_update(annotations[frame])
            for target in sorted(candidates):
                target_category = category(target)
                target_class_id = wrapper._class_id(target_category)
                if target in seen_targets or target_class_id <= 0:
                    continue
                if category_prefixes is not None and not any(
                        target_category.startswith(prefix)
                        for prefix in category_prefixes):
                    continue
                source_displacement_m = float(np.linalg.norm(
                    np.asarray(
                        annotations[start - 1][target]["translation"],
                        dtype=np.float64) -
                    np.asarray(
                        annotations[start - 2][target]["translation"],
                        dtype=np.float64)))
                if source_displacement_m < min_source_displacement_m:
                    continue
                if not all(visible(frame, target) for frame in range(
                        start, start + duration)):
                    continue
                output.append({
                    "base_index": int(base_index),
                    "target_instance_token": target,
                    "target_instance_hash": _stable_int64(target),
                    "target_category": target_category,
                    "target_class_id": target_class_id,
                    "missing_start": int(start),
                    "missing_duration": int(duration),
                    "missing_end_exclusive": int(start + duration),
                    "min_observed_before": wrapper.min_observed_before,
                    "min_observed_after": wrapper.min_observed_after,
                    "pair_seed": wrapper.seed + base_index * 1000003,
                    "source_displacement_m": source_displacement_m,
                })
                seen_targets.add(target)
    return output


def choose_balanced(
    candidates, scenes, durations, count, record_validator=None
):
    selected = []
    used_base_indices = set()
    used_targets = set()
    selected_by_scene = defaultdict(list)
    for round_index in range(count):
        for scene in scenes:
            scene_base_indices = sorted({
                record["base_index"]
                for duration in durations
                for record in candidates[(scene, duration)]
            })
            midpoint = (scene_base_indices[0] + scene_base_indices[-1]) / 2
            for duration in durations:
                pool = [
                    record for record in candidates[(scene, duration)]
                    if record["base_index"] not in used_base_indices
                ]
                if not pool:
                    raise RuntimeError(
                        f"Not enough unique windows for scene={scene}, "
                        f"duration={duration}")

                def score(record):
                    chosen_indices = selected_by_scene[scene]
                    if chosen_indices:
                        temporal_spread = min(
                            abs(record["base_index"] - base_index)
                            for base_index in chosen_indices)
                    else:
                        temporal_spread = -abs(
                            record["base_index"] - midpoint)
                    return (
                        record["target_instance_token"] not in used_targets,
                        temporal_spread,
                        -abs(record["base_index"] - midpoint),
                        -record["base_index"],
                    )

                ranked_pool = sorted(pool, key=score, reverse=True)
                record = next((
                    candidate for candidate in ranked_pool
                    if record_validator is None or
                    record_validator(candidate)
                ), None)
                if record is None:
                    raise RuntimeError(
                        f"Not enough non-vacuous windows for scene={scene}, "
                        f"duration={duration}")
                record = dict(record)
                record["scene_token"] = scene
                record["selection_round"] = round_index
                selected.append(record)
                used_base_indices.add(record["base_index"])
                used_targets.add(record["target_instance_token"])
                selected_by_scene[scene].append(record["base_index"])
    return selected


def main():
    args = parse_args()
    if args.records_per_duration_per_scene <= 0:
        raise ValueError("records-per-duration-per-scene must be positive")
    if args.min_source_displacement_m < 0:
        raise ValueError("min-source-displacement-m must be non-negative")
    if args.scene_partition_count <= 0:
        raise ValueError("scene-partition-count must be positive")
    if not 0 <= args.scene_partition_index < args.scene_partition_count:
        raise ValueError(
            "scene-partition-index must be in "
            "[0, scene-partition-count)")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in config.get("global_state", {}).items():
        dwm.common.global_state[key] = \
            dwm.common.create_instance_from_config(value)
    dataset = dwm.common.create_instance_from_config(config[args.dataset_key])
    if not isinstance(dataset, DatasetAdapter):
        raise TypeError(f"{args.dataset_key} must be a DatasetAdapter")
    wrapper = dataset.base_dataset
    base_item_count = len(wrapper.base_dataset)
    durations = sorted(wrapper.missing_durations)
    scenes_by_index = {
        base_index: scene_token(wrapper, base_index)
        for base_index in range(base_item_count)
    }
    all_scenes = sorted(set(scenes_by_index.values()))
    partition_scenes = [
        scene for scene_index, scene in enumerate(all_scenes)
        if scene_index % args.scene_partition_count ==
        args.scene_partition_index
    ]
    partition_scene_set = set(partition_scenes)

    candidates = defaultdict(list)
    for base_index in range(base_item_count):
        scene = scenes_by_index[base_index]
        if scene in partition_scene_set:
            for record in fast_candidates_for_base(
                    wrapper, base_index, durations, args.category_prefix,
                    args.min_source_displacement_m):
                candidates[(scene, record["missing_duration"])].append(
                    record)
        if (base_index + 1) % 100 == 0 or \
                base_index + 1 == base_item_count:
            print(json.dumps({
                "scanned": base_index + 1,
                "total": base_item_count,
            }), flush=True)

    required = args.records_per_duration_per_scene
    shortages = {
        f"{scene}:{duration}": len(candidates[(scene, duration)])
        for scene in partition_scenes for duration in durations
        if len(candidates[(scene, duration)]) < required
    }
    if shortages and not args.skip_insufficient_scenes:
        raise RuntimeError(f"Insufficient candidates: {shortages}")
    skipped_scene_set = {
        scene for scene in partition_scenes
        if any(
            len(candidates[(scene, duration)]) < required
            for duration in durations)
    }
    if args.skip_insufficient_scenes:
        for scene in partition_scenes:
            if scene in skipped_scene_set:
                continue
            try:
                choose_balanced(
                    candidates, [scene], durations, required)
            except RuntimeError:
                skipped_scene_set.add(scene)
    skipped_scenes = sorted(skipped_scene_set)
    scenes = [
        scene for scene in partition_scenes
        if scene not in skipped_scene_set
    ]
    if not scenes:
        raise RuntimeError(
            "No scenes have enough candidates for every duration")
    # Run the expensive raster-change test lazily while selecting. If a
    # high-ranked target is fully occluded in the box raster, selection falls
    # back to the next candidate instead of aborting the entire build.
    raster_change_cache = {}

    def is_non_vacuous(record):
        key = (
            record["base_index"], record["target_instance_token"],
            record["missing_start"], record["missing_duration"])
        if key in raster_change_cache:
            return raster_change_cache[key][0]
        segment = wrapper._segment(record["base_index"])
        changed_frames = [
            frame for frame in range(
                record["missing_start"], record["missing_end_exclusive"])
            if wrapper._changes_raster(
                segment[frame], record["target_instance_token"])
        ]
        expected_frames = list(range(
            record["missing_start"], record["missing_end_exclusive"]))
        is_valid = changed_frames == expected_frames
        raster_change_cache[key] = (is_valid, changed_frames)
        return is_valid

    selected_by_scene = {}
    raster_insufficient_scenes = []
    for scene_index, scene in enumerate(scenes):
        try:
            selected_by_scene[scene] = choose_balanced(
                candidates, [scene], durations,
                args.records_per_duration_per_scene,
                record_validator=is_non_vacuous)
        except RuntimeError:
            if not args.skip_insufficient_scenes:
                raise
            raster_insufficient_scenes.append(scene)
        if (scene_index + 1) % 10 == 0 or scene_index + 1 == len(scenes):
            print(json.dumps({
                "raster_validated_scenes": scene_index + 1,
                "raster_validation_total": len(scenes),
            }), flush=True)
    if raster_insufficient_scenes:
        skipped_scene_set.update(raster_insufficient_scenes)
        skipped_scenes = sorted(skipped_scene_set)
        scenes = [
            scene for scene in scenes
            if scene not in set(raster_insufficient_scenes)
        ]
    records = [
        record for scene in scenes for record in selected_by_scene[scene]
    ]
    if not records:
        raise RuntimeError("No non-vacuous records were selected")
    for record in records:
        key = (
            record["base_index"], record["target_instance_token"],
            record["missing_start"], record["missing_duration"])
        record["raster_change_verified_frames"] = \
            raster_change_cache[key][1]
    for manifest_index, record in enumerate(records):
        record["manifest_index"] = manifest_index

    scene_counts = Counter(record["scene_token"] for record in records)
    duration_counts = Counter(record["missing_duration"] for record in records)
    scene_duration_counts = Counter(
        (record["scene_token"], record["missing_duration"])
        for record in records)
    report = {
        "schema_version": 2,
        "records": records,
        "constraints": {
            "dataset_key": args.dataset_key,
            "base_item_count": base_item_count,
            "dataset_scene_count": len(all_scenes),
            "scene_partition_count": args.scene_partition_count,
            "scene_partition_index": args.scene_partition_index,
            "partition_scene_count": len(partition_scenes),
            "independent_scene_count": len(scenes),
            "scene_tokens": scenes,
            "skipped_insufficient_scene_count": len(skipped_scenes),
            "skipped_insufficient_scene_tokens": skipped_scenes,
            "records_per_duration_per_scene": required,
            "missing_durations": durations,
            "category_prefixes": args.category_prefix,
            "min_source_displacement_m": args.min_source_displacement_m,
            "unique_base_index": len({
                record["base_index"] for record in records}) == len(records),
            "unique_target_count": len({
                record["target_instance_token"] for record in records}),
            "scene_counts": dict(scene_counts),
            "duration_counts": {
                str(key): value for key, value in duration_counts.items()},
            "scene_duration_counts": {
                f"{scene}:{duration}": scene_duration_counts[(scene, duration)]
                for scene in scenes for duration in durations
            },
            "candidate_counts": {
                f"{scene}:{duration}": len(candidates[(scene, duration)])
                for scene in scenes for duration in durations
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary_constraints = {
        key: value for key, value in report["constraints"].items()
        if key not in {
            "scene_tokens", "scene_counts", "scene_duration_counts",
            "candidate_counts", "skipped_insufficient_scene_tokens",
        }
    }
    print(json.dumps({
        "status": "passed",
        "output": str(args.output),
        "record_count": len(records),
        "constraints": summary_constraints,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
