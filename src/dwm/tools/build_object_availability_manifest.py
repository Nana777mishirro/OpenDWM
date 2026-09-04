"""Build deterministic eligibility records for object availability training."""

import argparse
import json
from pathlib import Path

import dwm.common
from dwm.datasets.availability import ObjectAvailabilityDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a 3-before/2-after persistent-object manifest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument(
        "--dataset-key",
        choices=("training_dataset", "validation_dataset"),
        default="training_dataset")
    return parser.parse_args()


def find_wrapper_config(value):
    if isinstance(value, dict):
        if value.get("_class_name") == \
                "dwm.datasets.availability.ObjectAvailabilityDataset":
            return value
        for child in value.values():
            result = find_wrapper_config(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find_wrapper_config(child)
            if result is not None:
                return result
    return None


def referenced_global_state_keys(value):
    keys = set()
    if isinstance(value, dict):
        if value.get("_class_name") == "dwm.common.get_state":
            keys.add(value["key"])
        for child in value.values():
            keys.update(referenced_global_state_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(referenced_global_state_keys(child))
    return keys


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    wrapper_config = find_wrapper_config(config[args.dataset_key])
    if wrapper_config is None:
        raise RuntimeError(
            f"No ObjectAvailabilityDataset in {args.dataset_key}")
    # Dataset-only tooling must not initialize unrelated training state such as
    # a 16-GPU device mesh.
    required_states = referenced_global_state_keys(wrapper_config["base_dataset"])
    for key in required_states:
        value = config.get("global_state", {})[key]
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    base_dataset = dwm.common.create_instance_from_config(
        wrapper_config["base_dataset"])
    wrapper_args = {
        key: value for key, value in wrapper_config.items()
        if key not in ("_class_name", "base_dataset", "manifest_path")
    }
    wrapper = ObjectAvailabilityDataset(
        base_dataset=base_dataset, **wrapper_args)

    durations = wrapper.missing_durations
    records = []
    scan_limit = min(
        len(base_dataset), args.scan_limit or len(base_dataset))
    for base_index in range(scan_limit):
        duration = durations[len(records) % len(durations)]
        record = wrapper.find_record(base_index, duration=duration)
        if record is not None:
            records.append(record)
        if len(records) >= args.count:
            break
    if not records:
        raise RuntimeError("No eligible persistent objects were found")

    payload = {
        "schema_version": 1,
        "records": records,
        "constraints": {
            "single_selected_object": True,
            "missing_durations": list(durations),
            "min_observed_before": wrapper.min_observed_before,
            "min_observed_after": wrapper.min_observed_after,
            "target_visible_in_configured_views": True,
            "velocity_input": False,
            "future_state_as_model_input": False,
        },
        "scanned_item_count": base_index + 1,
        "eligible_record_count": len(records),
        "dataset_key": args.dataset_key,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "scanned_item_count": payload["scanned_item_count"],
        "eligible_record_count": len(records),
        "dataset_key": args.dataset_key,
        "durations": sorted({i["missing_duration"] for i in records}),
    }, indent=2))


if __name__ == "__main__":
    main()
