"""Extract and audit object-availability adapter checkpoint tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--optimizer", type=Path)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state_dict = payload.get("state_dict", payload)
    adapter = {
        key: value for key, value in state_dict.items()
        if "object_availability_adapter" in key
    }
    if not adapter:
        raise RuntimeError("No object-availability adapter tensors found")

    tensor_reports = {}
    for key, value in adapter.items():
        finite = bool(torch.isfinite(value).all())
        float_value = value.detach().float()
        tensor_reports[key] = {
            "shape": list(value.shape),
            "numel": value.numel(),
            "finite": finite,
            "nonzero": int(torch.count_nonzero(value)),
            "l2": float(torch.linalg.vector_norm(float_value)),
            "max_abs": float(float_value.abs().max()),
        }

    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter, args.adapter_output)
    report = {
        "status": "passed",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "adapter_output": str(args.adapter_output.resolve()),
        "adapter_output_size_bytes": args.adapter_output.stat().st_size,
        "adapter_output_sha256": sha256(args.adapter_output),
        "adapter_tensor_count": len(adapter),
        "adapter_parameter_count": sum(value.numel() for value in adapter.values()),
        "all_finite": all(item["finite"] for item in tensor_reports.values()),
        "all_nonzero": all(item["nonzero"] > 0 for item in tensor_reports.values()),
        "tensors": tensor_reports,
    }
    if args.optimizer is not None:
        optimizer = torch.load(
            args.optimizer, map_location="cpu", weights_only=True, mmap=True)
        steps = []
        for state in optimizer.get("state", {}).values():
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                step = step.item()
            if step is not None:
                steps.append(int(step))
        report["optimizer"] = {
            "path": str(args.optimizer.resolve()),
            "state_count": len(optimizer.get("state", {})),
            "steps": sorted(set(steps)),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "adapter_tensor_count": report["adapter_tensor_count"],
        "adapter_parameter_count": report["adapter_parameter_count"],
        "all_finite": report["all_finite"],
        "all_nonzero": report["all_nonzero"],
        "optimizer": report.get("optimizer"),
        "adapter_output_sha256": report["adapter_output_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
