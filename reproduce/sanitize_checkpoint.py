"""Create a path-sanitized release copy of a canonical PyTorch checkpoint.

Only non-scientific configuration paths are changed. The script verifies the
canonical input hash against the frozen result table, proves that every object
outside ``config`` is unchanged, reloads the release file, and emits a public
provenance report without recording the removed path values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = ROOT / "results/frozen_sources/nmnist/spectral_metrics_per_seed.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "metadata/checkpoints/nmnist_snn_dfa_seed_20260830_epoch_100_release.pt"
)
DEFAULT_REPORT = ROOT / "metadata/checkpoints/checkpoint_release_provenance.json"
DEFAULT_CONFIG_REPORT = ROOT / "metadata/canonical_checkpoint_config.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def update_digest(digest: Any, value: Any) -> None:
    """Stable recursive digest for checkpoint state, independent of torch.save."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            update_digest(digest, key)
            update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            update_digest(digest, item)
    elif isinstance(value, bytes):
        digest.update(b"bytes\0" + value)
    elif value is None:
        digest.update(b"none\0")
    elif isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif isinstance(value, int):
        digest.update(b"int\0" + str(value).encode())
    elif isinstance(value, float):
        digest.update(b"float\0" + struct.pack(">d", value))
    elif isinstance(value, str):
        digest.update(b"str\0" + value.encode("utf-8"))
    else:
        raise TypeError(f"Unsupported checkpoint value type: {type(value)!r}")


def scientific_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(k for k in state if k != "config"):
        update_digest(digest, key)
        update_digest(digest, state[key])
    return digest.hexdigest()


def tensor_inventory(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(value, torch.Tensor):
        rows[prefix] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256_tensor": tensor_sha256(value),
        }
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.update(tensor_inventory(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            rows.update(tensor_inventory(child, f"{prefix}[{index}]"))
    return rows


def config_differences(before: Any, after: Any, prefix: str = "config") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changed: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}"
            if key not in before or key not in after:
                changed.append(path)
            else:
                changed.extend(config_differences(before[key], after[key], path))
        return changed
    return [] if before == after else [prefix]


def iter_strings(value: Any, prefix: str = ""):
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_strings(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_strings(child, f"{prefix}[{index}]")


def is_private_path(value: str) -> bool:
    lowered = value.lower()
    return bool(
        re.match(r"^[a-z]:[\\/]", value, flags=re.IGNORECASE)
        or bool(re.match(r"^/(?:root|home|users)/", lowered))
        or lowered.startswith("ssh://")
        or bool(re.match(r"^[^@\s]+@[^\s:]+(?::\d+)?$", value))
    )


def canonical_sha256() -> str:
    with FROZEN_SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        matches = {
            row["checkpoint_sha256"]
            for row in csv.DictReader(handle)
            if row["dataset"] == "N-MNIST"
            and row["method"] == "SNN-DFA"
            and row["seed"] == "20260830"
            and row["checkpoint_rule"] == "completed_epoch_100"
        }
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen canonical hash, found {len(matches)}")
    return matches.pop()


def sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    # JSON round-trip is intentional: canonical configs contain only JSON types.
    clean = json.loads(json.dumps(config))
    clean.pop("_config_path", None)
    clean.setdefault("data", {})["root"] = "DATA_ROOT"
    clean.setdefault("training", {})["output_dir"] = "OUTPUT_DIR"
    return clean


def optimizer_groups(state: dict[str, Any]) -> list[dict[str, Any]]:
    groups = []
    for group in state.get("optimizer_state", {}).get("param_groups", []):
        groups.append(
            {key: value for key, value in group.items() if key != "params"}
            | {"parameter_tensor_count": len(group.get("params", []))}
        )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config-report", type=Path, default=DEFAULT_CONFIG_REPORT)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    config_report_path = args.config_report.resolve()
    if source == output:
        raise ValueError("Input and output must be different files")
    if output.exists():
        raise FileExistsError(output)

    frozen_hash = canonical_sha256()
    actual_input_hash = sha256_file(source)
    if actual_input_hash != frozen_hash:
        raise RuntimeError("Input checkpoint does not match the frozen canonical SHA-256")

    state = torch.load(source, map_location="cpu", weights_only=False)
    before_config = json.loads(json.dumps(state["config"]))
    before_scientific_hash = scientific_state_sha256(state)
    before_tensors = tensor_inventory({k: v for k, v in state.items() if k != "config"})
    state["config"] = sanitized_config(state["config"])
    changed_keys = config_differences(before_config, state["config"])
    allowed = {"config._config_path", "config.data.root", "config.training.output_dir"}
    if not set(changed_keys).issubset(allowed):
        raise RuntimeError(f"Unexpected config changes: {changed_keys}")
    remaining_private = [path for path, value in iter_strings(state) if is_private_path(value)]
    if remaining_private:
        raise RuntimeError(f"Private path-like values remain at keys: {remaining_private}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.stem + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        released = torch.load(temporary, map_location="cpu", weights_only=False)
        after_scientific_hash = scientific_state_sha256(released)
        after_tensors = tensor_inventory({k: v for k, v in released.items() if k != "config"})
        if after_scientific_hash != before_scientific_hash or after_tensors != before_tensors:
            raise RuntimeError("Scientific checkpoint state changed during sanitization")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    release_hash = sha256_file(output)
    report = {
        "operation": "path-only config sanitization; no training or optimizer step",
        "canonical_original_sha256": frozen_hash,
        "release_relative_path": output.relative_to(ROOT).as_posix(),
        "release_sha256": release_hash,
        "release_size_bytes": output.stat().st_size,
        "changed_config_keys": changed_keys,
        "removed_values_recorded": False,
        "scientific_state_sha256": before_scientific_hash,
        "scientific_state_equal_after_reload": True,
        "tensor_count": len(before_tensors),
        "all_non_config_tensors_equal_after_reload": True,
        "private_path_scan_after_reload": "passed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    feedback = []
    for name, tensor in sorted(released["feedback_state"].items()):
        feedback.append(
            {
                "state_key": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256_tensor": tensor_sha256(tensor),
            }
        )
    config_report = {
        "operation": "read-only release checkpoint inspection; no forward or optimizer step",
        "bundled_checkpoint": output.relative_to(ROOT).as_posix(),
        "canonical_original_checkpoint_sha256": frozen_hash,
        "release_checkpoint_sha256": release_hash,
        "release_sanitization_report": report_path.relative_to(ROOT).as_posix(),
        "stored_zero_based_epoch": int(released["epoch"]),
        "config": released["config"],
        "optimizer_parameter_groups": optimizer_groups(released),
        "scheduler_state": released.get("scheduler_state"),
        "feedback_buffers": feedback,
        "checkpoint_keys": sorted(released),
    }
    config_report_path.write_text(
        json.dumps(config_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
