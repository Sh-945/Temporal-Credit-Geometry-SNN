"""Extract sanitized metadata from the private experiment archive.

This helper is used by maintainers when assembling a release.  It never
modifies the source archive or any canonical result file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260830, 20260831, 20260901, 20260908, 20260909)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(config))
    clean.get("data", {})["root"] = "DATA_ROOT"
    clean.get("training", {})["output_dir"] = "OUTPUT_DIR"
    clean.pop("_config_path", None)
    return clean


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feedback_manifest(archive: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        source = (
            archive
            / "results/paper_v9_signal_processing/logs"
            / f"dfa_trained_{seed}/complete.json"
        )
        record = json.loads(source.read_text(encoding="utf-8"))
        for item in record["feedback"]:
            rows.append(
                {
                    "dataset": "N-MNIST",
                    "method": record["method"],
                    "seed": record["seed"],
                    "layer": item["layer"],
                    "rows": item["shape"][0],
                    "columns": item["shape"][1],
                    "measured_rank": item["measured_rank"],
                    "sha256_tensor": item["sha256_tensor"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "source_record": f"dfa_trained_{seed}/complete.json",
                }
            )
    if len({row["sha256_tensor"] for row in rows}) != 15:
        raise RuntimeError("feedback matrices are not unique across all seed/layer pairs")
    write_csv(output / "feedback_matrix_manifest.csv", rows)


def checkpoint_metadata(checkpoint: Path, output: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    release_report_path = ROOT / "metadata/checkpoints/checkpoint_release_provenance.json"
    release_report = json.loads(release_report_path.read_text(encoding="utf-8"))
    optimizer = state.get("optimizer_state", {})
    groups = []
    for group in optimizer.get("param_groups", []):
        groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "params"
            }
            | {"parameter_tensor_count": len(group.get("params", []))}
        )
    feedback = []
    for name, tensor in sorted(state["feedback_state"].items()):
        feedback.append(
            {
                "state_key": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256_tensor": sha256_tensor(tensor),
            }
        )
    record = {
        "operation": "read-only release checkpoint inspection; no forward or optimizer step",
        "bundled_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "canonical_original_checkpoint_sha256": release_report["canonical_original_sha256"],
        "release_checkpoint_sha256": sha256_file(checkpoint),
        "release_sanitization_report": release_report_path.relative_to(ROOT).as_posix(),
        "stored_zero_based_epoch": int(state["epoch"]),
        "config": sanitize_config(state["config"]),
        "optimizer_parameter_groups": groups,
        "scheduler_state": state.get("scheduler_state"),
        "feedback_buffers": feedback,
        "checkpoint_keys": sorted(state),
    }
    (output / "canonical_checkpoint_config.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def public_code_manifest(archive: Path, output: Path) -> None:
    files = (
        "analysis/paper1_geometry/training.py",
        "analysis/paper1_geometry/diagnostics.py",
        "analysis/paper1_geometry/metrics.py",
        "models/fc/network.py",
        "models/neurons/lif.py",
        "methods/feedback_bank.py",
        "training/config.py",
        "training/data.py",
    )
    rows = []
    for relative in files:
        archive_file = archive / relative
        public_file = ROOT / "src" / relative
        archive_hash = sha256_file(archive_file)
        public_hash = sha256_file(public_file)
        rows.append(
            {
                "archive_relative_path": relative,
                "public_relative_path": public_file.relative_to(ROOT).as_posix(),
                "archive_sha256": archive_hash,
                "public_sha256": public_hash,
                "byte_identical": archive_hash == public_hash,
            }
        )
    write_csv(output / "source_code_provenance.csv", rows)


def split_manifest(output: Path) -> None:
    rows = []
    for relative in (
        "metadata/splits/nmnist_train_val_split.csv",
        "metadata/splits/nmnist_diagnostic_probe.csv",
        "metadata/splits/dvs_train_val_split.csv",
        "metadata/splits/dvs_diagnostic_probe.csv",
    ):
        path = ROOT / relative
        with path.open(newline="", encoding="utf-8-sig") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        rows.append(
            {
                "relative_path": relative,
                "rows": row_count,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(output / "split_probe_manifest.csv", rows)


def all_checkpoint_metadata(archive: Path, output: Path) -> None:
    source = (
        archive
        / "results/paper_v9_signal_processing/07_manuscript_final_audit/checkpoint_metadata.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for item in raw["checkpoints"]:
        optimizer_groups = []
        for group in item.get("optimizer_groups", []):
            optimizer_groups.append(
                {key: value for key, value in group.items() if key != "params"}
                | {"parameter_tensor_count": len(group.get("params", []))}
            )
        records.append(
            {
                "method": item["method"],
                "seed": item["seed"],
                "stored_zero_based_epoch": item["epoch"],
                "checkpoint_sha256": item["sha256"],
                "config": sanitize_config(item["config"]),
                "optimizer_parameter_groups": optimizer_groups,
                "scheduler_state": item.get("scheduler_state"),
                "checkpoint_keys": item.get("keys", []),
            }
        )
    published = {
        "operation": "sanitized read-only extraction from archived checkpoint metadata",
        "record_count": len(records),
        "records": records,
    }
    (output / "all_checkpoint_metadata_sanitized.json").write_text(
        json.dumps(published, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "metadata")
    args = parser.parse_args()
    archive = args.archive_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feedback_manifest(archive, output)
    checkpoint_metadata(
        ROOT / "metadata/checkpoints/nmnist_snn_dfa_seed_20260830_epoch_100_release.pt",
        output,
    )
    public_code_manifest(archive, output)
    split_manifest(output)
    all_checkpoint_metadata(archive, output)
    print(f"sanitized provenance written to {output}")


if __name__ == "__main__":
    main()
