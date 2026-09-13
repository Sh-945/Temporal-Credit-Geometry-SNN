"""Create and audit the immutable Paper 1 Experiment 03 run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paper1_geometry.data import build_paper1_split  # noqa: E402
from training.config import load_config  # noqa: E402
from training.data import build_datasets  # noqa: E402


SEEDS = (20260830, 20260831, 20260901)
EPOCHS = (0, 10, 25, 50, 75, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nmnist/paper1_experiment03.yaml")
    parser.add_argument("--legacy-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--full-data-checksum", action="store_true")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> list[dict[str, Any]]:
    roots = (
        ROOT / "analysis" / "paper1_geometry",
        ROOT / "methods" / "gate_intervention",
        ROOT / "models" / "temporal_ann_control",
        ROOT / "training",
        ROOT / "models" / "fc",
    )
    files = [
        path
        for base in roots
        for path in base.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    return [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)}
        for path in sorted(set(files))
    ]


def dataset_checksum(files: list[Path], *, full: bool) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        digest.update(path.as_posix().encode())
        digest.update(str(size).encode())
        if full:
            digest.update(bytes.fromhex(file_hash(path)))
    return {
        "algorithm": "sha256_path_size_and_content" if full else "sha256_path_and_size",
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
    }


def checkpoint_name(epoch: int) -> str:
    return "init.pt" if epoch == 0 else f"epoch_{epoch:03d}.pt"


def audit_checkpoints(legacy: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    overall = True
    for seed in SEEDS:
        initial_path = legacy / "initial_states" / f"seed_{seed}.pt"
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        initial_model_hash = state_hash(initial["model_state"])
        initial_feedback_hash = state_hash(initial["feedback_state"])
        for legacy_method, paper_method in (("full_dense", "dfa_trained"), ("full_bptt", "bptt_trained")):
            method_dir = legacy / "checkpoints" / legacy_method / f"seed_{seed}"
            init = torch.load(method_dir / "init.pt", map_location="cpu", weights_only=False)
            forward_match = state_hash(init["model_state"]) == initial_model_hash
            feedback_match = (
                True
                if legacy_method == "full_bptt"
                else state_hash(init["feedback_state"]) == initial_feedback_hash
            )
            recipe_match = all(
                init["config"][section] == config[section]
                for section in ("data", "model", "neuron")
            ) and all(
                init["config"]["training"].get(key) == config["training"].get(key)
                for key in (
                    "epochs",
                    "learning_rate",
                    "weight_decay",
                    "gradient_clip",
                    "lr_step",
                    "lr_gamma",
                )
            )
            checkpoint_presence = {
                str(epoch): (method_dir / checkpoint_name(epoch)).is_file()
                for epoch in EPOCHS
            }
            passed = forward_match and feedback_match and recipe_match and all(checkpoint_presence.values())
            overall = overall and passed
            rows.append(
                {
                    "seed": seed,
                    "method": paper_method,
                    "forward_initialization_match": forward_match,
                    "feedback_initialization_match_or_na": feedback_match,
                    "matched_recipe": recipe_match,
                    "checkpoint_presence": checkpoint_presence,
                    "passed": passed,
                    "initial_model_sha256": initial_model_hash,
                    "initial_feedback_sha256": initial_feedback_hash,
                }
            )
    return {"passed": overall, "rows": rows}


def environment_text() -> str:
    try:
        nvidia = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.STDOUT
        )
    except Exception as error:  # pragma: no cover - environment record only
        nvidia = f"nvidia-smi unavailable: {error}"
    return "\n".join(
        (
            f"utc_time: {datetime.now(timezone.utc).isoformat()}",
            f"os: {platform.platform()}",
            f"python: {sys.version}",
            f"torch: {torch.__version__}",
            f"torch_cuda: {torch.version.cuda}",
            f"cudnn: {torch.backends.cudnn.version()}",
            f"cuda_available: {torch.cuda.is_available()}",
            f"seeds: {list(SEEDS)}",
            "\n[nvidia-smi]\n" + nvidia,
        )
    )


def main() -> None:
    args = parse_args()
    config_path = absolute(args.config)
    legacy = absolute(args.legacy_run)
    run_dir = absolute(args.run_dir)
    for name in (
        "stageA_bptt",
        "stageB_ann",
        "stageC_gate",
        "stageD_cross_dataset",
        "paper_figures",
        "paper_tables",
        "supplementary",
        "configs",
        "checkpoints",
        "logs",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "configs" / config_path.name)
    config = load_config(config_path)
    train_dataset, test_dataset = build_datasets(config)
    split = build_paper1_split(len(train_dataset))
    pd.DataFrame(
        [
            *({"dataset_index": int(index), "split": "training"} for index in split.training_indices),
            *({"dataset_index": int(index), "split": "validation"} for index in split.validation_indices),
        ]
    ).to_csv(run_dir / "data_split.csv", index=False)
    probe_rows = []
    for position, index in enumerate(split.probe_indices):
        label = int(train_dataset.labels[int(index)])
        probe_rows.append(
            {
                "probe_position": position,
                "dataset_index": int(index),
                "probe_split": "basis_fit" if position < 512 else "basis_eval",
                "label": label,
            }
        )
    pd.DataFrame(probe_rows).to_csv(run_dir / "diagnostic_probe.csv", index=False)
    audit = audit_checkpoints(legacy, config)
    (run_dir / "stageA_bptt" / "checkpoint_reuse_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if not audit["passed"]:
        raise AssertionError("legacy Dense/BPTT checkpoint reuse audit failed")
    sources = source_hashes()
    pd.DataFrame(sources).to_csv(run_dir / "source_hashes.csv", index=False)
    data_files = sorted(set(train_dataset.files + test_dataset.files))
    data_hash = dataset_checksum(data_files, full=args.full_data_checksum)
    (run_dir / "dataset_checksums.json").write_text(
        json.dumps({"nmnist": data_hash}, indent=2), encoding="utf-8"
    )
    (run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    label_counts = {
        "basis_fit": Counter(row["label"] for row in probe_rows[:512]),
        "basis_eval": Counter(row["label"] for row in probe_rows[512:]),
    }
    manifest = {
        "experiment": "Paper 1 Experiment 03",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage_order": ["A", "B", "C", "D"],
        "current_stage": "A",
        "seeds": list(SEEDS),
        "diagnostic_epochs": list(EPOCHS),
        "probe": {
            "training_only": True,
            "size": 1024,
            "basis_fit": 512,
            "basis_eval": 512,
            "probe_seed": 20260831,
            "label_counts": {key: dict(value) for key, value in label_counts.items()},
        },
        "checkpoint_reuse": "Dense DFA and truly-trained matched BPTT only; geometry is recomputed",
        "forbidden_evidence": ["LoDFA", "low-rank B", "spectral filtering", "soft weighting"],
        "test_used_for_selection": False,
        "dataset_checksums": {"nmnist": data_hash},
        "source_hash_file": "source_hashes.csv",
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "prepared", "run_dir": str(run_dir), "audit": audit["passed"], "probe_label_counts": manifest["probe"]["label_counts"]}), flush=True)


if __name__ == "__main__":
    main()
