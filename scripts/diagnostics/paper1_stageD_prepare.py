"""Audit DVS-Gesture and create paired initial states after stages A-C."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paper1_geometry.data import build_paper1_split
from analysis.soft_spectral.training import save_paired_initial_state
from scripts.diagnostics.paper1_prepare import dataset_checksum
from training.config import load_config
from training.data import build_datasets


SEEDS = (20260830, 20260831, 20260901)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dvs_gesture/paper1_experiment03.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--full-data-checksum", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    run_dir = Path(args.run_dir).resolve()
    stage = run_dir / "stageD_cross_dataset"
    stage.mkdir(parents=True, exist_ok=True)
    config_snapshot = run_dir / "configs" / f"dvs_gesture_{config_path.name}"
    config_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, config_snapshot)
    config = load_config(config_path)
    train, test = build_datasets(config)
    split = build_paper1_split(
        len(train),
        split_seed=int(config["experiment03"]["split_seed"]),
        probe_seed=int(config["experiment03"]["probe_seed"]),
        validation_fraction=float(config["experiment03"]["validation_fraction"]),
        probe_size=int(config["experiment03"]["probe_size"]),
    )
    pd.DataFrame(
        [
            *({"dataset_index": int(index), "split": "training"} for index in split.training_indices),
            *({"dataset_index": int(index), "split": "validation"} for index in split.validation_indices),
        ]
    ).to_csv(stage / "data_split.csv", index=False)
    probe_rows = []
    probe_half = len(split.probe_indices) // 2
    for position, index in enumerate(split.probe_indices):
        probe_rows.append(
            {
                "probe_position": position,
                "dataset_index": int(index),
                "probe_split": "basis_fit" if position < probe_half else "basis_eval",
                "label": int(train.labels[int(index)]),
            }
        )
    pd.DataFrame(probe_rows).to_csv(stage / "diagnostic_probe.csv", index=False)
    initial_root = run_dir / "checkpoints" / "stageD_cross_dataset" / "initial_states"
    audits = []
    for seed in SEEDS:
        seed_config = copy.deepcopy(config)
        seed_config["experiment"]["seed"] = seed
        audits.append(
            save_paired_initial_state(
                config=seed_config,
                seed=seed,
                path=initial_root / f"seed_{seed}.pt",
            )
        )
    files = sorted(set(train.files + test.files))
    checksum = dataset_checksum(files, full=args.full_data_checksum)
    sample, label = train[0]
    audit = {
        "status": "prepared",
        "dataset": "DVS-Gesture",
        "train_samples_official": len(train),
        "test_samples_official": len(test),
        "formal_training_samples": len(split.training_indices),
        "validation_samples": len(split.validation_indices),
        "probe_samples": len(split.probe_indices),
        "sample_shape": list(sample.shape),
        "first_label": int(label),
        "train_label_counts": dict(Counter(map(int, train.labels))),
        "test_label_counts": dict(Counter(map(int, test.labels))),
        "probe_label_counts": {
            "basis_fit": dict(Counter(row["label"] for row in probe_rows[:probe_half])),
            "basis_eval": dict(Counter(row["label"] for row in probe_rows[probe_half:])),
        },
        "dataset_checksum": checksum,
        "paired_initializations": audits,
    }
    (stage / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
