"""Aggregate the simplified cross-dataset DVS-Gesture replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def collect(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in root.glob(f"**/{filename}"):
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    stage = run_dir / "stageD_cross_dataset"
    diagnostic_root = stage / "diagnostics"
    geometry = collect(diagnostic_root, "geometry.csv")
    geometry.to_csv(stage / "geometry.csv", index=False)
    for filename in (
        "weight_updates.csv",
        "subspace_comparisons.csv",
        "parity.csv",
        "basis_generalization.csv",
    ):
        collect(diagnostic_root, filename).to_csv(stage / filename, index=False)

    checkpoint_root = run_dir / "checkpoints" / "stageD_cross_dataset"
    training_rows = []
    summaries = []
    for method in ("dvs_dfa", "dvs_bptt", "dvs_mean_gate"):
        for seed_dir in sorted((checkpoint_root / method).glob("seed_*")):
            metrics_path = seed_dir / "training_metrics.json"
            summary_path = seed_dir / "variant_summary.json"
            if metrics_path.is_file():
                training_rows.extend(json.loads(metrics_path.read_text()))
            if summary_path.is_file():
                summaries.append(json.loads(summary_path.read_text()))
    training = pd.DataFrame(training_rows)
    training.to_csv(stage / "training_metrics.csv", index=False)
    tests = []
    for summary in summaries:
        if summary["epochs"] < 100 and summary["method"] in {"dvs_dfa", "dvs_bptt"}:
            continue
        for checkpoint, key in (("best_validation", "best_validation_test"), ("final", "final_test")):
            tests.append(
                {
                    "method": summary["method"],
                    "seed": summary["seed"],
                    "checkpoint": checkpoint,
                    "test_accuracy": summary[key]["accuracy"],
                    "test_loss": summary[key]["loss"],
                    "best_validation_epoch": summary["best_validation_epoch"],
                }
            )
    tests = pd.DataFrame(tests)
    tests.to_csv(stage / "test_results.csv", index=False)

    primary = geometry[
        (geometry.epoch == 100)
        & (geometry.probe_split == "basis_eval")
        & (geometry.residualization == "raw")
        & (geometry.temporal_mode == "timestep")
    ]
    rows = []
    for seed in sorted(primary.seed.unique()):
        for layer in sorted(primary.layer.unique()):
            dfa = primary[
                (primary.method == "dvs_dfa")
                & (primary.seed == seed)
                & (primary.layer == layer)
                & (primary.signal_type == "dfa_actual")
            ]
            q = primary[
                (primary.method == "dvs_dfa")
                & (primary.seed == seed)
                & (primary.layer == layer)
                & (primary.signal_type == "dfa_q")
            ]
            bp = primary[
                (primary.method == "dvs_bptt")
                & (primary.seed == seed)
                & (primary.layer == layer)
                & (primary.signal_type == "bptt_actual")
            ]
            if dfa.empty or q.empty or bp.empty:
                continue
            rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "dfa_r95": int(dfa.iloc[0].r95),
                    "q_r95": int(q.iloc[0].r95),
                    "bptt_r95": int(bp.iloc[0].r95),
                    "GE": float(dfa.iloc[0].r95 / max(q.iloc[0].r95, 1)),
                    "dfa_temporal_coherence": float(dfa.iloc[0].temporal_coherence),
                    "bptt_temporal_coherence": float(bp.iloc[0].temporal_coherence),
                }
            )
    replication = pd.DataFrame(rows)
    replication.to_csv(stage / "dvs_replication.csv", index=False)
    best = tests[tests.checkpoint == "best_validation"]
    accuracy = best.groupby("method").test_accuracy.agg(["mean", "std"])
    reliable = bool(
        "dvs_dfa" in accuracy.index
        and "dvs_bptt" in accuracy.index
        and accuracy.loc["dvs_dfa", "mean"] >= .20
        and accuracy.loc["dvs_bptt", "mean"] >= .20
    )
    layer_support = []
    for _layer, group in replication.groupby("layer"):
        layer_support.append(
            bool((group.GE >= 1.5).all() and (group.dfa_r95 > group.bptt_r95).all())
        )
    replicated = bool(reliable and sum(layer_support) >= 2)
    summary = {
        "status": "complete",
        "dataset": "DVS-Gesture",
        "diagnostic_jobs": len(list(diagnostic_root.glob("**/complete.json"))),
        "best_validation_test_accuracy": accuracy.to_dict("index"),
        "baseline_reliable": reliable,
        "core_geometry_replicated": replicated,
        "replication_rule": "both baselines >=20% test accuracy and all-seed GE>=1.5 plus DFA r95>BPTT r95 in at least two layers",
    }
    (stage / "stageD_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
