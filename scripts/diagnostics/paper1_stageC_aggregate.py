"""Aggregate actual, temporal-mean, and time-shuffled causal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_SIGNALS = {
    "actual_gate": "dfa_actual",
    "mean_gate": "dfa_temporal_mean_normmatched",
    "shuffled_gate": "dfa_timestep_shuffled",
}


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


def paired_effect(difference: np.ndarray) -> dict[str, float | int]:
    mean = float(np.mean(difference))
    standard = float(np.std(difference, ddof=1)) if len(difference) > 1 else float("nan")
    return {
        "seed_count": len(difference),
        "paired_difference_mean": mean,
        "paired_difference_std": standard,
        "paired_effect_dz": mean / standard if standard > 0 else float("nan"),
        "seed_consistency": float(np.mean(np.sign(difference) == np.sign(mean))),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    stage = run_dir / "stageC_gate"
    diagnostic_root = stage / "diagnostics"
    intervention_geometry = collect(diagnostic_root, "geometry.csv")
    stage_a_geometry = pd.read_csv(run_dir / "stageA_bptt" / "geometry.csv")
    actual_geometry = stage_a_geometry[stage_a_geometry.method == "dfa_trained"].copy()
    actual_geometry["method"] = "actual_gate"
    geometry = pd.concat([actual_geometry, intervention_geometry], ignore_index=True)
    geometry.to_csv(stage / "geometry.csv", index=False)

    stage_a_updates = pd.read_csv(run_dir / "stageA_bptt" / "weight_updates.csv")
    stage_a_updates = stage_a_updates[stage_a_updates.method == "dfa_trained"].copy()
    stage_a_updates["method"] = "actual_gate"
    new_updates = collect(diagnostic_root, "weight_updates.csv")
    updates = pd.concat([stage_a_updates, new_updates], ignore_index=True)
    updates.to_csv(stage / "weight_updates.csv", index=False)
    for filename in ("parity.csv", "basis_generalization.csv", "subspace_comparisons.csv"):
        collect(diagnostic_root, filename).to_csv(stage / filename, index=False)

    actual_training = pd.read_csv(run_dir / "stageA_bptt" / "training_metrics.csv")
    actual_training = actual_training[actual_training.method == "dfa_trained"].copy()
    actual_training["method"] = "actual_gate"
    checkpoint_root = run_dir / "checkpoints" / "stageC_gate"
    new_training_rows = []
    summary_rows = []
    for method in ("mean_gate", "shuffled_gate"):
        for seed_dir in sorted((checkpoint_root / method).glob("seed_*")):
            new_training_rows.extend(json.loads((seed_dir / "training_metrics.json").read_text()))
            summary_rows.append(json.loads((seed_dir / "variant_summary.json").read_text()))
    training = pd.concat([actual_training, pd.DataFrame(new_training_rows)], ignore_index=True)
    training.to_csv(stage / "training_metrics.csv", index=False)

    actual_tests = pd.read_csv(run_dir / "stageA_bptt" / "test_results.csv")
    actual_tests = actual_tests[actual_tests.method == "dfa_trained"].copy()
    actual_tests["method"] = "actual_gate"
    new_tests = []
    for summary in summary_rows:
        for checkpoint, key in (("best_validation", "best_validation_test"), ("final", "final_test")):
            new_tests.append(
                {
                    "method": summary["method"],
                    "seed": summary["seed"],
                    "checkpoint": checkpoint,
                    "test_accuracy": summary[key]["accuracy"],
                    "test_loss": summary[key]["loss"],
                    "best_validation_epoch": summary["best_validation_epoch"],
                    "convergence_epoch_99pct_own_best": summary["convergence_epoch_99pct_own_best"],
                }
            )
    tests = pd.concat([actual_tests, pd.DataFrame(new_tests)], ignore_index=True)
    tests.to_csv(stage / "test_results.csv", index=False)

    primary_rows = []
    for method, signal in PRIMARY_SIGNALS.items():
        selected = geometry[
            (geometry.method == method)
            & (geometry.signal_type == signal)
            & (geometry.epoch == 100)
            & (geometry.probe_split == "basis_eval")
            & (geometry.residualization == "raw")
        ]
        primary_rows.append(selected)
    primary = pd.concat(primary_rows, ignore_index=True)
    primary.to_csv(stage / "final_primary_geometry.csv", index=False)

    effects = []
    for comparison in ("mean_gate", "shuffled_gate"):
        for layer in ("hidden_1", "hidden_2", "hidden_3"):
            for mode in ("timestep", "aggregated"):
                selected = primary[(primary.layer == layer) & (primary.temporal_mode == mode)]
                for metric in ("r95", "entropy_rank", "temporal_coherence", "mean_temporal_cosine"):
                    pivot = selected.pivot(index="seed", columns="method", values=metric)
                    effects.append(
                        {
                            "comparison": f"actual_gate_minus_{comparison}",
                            "layer": layer,
                            "temporal_mode": mode,
                            "metric": metric,
                            **paired_effect(
                                (pivot["actual_gate"] - pivot[comparison]).to_numpy(float)
                            ),
                        }
                    )
        best = tests[tests.checkpoint == "best_validation"].pivot(
            index="seed", columns="method", values="test_accuracy"
        )
        effects.append(
            {
                "comparison": f"actual_gate_minus_{comparison}",
                "layer": "all",
                "temporal_mode": "na",
                "metric": "best_validation_test_accuracy",
                **paired_effect((best["actual_gate"] - best[comparison]).to_numpy(float)),
            }
        )
    effects_frame = pd.DataFrame(effects)
    effects_frame.to_csv(stage / "paired_effects.csv", index=False)

    accuracy = tests[tests.checkpoint == "best_validation"].groupby("method").test_accuracy.agg(["mean", "std"])
    timestep_effects = effects_frame[
        (effects_frame.temporal_mode == "timestep") & (effects_frame.metric == "r95")
    ]
    mean_geometry_changed = bool(
        (timestep_effects[timestep_effects.comparison == "actual_gate_minus_mean_gate"].paired_difference_mean.abs() >= 5).any()
    )
    shuffle_geometry_changed = bool(
        (timestep_effects[timestep_effects.comparison == "actual_gate_minus_shuffled_gate"].paired_difference_mean.abs() >= 5).any()
    )
    mean_accuracy_effect_pp = float(
        (accuracy.loc["actual_gate", "mean"] - accuracy.loc["mean_gate", "mean"]) * 100
    )
    shuffle_accuracy_effect_pp = float(
        (accuracy.loc["actual_gate", "mean"] - accuracy.loc["shuffled_gate", "mean"]) * 100
    )
    learning_effect = abs(mean_accuracy_effect_pp) >= .20 or abs(shuffle_accuracy_effect_pp) >= .20
    if mean_geometry_changed and shuffle_geometry_changed:
        driver = "gate diversity and gate-time alignment jointly matter"
    elif mean_geometry_changed:
        driver = "gate temporal diversity is the clearer driver"
    elif shuffle_geometry_changed:
        driver = "gate-time alignment is the clearer driver"
    else:
        driver = "neither intervention produced a pre-registered material rank change"
    if mean_geometry_changed or shuffle_geometry_changed:
        causal = (
            "gate causally shapes geometry and learning behavior"
            if learning_effect
            else "gate causally shapes geometry; performance necessity is not supported"
        )
    else:
        causal = "gate causal mechanism not supported"
    summary = {
        "status": "complete",
        "diagnostic_jobs": len(list(diagnostic_root.glob("**/complete.json"))),
        "best_validation_test_accuracy": accuracy.to_dict("index"),
        "mean_gate_accuracy_effect_actual_minus_intervention_pp": mean_accuracy_effect_pp,
        "shuffled_gate_accuracy_effect_actual_minus_intervention_pp": shuffle_accuracy_effect_pp,
        "mean_gate_geometry_changed": mean_geometry_changed,
        "shuffled_gate_geometry_changed": shuffle_geometry_changed,
        "driver": driver,
        "causal_conclusion": causal,
        "materiality_rule": "absolute paired mean timestep-r95 difference >=5 in any layer; learning effect >=0.20 percentage point",
    }
    (stage / "stageC_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
