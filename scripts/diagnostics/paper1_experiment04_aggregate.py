"""Aggregate Experiment 04 offline and H3-targeted DVS evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (20260830, 20260831, 20260901)
LAYERS = ("hidden_1", "hidden_2", "hidden_3")
METHODS = (
    ("Actual", "dvs_actual", "dfa_actual"),
    ("H3-MeanGate", "dvs_h3_mean_gate", "dfa_temporal_mean_normmatched"),
    ("H3-ShuffledGate", "dvs_h3_shuffled_gate", "dfa_timestep_shuffled"),
)
COLORS = {"Actual": "#4c78a8", "H3-MeanGate": "#f2cf5b", "H3-ShuffledGate": "#b279a2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    return parser.parse_args()


def _metric_row(frame: pd.DataFrame, signal: str, layer: str, temporal_mode: str) -> pd.Series:
    selected = frame[
        frame.signal_type.eq(signal)
        & frame.layer.eq(layer)
        & frame.probe_split.eq("basis_eval")
        & frame.temporal_mode.eq(temporal_mode)
        & frame.residualization.eq("raw")
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one {signal}/{layer}/{temporal_mode} row, got {len(selected)}")
    return selected.iloc[0]


def _diagnostic_dir(run_dir: Path, method: str, seed: int) -> Path:
    return run_dir / "stageB_h3" / "diagnostics" / method / f"seed_{seed}" / "epoch_100"


def build_counterfactual(run_dir: Path, source_run: Path) -> pd.DataFrame:
    source_geometry = pd.read_csv(source_run / "stageD_cross_dataset" / "geometry.csv")
    source_geometry = source_geometry[
        source_geometry.method.eq("dvs_dfa") & source_geometry.epoch.eq(100)
    ]
    rows: list[dict] = []
    variants = (
        ("Actual", None, "dfa_actual", "dfa_q", "gate_actual", "current", "current"),
        ("MeanGate", "mean_gate", "dfa_temporal_mean_unmatched", "dfa_q", "gate_applied", "current", "temporal_mean_unmatched"),
        ("FixedReferenceQ", "fixed_reference_q", "dfa_fixed_reference_q_current_gate", "dfa_reference_q", "gate_actual", "initial_checkpoint", "current"),
    )
    for seed in SEEDS:
        for label, folder, delta_signal, q_signal, gate_signal, q_source, gate_source in variants:
            if folder is None:
                geometry = source_geometry[source_geometry.seed.eq(seed)]
            else:
                geometry = pd.read_csv(
                    run_dir / "stageA_counterfactual" / folder / f"seed_{seed}" / "geometry.csv"
                )
            for layer in LAYERS:
                q = _metric_row(geometry, q_signal, layer, "timestep")
                delta_t = _metric_row(geometry, delta_signal, layer, "timestep")
                delta_a = _metric_row(geometry, delta_signal, layer, "aggregated")
                gate = _metric_row(geometry, gate_signal, layer, "timestep")
                rows.append(
                    {
                        "variant": label, "seed": seed, "layer": layer,
                        "q_source": q_source, "gate_source": gate_source,
                        "q_r95": int(q.r95),
                        "delta_timestep_r95": int(delta_t.r95),
                        "delta_aggregated_r95": int(delta_a.r95),
                        "ge": float(delta_t.r95 / max(float(q.r95), 1.0)),
                        "delta_timestep_entropy_rank": float(delta_t.entropy_rank),
                        "delta_aggregated_entropy_rank": float(delta_a.entropy_rank),
                        "credit_temporal_coherence": float(delta_t.temporal_coherence),
                        "gate_temporal_cosine": float(gate.mean_temporal_cosine),
                        "gate_entropy_rank": float(gate.entropy_rank),
                        "probe_split": "basis_eval", "residualization": "raw",
                    }
                )
    return pd.DataFrame(rows)


def _training_summary(run_dir: Path, source_run: Path, method: str, seed: int) -> tuple[dict, Path]:
    if method == "dvs_actual":
        root = source_run / "checkpoints" / "stageD_cross_dataset" / "dvs_dfa" / f"seed_{seed}"
    else:
        root = run_dir / "checkpoints" / "stageB_h3" / method / f"seed_{seed}"
    return json.loads((root / "variant_summary.json").read_text(encoding="utf-8")), root


def build_training_trajectory(run_dir: Path, source_run: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for label, method, _signal in METHODS:
        for seed in SEEDS:
            _summary, root = _training_summary(run_dir, source_run, method, seed)
            for row in json.loads((root / "training_metrics.json").read_text(encoding="utf-8")):
                rows.append({"Method": label, "Seed": seed, **row})
    return pd.DataFrame(rows)


def build_causal_results(run_dir: Path, source_run: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for label, method, h3_signal in METHODS:
        for seed in SEEDS:
            summary, _root = _training_summary(run_dir, source_run, method, seed)
            diagnostic = _diagnostic_dir(run_dir, method, seed)
            geometry = pd.read_csv(diagnostic / "geometry.csv")
            updates = pd.read_csv(diagnostic / "weight_updates.csv")
            parity = pd.read_csv(diagnostic / "parity.csv")
            row = {
                "Method": label,
                "Seed": seed,
                "Test accuracy": float(summary["final_test"]["accuracy"]),
                "Best-val test accuracy": float(summary["best_validation_test"]["accuracy"]),
                "Best validation epoch": int(summary["best_validation_epoch"]),
                "Convergence epoch": int(summary["convergence_epoch_99pct_own_best"]),
            }
            for layer_index, layer in enumerate(LAYERS, start=1):
                signal = h3_signal if layer == "hidden_3" else "dfa_actual"
                q = _metric_row(geometry, "dfa_q", layer, "timestep")
                delta_t = _metric_row(geometry, signal, layer, "timestep")
                delta_a = _metric_row(geometry, signal, layer, "aggregated")
                gate_signal = "gate_applied" if layer == "hidden_3" else "gate_actual"
                gate = _metric_row(geometry, gate_signal, layer, "timestep")
                prefix = f"H{layer_index}"
                row.update(
                    {
                        f"{prefix} q r95": int(q.r95),
                        f"{prefix} delta timestep r95": int(delta_t.r95),
                        f"{prefix} delta aggregated r95": int(delta_a.r95),
                        f"{prefix} GE": float(delta_t.r95 / max(float(q.r95), 1.0)),
                        f"{prefix} entropy rank": float(delta_t.entropy_rank),
                        f"{prefix} gate temporal cosine": float(gate.mean_temporal_cosine),
                        f"{prefix} credit temporal coherence": float(delta_t.temporal_coherence),
                    }
                )
            h3_update = updates[updates.layer.eq("hidden_3")]
            if len(h3_update) != 1:
                raise ValueError(f"expected one H3 update row for {method}/{seed}")
            h3_parity = parity[parity.layer.eq("hidden_3")]
            row.update(
                {
                    "H3 gradient norm": float(h3_update.iloc[0].method_gradient_norm),
                    "H3 weight-update cosine to BP": float(h3_update.iloc[0].gradient_cosine_to_bp),
                    "H3 actual-vs-intervened delta cosine": float(h3_parity.actual_vs_intervened_delta_cosine.mean()),
                    "max production delta relative error": float(parity.production_delta_relative_error.max()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _points_with_summary(ax, frame: pd.DataFrame, column: str, ylabel: str) -> None:
    labels = [item[0] for item in METHODS]
    for x, label in enumerate(labels):
        values = frame[frame.Method.eq(label)][column].to_numpy(float)
        offsets = np.linspace(-.08, .08, len(values))
        ax.scatter(np.full(len(values), x) + offsets, values, s=38, color=COLORS[label], alpha=.75, zorder=3)
        ax.errorbar(
            x, values.mean(), yerr=values.std(ddof=1), fmt="o", color="#222222",
            markerfacecolor=COLORS[label], markersize=8, capsize=4, linewidth=1.5, zorder=4,
        )
    ax.set_xticks(range(len(labels)), labels, rotation=16, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.2)


def causal_figure(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    _points_with_summary(axes[0, 0], frame, "Best-val test accuracy", "Test accuracy")
    axes[0, 0].set_title("A. Learning outcome")

    ax = axes[0, 1]
    for x, (label, _method, _signal) in enumerate(METHODS):
        subset = frame[frame.Method.eq(label)]
        for offset, (_index, row) in zip(np.linspace(-.08, .08, len(subset)), subset.iterrows()):
            values = [row["H3 q r95"], row["H3 delta timestep r95"]]
            ax.plot([2*x + offset, 2*x + 1 + offset], values, color=COLORS[label], alpha=.45)
            ax.scatter([2*x + offset, 2*x + 1 + offset], values, color=COLORS[label], s=28)
    for x, (label, _method, _signal) in enumerate(METHODS):
        subset = frame[frame.Method.eq(label)]
        ax.plot(
            [2*x, 2*x + 1],
            [subset["H3 q r95"].mean(), subset["H3 delta timestep r95"].mean()],
            color="#222222", marker="o", linewidth=2.2,
        )
    ax.set_xticks(
        [value for x in range(3) for value in (2*x, 2*x+1)],
        [name for label, *_ in METHODS for name in (f"{label}\nq", f"{label}\ndelta")],
        rotation=18, ha="right",
    )
    ax.set_ylabel("H3 r95")
    ax.set_title("B. H3 q → post-gate credit")
    ax.grid(axis="y", alpha=.2)

    _points_with_summary(axes[1, 0], frame, "H3 gate temporal cosine", "Gate temporal cosine")
    axes[1, 0].set_title("C. H3 gate diversity")
    _points_with_summary(axes[1, 1], frame, "H3 GE", "GE")
    axes[1, 1].axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axes[1, 1].set_title("D. H3 geometry expansion")
    fig.suptitle("DVS-Gesture H3-targeted causal validation")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def verdict(frame: pd.DataFrame) -> dict:
    means = frame.groupby("Method").mean(numeric_only=True)
    actual_ge = float(means.loc["Actual", "H3 GE"])
    mean_ge = float(means.loc["H3-MeanGate", "H3 GE"])
    actual_acc = float(means.loc["Actual", "Best-val test accuracy"])
    mean_acc = float(means.loc["H3-MeanGate", "Best-val test accuracy"])
    shuffle_acc = float(means.loc["H3-ShuffledGate", "Best-val test accuracy"])
    indexed = frame.set_index(["Method", "Seed"])
    actual_seed = indexed.loc["Actual", "Best-val test accuracy"]
    mean_seed = indexed.loc["H3-MeanGate", "Best-val test accuracy"]
    shuffle_seed = indexed.loc["H3-ShuffledGate", "Best-val test accuracy"]
    mean_effect = actual_seed - mean_seed
    shuffle_effect = actual_seed - shuffle_seed
    strong_lifting = actual_ge >= 5.0
    mean_compression = mean_ge <= 0.8 * actual_ge
    learning_effect = max(abs(actual_acc - mean_acc), abs(actual_acc - shuffle_acc)) >= .002
    label = (
        "CONDITIONAL CROSS-DATASET SUPPORT"
        if strong_lifting and mean_compression and learning_effect
        else "DESCRIPTIVE ONLY"
        if strong_lifting
        else "NOT REPLICATED"
    )
    return {
        "verdict": label,
        "full_cross_dataset_replication_failed": True,
        "actual_h3_ge": actual_ge,
        "mean_gate_h3_ge": mean_ge,
        "actual_best_val_test_accuracy": actual_acc,
        "mean_gate_best_val_test_accuracy": mean_acc,
        "shuffled_gate_best_val_test_accuracy": shuffle_acc,
        "mean_gate_accuracy_effect_actual_minus_intervention_pp": float(mean_effect.mean() * 100),
        "mean_gate_degradation_seed_count": int((mean_effect > 0).sum()),
        "shuffled_gate_accuracy_effect_actual_minus_intervention_pp": float(shuffle_effect.mean() * 100),
        "shuffled_gate_degradation_seed_count": int((shuffle_effect > 0).sum()),
        "alignment_accuracy_replication_stable": bool(
            (shuffle_effect > 0).all() and shuffle_effect.mean() >= .002
        ),
        "criteria": {"actual_strong_lifting": strong_lifting, "mean_gate_compression": mean_compression, "learning_effect_at_least_0.2pp": learning_effect},
    }


def main() -> None:
    args = parse_args()
    run_dir, source_run = args.run_dir.resolve(), args.source_run.resolve()
    counterfactual = build_counterfactual(run_dir, source_run)
    counterfactual.to_csv(run_dir / "dvs_counterfactual_gate.csv", index=False)
    causal = build_causal_results(run_dir, source_run)
    causal.to_csv(run_dir / "dvs_h3_causal_results.csv", index=False)
    trajectory = build_training_trajectory(run_dir, source_run)
    trajectory.to_csv(run_dir / "dvs_h3_training_trajectory.csv", index=False)
    causal_figure(causal, run_dir / "dvs_h3_causal_validation.png")
    result = verdict(causal)
    (run_dir / "stageB_mechanism_verdict.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **result}))


if __name__ == "__main__":
    main()
