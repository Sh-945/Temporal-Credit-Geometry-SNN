"""Build Experiment 04 DVS temporal-lifting trajectories from Exp03 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAYERS = ["hidden_1", "hidden_2", "hidden_3"]
COLORS = {"hidden_1": "#4c78a8", "hidden_2": "#f58518", "hidden_3": "#e45756"}
MARKERS = {"hidden_1": "o", "hidden_2": "s", "hidden_3": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-geometry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _one(frame: pd.DataFrame, signal: str, temporal_mode: str) -> pd.Series:
    selected = frame[
        frame.signal_type.eq(signal)
        & frame.temporal_mode.eq(temporal_mode)
        & frame.residualization.eq("raw")
        & frame.probe_split.eq("basis_eval")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one {signal}/{temporal_mode}/raw/basis_eval row, got {len(selected)}"
        )
    return selected.iloc[0]


def trajectory(geometry: pd.DataFrame) -> pd.DataFrame:
    geometry = geometry[
        geometry.dataset.eq("dvs_gesture") & geometry.method.eq("dvs_dfa")
    ].copy()
    rows: list[dict] = []
    for (seed, epoch, layer), group in geometry.groupby(["seed", "epoch", "layer"]):
        q = _one(group, "dfa_q", "timestep")
        delta_t = _one(group, "dfa_actual", "timestep")
        delta_a = _one(group, "dfa_actual", "aggregated")
        gate = _one(group, "gate_actual", "timestep")
        rows.append(
            {
                "dataset": "dvs_gesture",
                "method": "dvs_dfa",
                "seed": int(seed),
                "epoch": int(epoch),
                "layer": layer,
                "q_r95": int(q.r95),
                "q_entropy_rank": float(q.entropy_rank),
                "delta_timestep_r95": int(delta_t.r95),
                "delta_aggregated_r95": int(delta_a.r95),
                "delta_timestep_entropy_rank": float(delta_t.entropy_rank),
                "delta_aggregated_entropy_rank": float(delta_a.entropy_rank),
                "gate_temporal_cosine": float(gate.mean_temporal_cosine),
                "credit_temporal_coherence": float(delta_t.temporal_coherence),
                "gate_entropy_rank": float(gate.entropy_rank),
                "ge": float(delta_t.r95 / max(float(q.r95), 1.0)),
                "source_probe_split": "basis_eval",
                "source_residualization": "raw",
            }
        )
    result = pd.DataFrame(rows).sort_values(["layer", "seed", "epoch"])
    expected = 3 * 3 * 6
    if len(result) != expected:
        raise ValueError(f"expected {expected} trajectory rows, got {len(result)}")
    return result


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for layer in LAYERS:
        layer_frame = frame[frame.layer.eq(layer)]
        for seed_label, subset in [("pooled", layer_frame)] + [
            (str(seed), group) for seed, group in layer_frame.groupby("seed")
        ]:
            for x_name in ("gate_temporal_cosine", "credit_temporal_coherence"):
                rows.append(
                    {
                        "layer": layer,
                        "seed_scope": seed_label,
                        "x_metric": x_name,
                        "y_metric": "ge",
                        "spearman_rho": float(
                            subset[x_name].rank(method="average").corr(
                                subset["ge"].rank(method="average")
                            )
                        ),
                        "observations": len(subset),
                        "interpretation": "descriptive follow-up; epoch observations are dependent; no p-value used",
                    }
                )
    return pd.DataFrame(rows)


def scatter_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    for layer in LAYERS:
        subset = frame[frame.layer.eq(layer)]
        ax.scatter(
            subset.gate_temporal_cosine,
            subset["ge"],
            s=54,
            alpha=.78,
            marker=MARKERS[layer],
            color=COLORS[layer],
            edgecolor="white",
            linewidth=.45,
            label=layer.replace("hidden_", "H"),
        )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1, label="GE = 1")
    ax.set_xlabel("Gate temporal cosine (higher = less differentiated)")
    ax.set_ylabel("Geometry expansion, GE = delta r95 / q r95")
    ax.set_title("DVS-Gesture: temporal gate diversity vs credit-space lifting")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def trajectory_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    for ax, layer in zip(axes, LAYERS):
        subset = frame[frame.layer.eq(layer)]
        summary = subset.groupby("epoch").agg(
            ge_mean=("ge", "mean"), ge_std=("ge", "std"),
            cosine_mean=("gate_temporal_cosine", "mean"),
            cosine_std=("gate_temporal_cosine", "std"),
        ).reset_index()
        for _seed, seed_frame in subset.groupby("seed"):
            ax.plot(seed_frame.epoch, seed_frame["ge"], color=COLORS[layer], alpha=.22, linewidth=1)
        ge_handle = ax.errorbar(
            summary.epoch, summary.ge_mean, yerr=summary.ge_std,
            color=COLORS[layer], marker="o", linewidth=2, capsize=3, label="GE",
        )
        ax2 = ax.twinx()
        for _seed, seed_frame in subset.groupby("seed"):
            ax2.plot(seed_frame.epoch, seed_frame.gate_temporal_cosine, color="#333333", alpha=.18, linewidth=1)
        cosine_handle = ax2.errorbar(
            summary.epoch, summary.cosine_mean, yerr=summary.cosine_std,
            color="#333333", marker="s", linestyle="--", linewidth=1.8, capsize=3,
            label="Gate cosine",
        )
        ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1)
        ax.set_title(layer.replace("hidden_", "H"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("GE", color=COLORS[layer])
        ax2.set_ylabel("Gate cosine", color="#333333")
        ax.grid(alpha=.18)
        ax.legend(
            [ge_handle, cosine_handle], ["GE", "Gate cosine"],
            frameon=False, loc="best",
        )
    fig.suptitle("Layer-selective lifting and temporal-state differentiation")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def findings(frame: pd.DataFrame, correlations_frame: pd.DataFrame) -> str:
    final = frame[frame.epoch.eq(frame.epoch.max())]
    summary = final.groupby("layer").agg(
        ge_mean=("ge", "mean"), ge_std=("ge", "std"),
        cosine_mean=("gate_temporal_cosine", "mean"), cosine_std=("gate_temporal_cosine", "std"),
    )
    initial = frame[frame.epoch.eq(frame.epoch.min())].groupby("layer")["ge"].mean()
    pooled = correlations_frame[correlations_frame.seed_scope.eq("pooled")]
    lines = [
        "# Experiment 04 — Stage A descriptive findings",
        "",
        "This is a post-registered descriptive analysis of reused Experiment 03 checkpoints; epochs are dependent observations and no p-values are reported.",
        "",
        "| Layer | GE epoch 0 | GE epoch 100 (mean±std) | Gate cosine epoch 100 (mean±std) |",
        "|---|---:|---:|---:|",
    ]
    for layer in LAYERS:
        row = summary.loc[layer]
        lines.append(
            f"| {layer.replace('hidden_', 'H')} | {initial.loc[layer]:.3f} | "
            f"{row.ge_mean:.3f}±{row.ge_std:.3f} | {row.cosine_mean:.4f}±{row.cosine_std:.4f} |"
        )
    lines.extend(["", "## Descriptive Spearman correlations", "", "| Layer | x | rho |", "|---|---|---:|"])
    for row in pooled.itertuples():
        lines.append(f"| {row.layer.replace('hidden_', 'H')} | {row.x_metric} | {row.spearman_rho:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.source_geometry)
    frame = trajectory(source)
    corr = correlations(frame)
    frame.to_csv(args.output_dir / "dvs_temporal_lifting_trajectory.csv", index=False)
    corr.to_csv(args.output_dir / "dvs_trajectory_correlations.csv", index=False)
    scatter_plot(frame, args.output_dir / "dvs_gate_diversity_vs_lifting.png")
    trajectory_plot(frame, args.output_dir / "dvs_layerwise_trajectory.png")
    (args.output_dir / "STAGE_A_FINDINGS.md").write_text(
        findings(frame, corr), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "rows": len(frame), "correlations": len(corr)}))


if __name__ == "__main__":
    main()
