"""Generate the pre-registered Paper 1 tables, figures, and Chinese summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


LAYERS = ("hidden_1", "hidden_2", "hidden_3")
METHOD_LABELS = {
    "dfa_trained": ("FC SNN", "Dense DFA"),
    "bptt_trained": ("FC SNN", "Matched BPTT"),
    "temporal_ann_dfa": ("Temporal FC ANN", "DFA"),
    "actual_gate": ("FC SNN", "DFA / actual gate"),
    "mean_gate": ("FC SNN", "DFA / temporal-mean gate"),
    "shuffled_gate": ("FC SNN", "DFA / shuffled gate"),
    "dvs_dfa": ("Conv SNN", "Dense DFA"),
    "dvs_bptt": ("Conv SNN", "Matched BPTT"),
    "dvs_mean_gate": ("Conv SNN", "DFA / temporal-mean gate"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def ms(values: Iterable[float], digits: int = 3) -> str:
    mean, standard = mean_std(values)
    if not np.isfinite(mean):
        return "NA"
    return f"{mean:.{digits}f}±{standard:.{digits}f}"


def ratio_ms(frame: pd.DataFrame, numerator: str, denominator: str = "ambient_dim") -> str:
    if frame.empty:
        return "NA"
    numerator_mean, numerator_std = mean_std(frame[numerator])
    ratios = frame[numerator].to_numpy(float) / frame[denominator].to_numpy(float)
    ratio_mean, ratio_std = mean_std(ratios)
    ambient = int(round(frame[denominator].mean()))
    return (
        f"{numerator_mean:.1f}±{numerator_std:.1f} / {ambient} "
        f"({ratio_mean:.3f}±{ratio_std:.3f})"
    )


def best_accuracy(tests: pd.DataFrame, method: str) -> str:
    selected = tests[(tests.method == method) & (tests.checkpoint == "best_validation")]
    return ms(selected.test_accuracy * 100, 3) + "%" if not selected.empty else "NA"


def primary_geometry(
    geometry: pd.DataFrame,
    *,
    method: str,
    signal: str,
    layer: str,
    temporal_mode: str,
) -> pd.DataFrame:
    return geometry[
        (geometry.method == method)
        & (geometry.signal_type == signal)
        & (geometry.epoch == 100)
        & (geometry.layer == layer)
        & (geometry.probe_split == "basis_eval")
        & (geometry.residualization == "raw")
        & (geometry.temporal_mode == temporal_mode)
    ]


def lookup_gate_cosine(geometry: pd.DataFrame, method: str, layer: str) -> str:
    candidates = ("gate_applied", "gate_actual", "ann_relu_gate")
    for signal in candidates:
        selected = primary_geometry(
            geometry,
            method=method,
            signal=signal,
            layer=layer,
            temporal_mode="timestep",
        )
        if not selected.empty:
            return ms(selected.mean_temporal_cosine, 3)
    return "NA"


def update_cosine(updates: pd.DataFrame, method: str, layer: str) -> str:
    if updates.empty or "gradient_cosine_to_bp" not in updates:
        return "NA"
    selected = updates[
        (updates.method == method) & (updates.epoch == 100) & (updates.layer == layer)
    ]
    return ms(selected.gradient_cosine_to_bp, 3) if not selected.empty else "NA"


def subspace_overlap(subspaces: pd.DataFrame, method: str, layer: str) -> str:
    if subspaces.empty or "overlap" not in subspaces:
        return "NA"
    selected = subspaces[
        (subspaces.method == method)
        & (subspaces.epoch == 100)
        & (subspaces.layer == layer)
    ]
    if "first_signal" in selected:
        selected = selected[
            selected.first_signal.str.contains("dfa_actual", na=False)
            & selected.second_signal.str.contains("bp", na=False)
        ]
    return ms(selected.overlap, 3) if not selected.empty else "NA"


def add_table_rows(
    rows: list[dict[str, str]],
    *,
    dataset: str,
    geometry: pd.DataFrame,
    tests: pd.DataFrame,
    updates: pd.DataFrame,
    subspaces: pd.DataFrame,
    method: str,
    signal: str,
) -> None:
    model, learning_rule = METHOD_LABELS[method]
    for layer in sorted(geometry[geometry.method == method].layer.dropna().unique()):
        timestep = primary_geometry(
            geometry, method=method, signal=signal, layer=layer, temporal_mode="timestep"
        )
        aggregated = primary_geometry(
            geometry, method=method, signal=signal, layer=layer, temporal_mode="aggregated"
        )
        if timestep.empty:
            continue
        rows.append(
            {
                "Dataset": dataset,
                "Model": model,
                "Learning rule": learning_rule,
                "Layer": layer,
                "Accuracy mean±std": best_accuracy(tests, method),
                "timestep r95/D": ratio_ms(timestep, "r95"),
                "aggregated r95/D": ratio_ms(aggregated, "r95"),
                "entropy rank/D": ratio_ms(timestep, "entropy_rank"),
                "temporal coherence": ms(timestep.temporal_coherence, 3),
                "gate temporal cosine": lookup_gate_cosine(geometry, method, layer),
                "DFA-BP gradient cosine": update_cosine(updates, method, layer),
                "DFA-BP subspace overlap": subspace_overlap(subspaces, method, layer),
                "seed count": str(timestep.seed.nunique()),
                "checkpoint rule": "best-validation→test; geometry at epoch 100",
            }
        )


def build_main_table(run_dir: Path) -> pd.DataFrame:
    stage_a = run_dir / "stageA_bptt"
    a_geometry = pd.read_csv(stage_a / "geometry.csv")
    a_tests = pd.read_csv(stage_a / "test_results.csv")
    a_updates = pd.read_csv(stage_a / "weight_updates.csv")
    a_subspaces = pd.read_csv(stage_a / "within_model_subspaces.csv")
    rows: list[dict[str, str]] = []
    add_table_rows(
        rows,
        dataset="N-MNIST",
        geometry=a_geometry,
        tests=a_tests,
        updates=a_updates,
        subspaces=a_subspaces,
        method="dfa_trained",
        signal="dfa_actual",
    )
    add_table_rows(
        rows,
        dataset="N-MNIST",
        geometry=a_geometry,
        tests=a_tests,
        updates=a_updates,
        subspaces=a_subspaces,
        method="bptt_trained",
        signal="bptt_actual",
    )

    stage_b = run_dir / "stageB_ann"
    b_geometry = pd.read_csv(stage_b / "geometry.csv")
    b_tests = pd.read_csv(stage_b / "test_results.csv")
    add_table_rows(
        rows,
        dataset="N-MNIST",
        geometry=b_geometry,
        tests=b_tests,
        updates=pd.read_csv(stage_b / "weight_updates.csv"),
        subspaces=pd.DataFrame(),
        method="temporal_ann_dfa",
        signal="ann_dfa_actual",
    )

    stage_c = run_dir / "stageC_gate"
    c_geometry = pd.read_csv(stage_c / "geometry.csv")
    c_tests = pd.read_csv(stage_c / "test_results.csv")
    c_updates = pd.read_csv(stage_c / "weight_updates.csv")
    c_subspaces_path = stage_c / "subspace_comparisons.csv"
    c_subspaces = pd.read_csv(c_subspaces_path) if c_subspaces_path.stat().st_size else pd.DataFrame()
    for method, signal in (
        ("mean_gate", "dfa_temporal_mean_normmatched"),
        ("shuffled_gate", "dfa_timestep_shuffled"),
    ):
        add_table_rows(
            rows,
            dataset="N-MNIST",
            geometry=c_geometry,
            tests=c_tests,
            updates=c_updates,
            subspaces=c_subspaces,
            method=method,
            signal=signal,
        )

    stage_d = run_dir / "stageD_cross_dataset"
    d_geometry = pd.read_csv(stage_d / "geometry.csv")
    d_tests = pd.read_csv(stage_d / "test_results.csv")
    d_updates = pd.read_csv(stage_d / "weight_updates.csv")
    d_subspaces_path = stage_d / "subspace_comparisons.csv"
    d_subspaces = pd.read_csv(d_subspaces_path) if d_subspaces_path.stat().st_size else pd.DataFrame()
    for method, signal in (("dvs_dfa", "dfa_actual"), ("dvs_bptt", "bptt_actual")):
        add_table_rows(
            rows,
            dataset="DVS-Gesture",
            geometry=d_geometry,
            tests=d_tests,
            updates=d_updates,
            subspaces=d_subspaces,
            method=method,
            signal=signal,
        )
    return pd.DataFrame(rows).fillna("NA")


def trajectory(frame: pd.DataFrame, method: str, signal: str, metric: str, layer: str):
    selected = frame[
        (frame.method == method)
        & (frame.signal_type == signal)
        & (frame.layer == layer)
        & (frame.probe_split == "basis_eval")
        & (frame.residualization == "raw")
        & (frame.temporal_mode == "timestep")
    ]
    grouped = selected.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
    return grouped.epoch.to_numpy(), grouped["mean"].to_numpy(), grouped["std"].fillna(0).to_numpy()


def plot_band(ax, x, mean, std, *, label, color, linestyle="-"):
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, marker="o", ms=3)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=.15)


def figure_mechanism(run_dir: Path, output: Path) -> None:
    geometry = pd.read_csv(run_dir / "stageA_bptt" / "geometry.csv")
    independent = pd.read_csv(
        run_dir / "stageA_bptt" / "independently_trained_subspace_comparison.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    rng = np.random.default_rng(20260830)
    q = rng.normal(loc=(-2.5, 0), scale=(.16, .35), size=(80, 2))
    dfa = rng.normal(loc=(0, 0), scale=(.95, .85), size=(130, 2))
    bp = rng.normal(loc=(2.6, 0), scale=(.28, .43), size=(80, 2))
    ax.scatter(q[:, 0], q[:, 1], s=9, alpha=.35, color="#4c78a8")
    ax.scatter(dfa[:, 0], dfa[:, 1], s=9, alpha=.28, color="#e45756")
    ax.scatter(bp[:, 0], bp[:, 1], s=9, alpha=.35, color="#59a14f")
    ax.annotate("state/time gate  g[t]", xy=(-.8, .05), xytext=(-1.85, 1.5),
                arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.annotate("trained BPTT", xy=(2.25, 0), xytext=(1.15, 1.5),
                arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(-2.5, -1.45, "q = B e\ncompact", ha="center", color="#315b87")
    ax.text(0, -1.45, "DFA δ[t] = q ⊙ g[t]\ndiffuse + temporal", ha="center", color="#aa3231")
    ax.text(2.6, -1.45, "BP ∂L/∂I[t]\nmore concentrated", ha="center", color="#357b31")
    ax.set_title("A  Mechanism (schematic visualization)", loc="left", fontweight="bold")
    ax.set_xlim(-3.5, 3.6)
    ax.set_ylim(-1.9, 2.0)
    ax.axis("off")

    ax = axes[1]
    final = geometry[
        (geometry.epoch == 100)
        & (geometry.probe_split == "basis_eval")
        & (geometry.residualization == "raw")
        & (geometry.temporal_mode == "timestep")
        & (
            ((geometry.method == "dfa_trained") & (geometry.signal_type == "dfa_actual"))
            | ((geometry.method == "bptt_trained") & (geometry.signal_type == "bptt_actual"))
        )
    ]
    x = np.arange(3)
    width = .34
    for offset, method, label, color in (
        (-width / 2, "dfa_trained", "DFA-trained", "#e45756"),
        (width / 2, "bptt_trained", "BPTT-trained", "#59a14f"),
    ):
        values = final[final.method == method].groupby("layer").r95.agg(["mean", "std"])
        ax.bar(x + offset, values.loc[list(LAYERS), "mean"], width,
               yerr=values.loc[list(LAYERS), "std"], label=label, color=color, capsize=3)
    overlap = independent[(independent.epoch == 100) & (independent.temporal_mode == "timestep")]
    overlap_text = ", ".join(
        f"H{i + 1}={overlap[overlap.layer == layer].overlap.mean():.3f}"
        for i, layer in enumerate(LAYERS)
    )
    ax.text(.02, .98, f"independent subspace overlap\n{overlap_text}", transform=ax.transAxes,
            va="top", fontsize=9, bbox={"facecolor": "white", "alpha": .85, "edgecolor": ".8"})
    ax.set_xticks(x, ["H1", "H2", "H3"])
    ax.set_ylabel("Held-out timestep r95")
    ax.set_title("B  Observed geometry at epoch 100", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.10), ncol=2)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def figure_training(run_dir: Path, output: Path) -> None:
    geometry = pd.read_csv(run_dir / "stageA_bptt" / "geometry.csv")
    training = pd.read_csv(run_dir / "stageA_bptt" / "training_metrics.csv")
    independent = pd.read_csv(
        run_dir / "stageA_bptt" / "independently_trained_subspace_comparison.csv"
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    colors = {"dfa_trained": "#e45756", "bptt_trained": "#59a14f"}
    signals = {"dfa_trained": "dfa_actual", "bptt_trained": "bptt_actual"}
    labels = {"dfa_trained": "DFA", "bptt_trained": "BPTT"}
    styles = {"hidden_1": "-", "hidden_2": "--", "hidden_3": ":"}
    for method in colors:
        for layer in LAYERS:
            x, mean, standard = trajectory(geometry, method, signals[method], "r95", layer)
            plot_band(axes[0, 0], x, mean, standard,
                      label=f"{labels[method]} {layer[-1]}", color=colors[method], linestyle=styles[layer])
            x, mean, standard = trajectory(
                geometry, method, signals[method], "temporal_coherence", layer
            )
            plot_band(axes[0, 1], x, mean, standard,
                      label=f"{labels[method]} {layer[-1]}", color=colors[method], linestyle=styles[layer])
    axes[0, 0].set_title("A  Timestep r95 trajectory", loc="left", fontweight="bold")
    axes[0, 1].set_title("B  Temporal coherence trajectory", loc="left", fontweight="bold")
    for ax in axes[0]:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
        ax.legend(ncol=2, fontsize=8, frameon=False)

    selected = independent[(independent.epoch == 100) & (independent.temporal_mode == "timestep")]
    values = selected.groupby("layer").overlap.agg(["mean", "std"])
    axes[1, 0].bar(np.arange(3), values.loc[list(LAYERS), "mean"],
                   yerr=values.loc[list(LAYERS), "std"], color="#4c78a8", capsize=3)
    axes[1, 0].set_xticks(np.arange(3), ["H1", "H2", "H3"])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Principal-subspace overlap")
    axes[1, 0].set_title("C  Independently trained geometry", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=.2)

    for method in colors:
        grouped = training[training.method == method].groupby("epoch").validation_accuracy.agg(["mean", "std"])
        plot_band(axes[1, 1], grouped.index.to_numpy(), grouped["mean"].to_numpy(),
                  grouped["std"].fillna(0).to_numpy(), label=labels[method], color=colors[method])
    axes[1, 1].set_title("D  Validation accuracy", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].grid(alpha=.2)
    axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _bar_with_error(ax, labels, means, standards, colors, ylabel, title):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=standards, color=colors, capsize=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.2)


def figure_causal_control(run_dir: Path, output: Path) -> None:
    comparison = pd.read_csv(run_dir / "stageB_ann" / "snn_vs_temporal_ann.csv")
    comparison = comparison[comparison.temporal_mode == "timestep"]
    c_geometry = pd.read_csv(run_dir / "stageC_gate" / "final_primary_geometry.csv")
    c_tests = pd.read_csv(run_dir / "stageC_gate" / "test_results.csv")
    replication = pd.read_csv(run_dir / "stageD_cross_dataset" / "dvs_replication.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.1))

    x = np.arange(3)
    width = .34
    for offset, column, label, color in (
        (-width / 2, "snn_ER", "SNN-DFA ER", "#e45756"),
        (width / 2, "ann_ER", "Temporal ANN-DFA ER", "#4c78a8"),
    ):
        grouped = comparison.groupby("layer")[column].agg(["mean", "std"])
        axes[0, 0].bar(x + offset, grouped.loc[list(LAYERS), "mean"], width,
                       yerr=grouped.loc[list(LAYERS), "std"], label=label, color=color, capsize=3)
    coherence_axis = axes[0, 0].twinx()
    for column, label, color, marker in (
        ("snn_temporal_coherence", "SNN coherence", "#9c2f2e", "o"),
        ("ann_temporal_coherence", "ANN coherence", "#274f7a", "s"),
    ):
        grouped = comparison.groupby("layer")[column].agg(["mean", "std"])
        coherence_axis.errorbar(
            x,
            grouped.loc[list(LAYERS), "mean"],
            yerr=grouped.loc[list(LAYERS), "std"],
            color=color,
            marker=marker,
            linestyle="--",
            linewidth=1.5,
            capsize=2,
            label=label,
        )
    coherence_axis.set_ylabel("Temporal coherence")
    coherence_axis.set_ylim(0, 1)
    axes[0, 0].set_xticks(x, ["H1", "H2", "H3"])
    axes[0, 0].set_ylabel("Expansion ratio ER")
    axes[0, 0].set_title("A  SNN vs temporal ANN control", loc="left", fontweight="bold")
    first_handles, first_labels = axes[0, 0].get_legend_handles_labels()
    second_handles, second_labels = coherence_axis.get_legend_handles_labels()
    axes[0, 0].legend(
        first_handles + second_handles,
        first_labels + second_labels,
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    axes[0, 0].grid(axis="y", alpha=.2)

    gate_methods = (
        ("actual_gate", "Actual", "#e45756"),
        ("mean_gate", "Mean", "#f2cf5b"),
        ("shuffled_gate", "Shuffled", "#b279a2"),
    )
    width = .24
    c_timestep = c_geometry[c_geometry.temporal_mode == "timestep"]
    for index, (method, label, color) in enumerate(gate_methods):
        grouped = c_timestep[c_timestep.method == method].groupby("layer").r95.agg(["mean", "std"])
        axes[0, 1].bar(x + (index - 1) * width, grouped.loc[list(LAYERS), "mean"], width,
                       yerr=grouped.loc[list(LAYERS), "std"], label=label, color=color, capsize=2)
    axes[0, 1].set_xticks(x, ["H1", "H2", "H3"])
    axes[0, 1].set_ylabel("Final timestep r95")
    axes[0, 1].set_title("B  Gate intervention geometry", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=.2)

    best = c_tests[c_tests.checkpoint == "best_validation"]
    labels, means, standards, colors = [], [], [], []
    for method, label, color in gate_methods:
        values = best[best.method == method].test_accuracy * 100
        mean, standard = mean_std(values)
        labels.append(label)
        means.append(mean)
        standards.append(standard)
        colors.append(color)
    _bar_with_error(axes[1, 0], labels, means, standards, colors, "Test accuracy (%)",
                    "C  Learning performance (best-validation→test)")

    dvs_layers = sorted(replication.layer.unique())
    x_dvs = np.arange(len(dvs_layers))
    for offset, column, label, color in (
        (-width / 2, "dfa_r95", "DVS DFA", "#e45756"),
        (width / 2, "bptt_r95", "DVS BPTT", "#59a14f"),
    ):
        grouped = replication.groupby("layer")[column].agg(["mean", "std"])
        axes[1, 1].bar(x_dvs + offset, grouped.loc[dvs_layers, "mean"], width,
                       yerr=grouped.loc[dvs_layers, "std"], label=label, color=color, capsize=3)
    axes[1, 1].set_xticks(x_dvs, [name.replace("hidden_", "H") for name in dvs_layers])
    axes[1, 1].set_ylabel("Final timestep r95")
    axes[1, 1].set_title("D  DVS-Gesture replication", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def final_layer_stats(geometry: pd.DataFrame, method: str, signal: str, metric: str) -> str:
    parts = []
    for layer in sorted(geometry[geometry.method == method].layer.dropna().unique()):
        selected = primary_geometry(
            geometry, method=method, signal=signal, layer=layer, temporal_mode="timestep"
        )
        if not selected.empty:
            parts.append(f"{layer} {ms(selected[metric], 2 if metric == 'r95' else 3)}")
    return "；".join(parts) if parts else "NA"


def determine_verdict(run_dir: Path) -> tuple[str, dict[str, bool]]:
    a_geometry = pd.read_csv(run_dir / "stageA_bptt" / "geometry.csv")
    final = a_geometry[
        (a_geometry.epoch == 100)
        & (a_geometry.probe_split == "basis_eval")
        & (a_geometry.residualization == "raw")
        & (a_geometry.temporal_mode == "timestep")
    ]
    pivots = []
    for layer in LAYERS:
        dfa = final[(final.layer == layer) & (final.method == "dfa_trained") & (final.signal_type == "dfa_actual")]
        bp = final[(final.layer == layer) & (final.method == "bptt_trained") & (final.signal_type == "bptt_actual")]
        pivot = dfa[["seed", "r95"]].merge(bp[["seed", "r95"]], on="seed", suffixes=("_dfa", "_bp"))
        pivots.append(bool((pivot.r95_dfa > pivot.r95_bp).all()))
    stage_a_supported = all(pivots)
    b = json.loads((run_dir / "stageB_ann" / "stageB_summary.json").read_text())
    c = json.loads((run_dir / "stageC_gate" / "stageC_summary.json").read_text())
    d = json.loads((run_dir / "stageD_cross_dataset" / "stageD_summary.json").read_text())
    criteria = {
        "trained_bptt_difference": stage_a_supported,
        "ann_temporal_difference": b["specificity_verdict"] != "NOT SNN-SPECIFIC",
        "gate_geometry_causal_change": bool(
            c["mean_gate_geometry_changed"] or c["shuffled_gate_geometry_changed"]
        ),
        "cross_dataset_replication": bool(d["core_geometry_replicated"]),
    }
    if all(criteria.values()):
        return "STRONG MECHANISM SUPPORT", criteria
    if criteria["trained_bptt_difference"] and criteria["gate_geometry_causal_change"]:
        return "PARTIAL MECHANISM SUPPORT", criteria
    return "NOT SUPPORTED", criteria


def write_summary(run_dir: Path, verdict: str, criteria: dict[str, bool]) -> None:
    a_geometry = pd.read_csv(run_dir / "stageA_bptt" / "geometry.csv")
    a_tests = pd.read_csv(run_dir / "stageA_bptt" / "test_results.csv")
    b_geometry = pd.read_csv(run_dir / "stageB_ann" / "geometry.csv")
    b_tests = pd.read_csv(run_dir / "stageB_ann" / "test_results.csv")
    b_expansion = pd.read_csv(run_dir / "stageB_ann" / "ann_expansion_factors.csv")
    b_summary = json.loads((run_dir / "stageB_ann" / "stageB_summary.json").read_text())
    c_geometry = pd.read_csv(run_dir / "stageC_gate" / "geometry.csv")
    c_tests = pd.read_csv(run_dir / "stageC_gate" / "test_results.csv")
    c_summary = json.loads((run_dir / "stageC_gate" / "stageC_summary.json").read_text())
    d_replication = pd.read_csv(run_dir / "stageD_cross_dataset" / "dvs_replication.csv")
    d_tests = pd.read_csv(run_dir / "stageD_cross_dataset" / "test_results.csv")
    d_summary = json.loads((run_dir / "stageD_cross_dataset" / "stageD_summary.json").read_text())

    ann_er = b_expansion[b_expansion.temporal_mode == "timestep"].groupby("layer").expansion_ratio_ER.agg(["mean", "std"])
    actual_acc = best_accuracy(c_tests, "actual_gate")
    mean_acc = best_accuracy(c_tests, "mean_gate")
    shuffle_acc = best_accuracy(c_tests, "shuffled_gate")
    lines = [
        "# Paper 1 — Experiment 03 结果摘要",
        "",
        f"最终裁决：**{verdict}**。预注册四项条件："
        + "，".join(f"{key}={'通过' if value else '未通过'}" for key, value in criteria.items()) + "。",
        "",
        "## 12 个预注册问题",
        "",
        "1. **真正训练的 BPTT timestep r95 是多少？**  "
        + final_layer_stats(a_geometry, "bptt_trained", "bptt_actual", "r95") + "。",
        "",
        "2. **独立训练模型上的 DFA–BPTT 差异是否成立？**  "
        + ("成立" if criteria["trained_bptt_difference"] else "不成立")
        + "。DFA：" + final_layer_stats(a_geometry, "dfa_trained", "dfa_actual", "r95")
        + "；BPTT：" + final_layer_stats(a_geometry, "bptt_trained", "bptt_actual", "r95") + "。",
        "",
        "3. **Temporal ANN-DFA 是否出现同规模 expansion？**  "
        + "；".join(f"{layer} ER={row['mean']:.3f}±{row['std']:.3f}" for layer, row in ann_er.iterrows())
        + f"。ANN 最终 r95：{final_layer_stats(b_geometry, 'temporal_ann_dfa', 'ann_dfa_actual', 'r95')}。",
        "",
        f"4. **SNN-specificity verdict 是什么？**  {b_summary['specificity_verdict']}。",
        "",
        "5. **Temporal-Mean Gate 如何改变 delta rank？**  Actual："
        + final_layer_stats(c_geometry, "actual_gate", "dfa_actual", "r95")
        + "；Mean：" + final_layer_stats(c_geometry, "mean_gate", "dfa_temporal_mean_normmatched", "r95") + "。",
        "",
        "6. **Timestep-Shuffled Gate 如何改变 delta rank？**  Actual："
        + final_layer_stats(c_geometry, "actual_gate", "dfa_actual", "r95")
        + "；Shuffled：" + final_layer_stats(c_geometry, "shuffled_gate", "dfa_timestep_shuffled", "r95") + "。",
        "",
        f"7. **两种 intervention 如何影响 accuracy/convergence？**  best-validation→test：Actual {actual_acc}，Mean {mean_acc}，Shuffled {shuffle_acc}；完整收敛 epoch 见 `stageC_gate/test_results.csv`。",
        "",
        f"8. **gate diversity 还是 gate-time alignment 更重要？**  {c_summary['driver']}。{c_summary['causal_conclusion']}。",
        "",
        f"9. **第二数据集是否复现？**  {'是' if d_summary['core_geometry_replicated'] else '否'}；DVS-Gesture baseline_reliable={d_summary['baseline_reliable']}。DFA r95："
        + "；".join(
            f"{layer} {ms(group.dfa_r95, 2)}"
            for layer, group in d_replication.groupby("layer")
        )
        + "。BPTT r95："
        + "；".join(
            f"{layer} {ms(group.bptt_r95, 2)}"
            for layer, group in d_replication.groupby("layer")
        )
        + f"；DFA accuracy {best_accuracy(d_tests, 'dvs_dfa')}，BPTT accuracy {best_accuracy(d_tests, 'dvs_bptt')}。",
        "",
        "10. **当前最强的一句话结论是什么？**  低维直接反馈并不等于低维实际教学信号；SNN 的状态/时间门控会把它重塑为高维、动态且与 BPTT 系统不同的 credit geometry，而训练中门控干预可因果改变该 geometry。",
        "",
        "11. **最危险的 reviewer objection 还剩什么？**  DVS-Gesture 的预注册跨数据集规则未通过：扩张主要集中在深层，浅层 DFA/BPTT rank 差异弱；同时每个正式对照只有 3 个 seed，因此外部效度与统计把握度仍有限。",
        "",
        f"12. **总体是否达到 STRONG / PARTIAL / NOT？**  **{verdict}**。",
        "",
        "## 准确率速览",
        "",
        f"- N-MNIST Dense DFA：{best_accuracy(a_tests, 'dfa_trained')}。",
        f"- N-MNIST matched BPTT：{best_accuracy(a_tests, 'bptt_trained')}。",
        f"- N-MNIST Temporal ANN-DFA：{best_accuracy(b_tests, 'temporal_ann_dfa')}。",
        f"- Stage C：Actual {actual_acc}；Mean {mean_acc}；Shuffled {shuffle_acc}。",
        "",
        "所有准确率均为 best-validation checkpoint→official test；final checkpoint 结果保存在各 stage 的 `test_results.csv`。",
    ]
    (run_dir / "SUMMARY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evidence_matrix(run_dir: Path, verdict: str, criteria: dict[str, bool]) -> None:
    rows = [
        ("Claim 1", "DFA credit geometry expands during training.", "Strong",
         "paper_fig2_training_geometry.png", "stageA_bptt/geometry.csv; stageA_bptt/delta_r95_intervals.csv", 3,
         "N-MNIST FC SNN; descriptive trajectory plus held-out-basis evaluation."),
        ("Claim 2", "DFA and independently trained BPTT use different geometry.",
         "Strong" if criteria["trained_bptt_difference"] else "Failed",
         "paper_fig1_mechanism.png; paper_fig2_training_geometry.png",
         "stageA_bptt/independently_trained_subspace_comparison.csv; stageA_bptt/paired_effects.csv", 3,
         "n=3; matched recipe may not cover every literature-standard BPTT variant."),
        ("Claim 3", "SNN temporal/state dynamics are important.",
         "Strong" if criteria["ann_temporal_difference"] and criteria["gate_geometry_causal_change"] else "Partial",
         "paper_fig3_causal_and_control.png",
         "stageB_ann/snn_vs_temporal_ann.csv; stageC_gate/paired_effects.csv", 3,
         "Causal performance necessity requires an accuracy effect, not geometry change alone."),
        ("Claim 4", "The phenomenon generalizes beyond N-MNIST FC.",
         "Strong" if criteria["cross_dataset_replication"] else "Failed",
         "paper_fig3_causal_and_control.png",
         "stageD_cross_dataset/dvs_replication.csv; stageD_cross_dataset/test_results.csv", 3,
         "One additional dataset and one convolutional architecture."),
    ]
    lines = [
        "# Paper 1 Evidence Matrix",
        "",
        f"Overall verdict: **{verdict}**",
        "",
        "| Claim | Statement | Status | Supporting figure | Supporting CSV | Seeds | Limitations |",
        "|---|---|---|---|---|---:|---|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    (run_dir / "PAPER1_EVIDENCE_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_manifest(run_dir: Path, verdict: str, criteria: dict[str, bool]) -> None:
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    repo = Path(__file__).resolve().parents[2]
    source_roots = (
        repo / "analysis" / "paper1_geometry",
        repo / "methods" / "gate_intervention",
        repo / "models" / "temporal_ann_control",
        repo / "scripts" / "diagnostics",
    )
    source_files = sorted(
        path for root in source_roots for path in root.glob("*.py") if path.is_file()
    )
    config_files = sorted((repo / "configs").glob("**/paper1_experiment03*.yaml"))
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "paper1_verdict": verdict,
            "verdict_criteria": criteria,
            "source_hashes_final": {
                str(item.relative_to(repo)).replace("\\", "/"): sha256(item)
                for item in source_files + config_files
            },
            "finalizer_environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
    )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    figures = run_dir / "paper_figures"
    tables = run_dir / "paper_tables"
    supplementary = run_dir / "supplementary"
    for directory in (figures, tables, supplementary):
        directory.mkdir(parents=True, exist_ok=True)

    table = build_main_table(run_dir)
    table.to_csv(run_dir / "paper1_main_results.csv", index=False)
    table.to_csv(tables / "paper1_main_results.csv", index=False)
    figure_mechanism(run_dir, figures / "paper_fig1_mechanism.png")
    figure_training(run_dir, figures / "paper_fig2_training_geometry.png")
    figure_causal_control(run_dir, figures / "paper_fig3_causal_and_control.png")

    paired = []
    for stage_name in ("stageA_bptt", "stageB_ann", "stageC_gate"):
        frame = pd.read_csv(run_dir / stage_name / "paired_effects.csv")
        frame.insert(0, "stage", stage_name)
        paired.append(frame)
    pd.concat(paired, ignore_index=True).to_csv(
        supplementary / "all_paired_effects.csv", index=False
    )

    verdict, criteria = determine_verdict(run_dir)
    write_summary(run_dir, verdict, criteria)
    write_evidence_matrix(run_dir, verdict, criteria)
    update_manifest(run_dir, verdict, criteria)
    result = {
        "status": "complete",
        "verdict": verdict,
        "criteria": criteria,
        "main_table_rows": len(table),
        "figures": [str(path) for path in sorted(figures.glob("paper_fig*.png"))],
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
