#!/usr/bin/env python3
"""Regenerate the frozen v9 figures from archived per-seed CSV files only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


COLORS = {
    "SNN-DFA": "#245E93",
    "matched SNN-BPTT": "#A44A3F",
    "temporal ANN-DFA": "#577F42",
    "MeanGate": "#A67422",
    "ShuffledGate": "#775590",
}
MAIN = ["SNN-DFA", "matched SNN-BPTT", "temporal ANN-DFA"]
INTERVENTIONS = ["SNN-DFA", "MeanGate", "ShuffledGate"]
LAYERS = ["H1", "H2", "H3"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(root: Path) -> dict[str, Path]:
    return {
        "spectral": root / "results/frozen_sources/nmnist/spectral_metrics_per_seed.csv",
        "accuracy": root / "results/frozen_sources/nmnist/all_runs.csv",
        "trajectory": root / "results/frozen_sources/nmnist/trajectory_per_checkpoint.csv",
        "correlation": root / "results/frozen_sources/nmnist/correlation_breakdown.csv",
        "dvs": root / "results/frozen_sources/dvs/gate_per_seed.csv",
    }


def prepare_output(path: Path, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [item for item in path.iterdir() if item.is_file()]
    if existing and not force:
        raise FileExistsError(
            f"output directory is not empty; use --force for generated files only: {path}"
        )
    if force:
        for item in existing:
            if item.name.startswith(("fig1_", "fig2_", "fig3_", "accuracy_source")) or item.name == "figure_manifest.json":
                item.unlink()


def save_csv(frame: pd.DataFrame, output: Path, name: str) -> None:
    frame.to_csv(output / name, index=False)


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    for extension in ("svg", "pdf", "png"):
        figure.savefig(
            output / f"{name}.{extension}", dpi=300, bbox_inches="tight"
        )
    plt.close(figure)


def grouped(
    axis: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    methods: list[str],
    title: str,
    ylabel: str,
) -> None:
    for method_index, method in enumerate(methods):
        summary = (
            data[data.method == method]
            .groupby("layer")[metric]
            .agg(["mean", "std"])
            .reindex(LAYERS)
        )
        axis.bar(
            np.arange(3) + (method_index - (len(methods) - 1) / 2) * 0.23,
            summary["mean"],
            width=0.22,
            yerr=summary["std"],
            capsize=2,
            color=COLORS[method],
            label=method,
        )
    axis.set_xticks(range(3), LAYERS)
    axis.set_title(title, loc="left")
    axis.set_ylabel(ylabel)


def load_sources(root: Path) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    paths = source_paths(root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen figure source(s): " + "; ".join(missing))
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    spectral = frames["spectral"]
    frames["spectral"] = spectral[
        (spectral.probe_split == "basis_eval")
        & (spectral.temporal_mode == "timestep")
    ].copy()
    accuracy = frames["accuracy"]
    frames["accuracy"] = accuracy[
        accuracy.checkpoint_rule == "validation_accuracy_max_tie_lower_loss"
    ].copy()
    dvs = frames["dvs"]
    frames["dvs"] = dvs[
        dvs.gate_space == "canonical_spatial_mean_channels"
    ].copy()
    if not (
        frames["spectral"].groupby(["method", "layer"]).seed.nunique() == 5
    ).all():
        raise ValueError("spectral source is not a complete five-seed table")
    if not (frames["accuracy"].groupby("method").seed.nunique() == 5).all():
        raise ValueError("accuracy source is not a complete five-seed table")
    if frames["trajectory"].seed.nunique() != 5:
        raise ValueError("trajectory source is not a complete five-seed table")
    if frames["dvs"].seed.nunique() != 3:
        raise ValueError("DVS source is not a complete three-seed table")
    return paths, frames


def make_figure_1(frames: dict[str, pd.DataFrame], output: Path) -> None:
    trajectory = frames["trajectory"]
    figure = plt.figure(figsize=(7.15, 5.0), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.05])
    axis = figure.add_subplot(grid[0, :])
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("A  State-dependent local error modulation", loc="left")
    nodes = {
        "x": (0.07, 0.75, "Event $x_t$"),
        "u": (0.27, 0.75, "LIF state $u_t$"),
        "g": (0.49, 0.75, "Surrogate gate $g_t$"),
        "e": (0.07, 0.25, "Output error $e$"),
        "b": (0.27, 0.25, "Fixed feedback $B_l$"),
        "q": (0.49, 0.25, "Pre-gate $q_l$"),
        "d": (0.70, 0.50, "$\\delta_{l,t}=q_l\\odot g_t$"),
        "X": (0.91, 0.50, "Stack $X_l$\nSVD → spectrum"),
    }
    for x_coord, y_coord, label in nodes.values():
        axis.text(
            x_coord,
            y_coord,
            label,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#444444",
                "linewidth": 0.7,
            },
        )
    for source, target in (
        ("x", "u"),
        ("u", "g"),
        ("e", "b"),
        ("b", "q"),
        ("g", "d"),
        ("q", "d"),
        ("d", "X"),
    ):
        x_coord, y_coord, _ = nodes[source]
        target_x, target_y, _ = nodes[target]
        axis.annotate(
            "",
            xy=(target_x - 0.07, target_y),
            xytext=(x_coord + 0.07, y_coord),
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )

    axis = figure.add_subplot(grid[1, 0])
    endpoint = trajectory[trajectory.epoch == 100]
    for index, signal in enumerate(("pre-gate", "time-collapsed", "timestep")):
        summary = (
            endpoint[endpoint.signal == signal]
            .groupby("layer").r95.agg(["mean", "std"]).reindex(LAYERS)
        )
        axis.bar(
            np.arange(3) + (index - 1) * 0.24,
            summary["mean"],
            width=0.23,
            yerr=summary["std"],
            capsize=2,
            label=signal,
            color=["#949494", "#A67422", "#245E93"][index],
        )
    axis.set_xticks(range(3), LAYERS)
    axis.set_ylabel("$r_{95}$")
    axis.set_title("B  Signal hierarchy", loc="left")
    axis.legend(
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
    )

    axis = figure.add_subplot(grid[1, 1])
    selected = trajectory[
        (trajectory.signal == "timestep")
        & trajectory.epoch.isin([0, 10, 25, 50, 75, 100])
    ]
    for layer, color in zip(LAYERS, ["#245E93", "#A67422", "#775590"]):
        summary = selected[selected.layer == layer].groupby("epoch").r95_over_D.agg(
            ["mean", "std"]
        )
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            marker="o",
            markersize=3,
            capsize=2,
            label=layer,
            color=color,
        )
    axis.set_xlabel("Completed epoch")
    axis.set_ylabel("$r_{95}/D$")
    axis.set_title("C  SNN-DFA trajectory", loc="left")
    axis.legend(frameon=False, fontsize=7)
    save_figure(figure, output, "fig1_v9")


def make_figure_2(frames: dict[str, pd.DataFrame], output: Path) -> None:
    spectral = frames["spectral"]
    accuracy = frames["accuracy"]
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.2), layout="constrained")
    grouped(
        axes[0, 0],
        spectral,
        "r95",
        MAIN,
        "A  Timestep spectral dimension",
        "$r_{95}$",
    )
    axes[0, 0].legend(
        fontsize=6.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
    )
    axis = axes[0, 1]
    for index, method in enumerate(MAIN):
        values = accuracy[accuracy.method == method].test_accuracy * 100
        axis.bar(
            index,
            values.mean(),
            yerr=values.std(ddof=1),
            capsize=2,
            color=COLORS[method],
        )
    axis.set_xticks(range(3), ["SNN-DFA", "SNN-BPTT", "ANN-DFA"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("B  Validation-selected checkpoint", loc="left")

    axis = axes[1, 0]
    for method in MAIN:
        for layer, style in zip(LAYERS, ["-", "--", ":"]):
            values = spectral[
                (spectral.method == method) & (spectral.layer == layer)
            ][["r50", "r80", "r90", "r95", "r99"]].mean()
            axis.plot(
                [50, 80, 90, 95, 99],
                values,
                style,
                color=COLORS[method],
                linewidth=1,
            )
    axis.set_xlabel("Energy threshold (%)")
    axis.set_ylabel("Dimension")
    axis.set_title("C  Threshold dependence", loc="left")
    axis.text(
        0.03,
        0.97,
        "H1 —  H2 --  H3 ···",
        transform=axis.transAxes,
        va="top",
        fontsize=7,
    )

    axis = axes[1, 1]
    for method_index, method in enumerate(MAIN):
        for metric, marker in (("stable_rank", "o"), ("entropy_effective_rank", "s")):
            summary = (
                spectral[spectral.method == method]
                .groupby("layer")[metric]
                .agg(["mean", "std"])
                .reindex(LAYERS)
            )
            axis.errorbar(
                np.arange(3) + (method_index - 1) * 0.12,
                summary["mean"],
                yerr=summary["std"],
                marker=marker,
                linestyle="none",
                markersize=4,
                capsize=2,
                color=COLORS[method],
            )
    axis.set_xticks(range(3), LAYERS)
    axis.set_ylabel("Effective dimension")
    axis.set_title("D  Stable rank / entropy rank", loc="left")
    axis.text(
        0.98,
        0.97,
        "○ stable   □ entropy",
        transform=axis.transAxes,
        va="top",
        ha="right",
        fontsize=7,
    )
    save_figure(figure, output, "fig2_v9")


def make_figure_3(frames: dict[str, pd.DataFrame], output: Path) -> None:
    spectral = frames["spectral"]
    accuracy = frames["accuracy"]
    trajectory = frames["trajectory"]
    correlation = frames["correlation"]
    dvs = frames["dvs"]
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.2), layout="constrained")

    axis = axes[0, 0]
    for index, method in enumerate(INTERVENTIONS):
        values = accuracy[accuracy.method == method].test_accuracy * 100
        axis.bar(
            index,
            values.mean(),
            yerr=values.std(ddof=1),
            capsize=2,
            color=COLORS[method],
        )
    axis.set_xticks(range(3), ["Actual", "MeanGate", "Shuffle"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("A  Gate-time intervention", loc="left")

    ratios: list[dict[str, object]] = []
    actual = spectral[spectral.method == "SNN-DFA"].set_index(
        ["seed", "layer"]
    ).r95
    for method in INTERVENTIONS[1:]:
        for row in spectral[spectral.method == method].itertuples():
            ratios.append(
                {
                    "method": method,
                    "seed": row.seed,
                    "layer": row.layer,
                    "r95_ratio": row.r95 / actual.loc[(row.seed, row.layer)],
                }
            )
    ratio_frame = pd.DataFrame(ratios)
    save_csv(ratio_frame, output, "fig3_intervention_ratio_source.csv")
    grouped(
        axes[0, 1],
        ratio_frame,
        "r95_ratio",
        INTERVENTIONS[1:],
        "B  Spectral breadth relative to Actual",
        "Paired $r_{95}$ ratio",
    )
    axes[0, 1].axhline(1, color="gray", linestyle=":", linewidth=0.8)
    axes[0, 1].legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
    )

    axis = axes[1, 0]
    timestep = trajectory[trajectory.signal == "timestep"]
    save_csv(timestep, output, "fig3_correlation_source.csv")
    for layer, color in zip(LAYERS, ["#245E93", "#A67422", "#775590"]):
        group = timestep[timestep.layer == layer]
        axis.scatter(
            group.temporal_gate_cosine,
            group.r95_over_D,
            s=12,
            color=color,
            label=layer,
        )
    pooled = correlation[
        (correlation.group_type == "pooled")
        & (correlation.signal == "timestep")
    ]
    if len(pooled) != 1:
        raise ValueError(f"expected one pooled timestep correlation row, found {len(pooled)}")
    row = pooled.iloc[0]
    axis.text(
        0.44,
        0.95,
        f"Spearman ρ = {row.rho:.3f}\nn = {int(row.n)}",
        transform=axis.transAxes,
        va="top",
    )
    axis.set_xlabel("Temporal gate cosine")
    axis.set_ylabel("$r_{95}/D$")
    axis.set_title("C  Gate similarity and dimension", loc="left")
    axis.legend(frameon=False, fontsize=7, loc="lower left")

    axis = axes[1, 1]
    summary = (
        dvs.groupby("layer")[["GE", "temporal_gate_cosine"]]
        .agg(["mean", "std"])
        .reindex(LAYERS)
    )
    axis.bar(
        range(3),
        summary[("GE", "mean")],
        yerr=summary[("GE", "std")],
        capsize=2,
        color="#245E93",
    )
    axis.set_xticks(range(3), LAYERS)
    axis.set_ylabel("Expansion $r_{95}(δ)/r_{95}(q)$")
    secondary = axis.twinx()
    secondary.errorbar(
        range(3),
        summary[("temporal_gate_cosine", "mean")],
        yerr=summary[("temporal_gate_cosine", "std")],
        color="#A44A3F",
        marker="o",
        capsize=2,
    )
    secondary.set_ylim(0, 1.05)
    secondary.set_ylabel("Projected gate cosine", color="#A44A3F")
    axis.set_title("D  DVS boundary condition (3 seeds)", loc="left")
    save_figure(figure, output, "fig3_v9")


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_root,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "reproduce" / "_outputs" / "figures",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output_dir.resolve()
    prepare_output(output, args.force)

    paths, frames = load_sources(source_root)
    frames["spectral"] = frames["spectral"][
        (frames["spectral"].probe_split == "basis_eval")
        & (frames["spectral"].temporal_mode == "timestep")
    ]
    frames["accuracy"] = frames["accuracy"][
        frames["accuracy"].checkpoint_rule
        == "validation_accuracy_max_tie_lower_loss"
    ]
    frames["dvs"] = frames["dvs"][
        frames["dvs"].gate_space == "canonical_spatial_mean_channels"
    ]
    if not (
        frames["spectral"].groupby(["method", "layer"]).seed.nunique() == 5
    ).all():
        raise ValueError("spectral source is not a complete five-seed table")
    if not (frames["accuracy"].groupby("method").seed.nunique() == 5).all():
        raise ValueError("accuracy source is not a complete five-seed table")
    if frames["trajectory"].seed.nunique() != 5:
        raise ValueError("trajectory source is not a complete five-seed table")
    if frames["dvs"].seed.nunique() != 3:
        raise ValueError("DVS source is not a complete three-seed table")

    save_csv(frames["spectral"], output, "fig2_spectral_source.csv")
    save_csv(frames["accuracy"], output, "accuracy_source.csv")
    save_csv(
        frames["trajectory"], output, "fig1_trajectory_and_hierarchy_source.csv"
    )
    save_csv(frames["dvs"], output, "fig3_dvs_source.csv")
    make_figure_1(frames, output)
    make_figure_2(frames, output)
    make_figure_3(frames, output)

    manifest = {
        "status": "reproduced_from_frozen_per_seed_csvs",
        "operation": "figure generation only; no checkpoint replay; no training",
        "source_root": str(source_root),
        "N_MNIST_seed_count": 5,
        "DVS_seed_count": 3,
        "geometry_checkpoint_rule": "completed_epoch_100",
        "accuracy_checkpoint_rule": "validation_accuracy_max_tie_lower_loss",
        "error_bars": "sample SD across seeds, ddof=1",
        "source_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    main()
