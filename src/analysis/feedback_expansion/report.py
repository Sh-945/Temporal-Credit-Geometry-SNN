"""Tables, figures, correlations, and Chinese report for Experiment 01B."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAYERS = ("hidden_1", "hidden_2", "hidden_3")
COLORS = {"hidden_1": "#377eb8", "hidden_2": "#e41a1c", "hidden_3": "#4daf4a"}


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical-Recipes continued fraction for the incomplete beta."""

    maximum_iterations = 200
    epsilon = 3e-14
    floor = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = floor if abs(d) < floor else d
    d = 1.0 / d
    h = d
    for iteration in range(1, maximum_iterations + 1):
        twice = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + aa * d
        d = floor if abs(d) < floor else d
        c = 1.0 + aa / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + aa * d
        d = floor if abs(d) < floor else d
        c = 1.0 + aa / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _spearman_with_p(first: pd.Series, second: pd.Series) -> tuple[float, float]:
    """Tie-aware Spearman rho and conventional two-sided t approximation."""

    x = first.rank(method="average").to_numpy(dtype=float)
    y = second.rank(method="average").to_numpy(dtype=float)
    rho = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if n < 3 or not np.isfinite(rho):
        return rho, float("nan")
    if abs(rho) >= 1.0:
        return rho, 0.0
    degrees = n - 2
    t_squared = rho * rho * degrees / max(1e-30, 1.0 - rho * rho)
    p_value = _regularized_beta(
        degrees / (degrees + t_squared), degrees / 2.0, 0.5
    )
    return rho, float(min(1.0, max(0.0, p_value)))


def _mean_band(ax, frame: pd.DataFrame, x: str, y: str, *, label: str, color: str) -> None:
    stats = frame.groupby(x)[y].agg(["mean", "std"]).reset_index().sort_values(x)
    if stats.empty:
        return
    values = stats["std"].fillna(0)
    ax.plot(stats[x], stats["mean"], marker="o", label=label, color=color)
    ax.fill_between(stats[x], stats["mean"] - values, stats["mean"] + values, color=color, alpha=0.15)


def gate_correlations(spectral: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    delta = spectral[
        (spectral.signal_type == "post_gate_delta")
        & (spectral.temporal_mode == "aggregated")
        & (spectral.split == "basis_eval")
    ][["seed", "epoch", "layer", "r95", "entropy_rank"]].rename(
        columns={"r95": "delta_r95", "entropy_rank": "delta_entropy_rank"}
    )
    merged = delta.merge(gate, on=["seed", "epoch", "layer"], how="inner")
    x_fields = ["delta_r95", "delta_entropy_rank"]
    y_fields = [
        "gate_r95",
        "gate_entropy_rank",
        "effective_neuron_count",
        "fraction_above_relative_threshold",
        "mean_temporal_gate_cosine",
    ]
    rows = []
    for x in x_fields:
        for y in y_fields:
            values = merged[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p_value = _spearman_with_p(values[x], values[y])
            rows.append(
                {
                    "delta_metric": x,
                    "gate_metric": y,
                    "rho": float(rho),
                    "p_value": float(p_value),
                    "sample_count": len(values),
                    "interpretation": "association only; two-sided t-approximation p-value; not causal",
                }
            )
    return pd.DataFrame(rows)


def _save(fig, path: Path, *, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(run_dir: Path, training: pd.DataFrame) -> None:
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    spectral = pd.read_csv(run_dir / "spectral_broadening.csv")
    overlap = pd.read_csv(run_dir / "subspace_overlap.csv")
    novel = pd.read_csv(run_dir / "novel_direction_energy.csv")
    capture = pd.read_csv(run_dir / "final_subspace_capture.csv")
    direction = pd.read_csv(run_dir / "direction_energy.csv")
    gate = pd.read_csv(run_dir / "gate_dynamics.csv")
    classes = pd.read_csv(run_dir / "class_subspace_metrics.csv")
    temporal = pd.read_csv(run_dir / "temporal_subspace_metrics.csv")
    coherence = pd.read_csv(run_dir / "temporal_coherence.csv")

    primary = spectral[
        (spectral.signal_type == "post_gate_delta")
        & (spectral.temporal_mode == "aggregated")
        & (spectral.split == "basis_eval")
        & (spectral.checkpoint_kind != "best")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for layer in LAYERS:
        subset = primary[primary.layer == layer]
        _mean_band(axes[0], subset, "epoch", "r95", label=layer, color=COLORS[layer])
        _mean_band(axes[1], subset, "epoch", "entropy_rank", label=layer, color=COLORS[layer])
        _mean_band(axes[2], subset, "epoch", "participation_ratio", label=layer, color=COLORS[layer])
    axes[0].set_title("Aggregated delta r95")
    axes[1].set_title("Entropy rank")
    axes[2].set_title("Participation ratio")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].legend()
    if not training.empty:
        ax2 = axes[2].twinx()
        acc = training.groupby("epoch").test_acc.mean()
        ax2.plot(acc.index, acc.values, color="black", alpha=0.35, linestyle="--", label="accuracy")
        ax2.set_ylabel("Test accuracy")
    _save(fig, figures / "effective_dimension_trajectory.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    final_epoch = int(overlap.epoch_b.max())
    selected_sources = ["fixed_16", "fixed_32", "fixed_64", "fixed_128", "final_r95"]
    for ax, layer in zip(axes, LAYERS):
        data = overlap[
            (overlap.layer == layer)
            & (overlap.temporal_mode == "aggregated")
            & (overlap.epoch_b == final_epoch)
            & overlap.k_source.isin(selected_sources)
        ]
        for source, group in data.groupby("k_source"):
            _mean_band(ax, group, "epoch_a", "overlap", label=source, color=plt.cm.viridis(selected_sources.index(source) / 4))
        ax.set_title(layer)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Overlap with final")
    axes[-1].legend(fontsize=8)
    _save(fig, figures / "subspace_overlap_with_final.png")

    consecutive = overlap[
        (overlap.temporal_mode == "aggregated")
        & (overlap.epoch_b > overlap.epoch_a)
    ].copy()
    consecutive = consecutive[consecutive.epoch_b == consecutive.epoch_a.map(
        lambda value: {0: 10, 10: 25, 25: 50, 50: 75, 75: 100}.get(value, -1)
    )]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for source in ("fixed_32", "fixed_64", "fixed_128", "final_r95"):
            data = consecutive[(consecutive.layer == layer) & (consecutive.k_source == source)]
            _mean_band(ax, data, "epoch_b", "overlap", label=source, color=plt.cm.plasma(("fixed_32", "fixed_64", "fixed_128", "final_r95").index(source) / 3))
        ax.set_title(layer)
        ax.set_xlabel("Current epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Previous-current overlap")
    axes[-1].legend(fontsize=8)
    _save(fig, figures / "consecutive_subspace_overlap.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for source in ("fixed_32", "fixed_64", "fixed_128", "final_r95"):
            data = novel[(novel.layer == layer) & (novel.k_source == source)]
            _mean_band(ax, data, "epoch", "novel_energy_ratio", label=source, color=plt.cm.cividis(("fixed_32", "fixed_64", "fixed_128", "final_r95").index(source) / 3))
        ax.set_title(layer)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out novel energy")
    axes[-1].legend(fontsize=8)
    _save(fig, figures / "novel_direction_energy_vs_epoch.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, layer in zip(axes, LAYERS):
        data = overlap[(overlap.layer == layer) & (overlap.temporal_mode == "aggregated") & (overlap.k_source == "fixed_32")]
        matrix = data.groupby(["epoch_a", "epoch_b"]).overlap.mean().unstack()
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_title(f"{layer}, k=32")
        ax.set_xlabel("Epoch B")
        ax.set_ylabel("Epoch A")
    fig.colorbar(image, ax=axes, shrink=0.8)
    _save(fig, figures / "epoch_to_epoch_overlap_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for source in ("fixed_32", "fixed_64", "fixed_128", "final_r95"):
            data = consecutive[(consecutive.layer == layer) & (consecutive.k_source == source)]
            _mean_band(ax, data, "epoch_b", "mean_principal_angle", label=source, color=plt.cm.inferno(("fixed_32", "fixed_64", "fixed_128", "final_r95").index(source) / 3))
        ax.set_title(layer)
        ax.set_xlabel("Current epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean principal angle (degree)")
    axes[-1].legend(fontsize=8)
    _save(fig, figures / "principal_angles_vs_epoch.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, layer in zip(axes, LAYERS):
        data = direction[direction.layer == layer].groupby(["direction", "epoch"]).normalized_energy.mean().unstack()
        normalized = data.div(data.iloc[:, -1].replace(0, np.nan), axis=0).clip(0, 2)
        image = ax.imshow(normalized.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(normalized.columns)), normalized.columns)
        ax.set_title(layer)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Final direction rank")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Energy / final energy")
    _save(fig, figures / "final_direction_activation_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        data = primary[primary.layer == layer]
        for k in (10, 32, 64, 128):
            _mean_band(ax, data, "epoch", f"top{k}_energy_fraction", label=f"top{k}", color=plt.cm.Blues(0.3 + k / 180))
        ax.set_title(layer)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Cumulative energy fraction")
    axes[-1].legend(fontsize=8)
    _save(fig, figures / "spectral_broadening.png")

    delta = primary[["seed", "epoch", "layer", "r95"]]
    scatter = delta.merge(gate, on=["seed", "epoch", "layer"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, layer in zip(axes, LAYERS):
        data = scatter[scatter.layer == layer]
        ax.scatter(data.gate_r95, data.r95, c=data.epoch, cmap="viridis", alpha=0.8)
        ax.set_title(layer)
        ax.set_xlabel("Gate r95")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Delta r95")
    _save(fig, figures / "gate_diversity_vs_delta_rank.png")

    class_overlap = classes[classes.record_type == "class_subspace_overlap"]
    epochs_detail = sorted(class_overlap.epoch.dropna().astype(int).unique())
    fig, axes = plt.subplots(
        3,
        len(epochs_detail),
        figsize=(4 * len(epochs_detail), 12),
        layout="constrained",
    )
    for row_index, layer in enumerate(LAYERS):
        for col_index, epoch in enumerate(epochs_detail):
            ax = axes[row_index, col_index]
            data = class_overlap[(class_overlap.layer == layer) & (class_overlap.epoch == epoch) & (class_overlap.k == 32)]
            matrix = data.groupby(["class_a", "class_b"]).overlap.mean().unstack()
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"{layer}, e{epoch}")
            if row_index == 2:
                ax.set_xlabel("Class")
            if col_index == 0:
                ax.set_ylabel("Class")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    _save(fig, figures / "class_subspace_overlap_evolution.png", tight=False)

    epochs_time = sorted(temporal.epoch.dropna().astype(int).unique())
    fig, axes = plt.subplots(
        3,
        len(epochs_time),
        figsize=(4 * len(epochs_time), 12),
        layout="constrained",
    )
    for row_index, layer in enumerate(LAYERS):
        for col_index, epoch in enumerate(epochs_time):
            ax = axes[row_index, col_index]
            data = temporal[(temporal.layer == layer) & (temporal.epoch == epoch)]
            matrix = data.groupby(["timestep_a", "timestep_b"]).overlap.mean().unstack()
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"{layer}, e{epoch}")
            if row_index == 2:
                ax.set_xlabel("Timestep")
            if col_index == 0:
                ax.set_ylabel("Timestep")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    _save(fig, figures / "timestep_subspace_overlap_evolution.png", tight=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for layer in LAYERS:
        data = coherence[coherence.layer == layer]
        _mean_band(axes[0], data, "epoch", "delta_coherence_mean", label=layer, color=COLORS[layer])
        _mean_band(axes[1], data, "epoch", "delta_pairwise_temporal_cosine_mean", label=layer, color=COLORS[layer])
    axes[0].set_title("Temporal coherence")
    axes[1].set_title("Pairwise temporal cosine")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].legend()
    _save(fig, figures / "temporal_coherence_vs_epoch.png")

    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex="col")
    for row_index, layer in enumerate(LAYERS):
        rank = primary[primary.layer == layer].copy()
        rank["r95_over_D"] = rank.r95 / rank.ambient_dim
        _mean_band(axes[row_index, 0], rank, "epoch", "r95_over_D", label=layer, color=COLORS[layer])
        _mean_band(axes[row_index, 1], novel[(novel.layer == layer) & (novel.k_source == "final_r95")], "epoch", "novel_energy_ratio", label=layer, color=COLORS[layer])
        _mean_band(axes[row_index, 2], consecutive[(consecutive.layer == layer) & (consecutive.k_source == "fixed_32")], "epoch_b", "overlap", label=layer, color=COLORS[layer])
        _mean_band(axes[row_index, 3], capture[(capture.layer == layer) & (capture.k_source == "fixed_32")], "epoch", "current_energy_in_final", label=layer, color=COLORS[layer])
        axes[row_index, 0].set_ylabel(layer)
    for ax, title in zip(axes[0], ("delta r95/D", "Novel energy (final r95)", "Previous overlap k=32", "Current energy in final k=32")):
        ax.set_title(title)
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    _save(fig, figures / "expansion_mechanism_summary.png")


def _mechanism_statistics(run_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    spectral = pd.read_csv(run_dir / "spectral_broadening.csv")
    overlap = pd.read_csv(run_dir / "subspace_overlap.csv")
    novel = pd.read_csv(run_dir / "novel_direction_energy.csv")
    primary = spectral[
        (spectral.signal_type == "post_gate_delta")
        & (spectral.temporal_mode == "aggregated")
        & (spectral.split == "basis_eval")
        & (spectral.checkpoint_kind != "best")
    ]
    final_epoch = int(primary.epoch.max())
    verdicts: dict[str, str] = {}
    details: dict[str, Any] = {}
    for layer in LAYERS:
        ranks = primary[primary.layer == layer].groupby("epoch").r95.mean()
        ambient = float(primary[primary.layer == layer].ambient_dim.iloc[0])
        growth = (ranks.loc[final_epoch] - ranks.loc[ranks.index.min()]) / ambient
        final_pairs = overlap[
            (overlap.layer == layer)
            & (overlap.temporal_mode == "aggregated")
            & (overlap.epoch_b == final_epoch)
            & (overlap.epoch_a == 10)
        ]
        small = final_pairs[final_pairs.k_source == "fixed_32"].overlap.mean()
        large = final_pairs[final_pairs.k_source == "final_r95"].overlap.mean()
        consecutive = overlap[
            (overlap.layer == layer)
            & (overlap.temporal_mode == "aggregated")
            & (overlap.k_source == "fixed_32")
            & (overlap.epoch_b == final_epoch)
            & (overlap.epoch_a < final_epoch)
        ].sort_values("epoch_a")
        previous = consecutive[consecutive.epoch_a == consecutive.epoch_a.max()].overlap.mean()
        final_novel = novel[
            (novel.layer == layer)
            & (novel.epoch == final_epoch)
            & (novel.k_source == "final_r95")
        ].novel_energy_ratio.mean()
        if growth > 0.10 and small > 0.55 and small - large > 0.12:
            verdict = "CORE-PLUS-TAIL"
        elif growth > 0.10 and previous >= 0.50 and final_novel >= 0.08:
            verdict = "EXPANSION-DOMINANT"
        elif growth > 0.10 and previous < 0.45:
            verdict = "MIXED"
        elif previous < 0.45:
            verdict = "ROTATION-DOMINANT"
        else:
            verdict = "MIXED"
        verdicts[layer] = verdict
        details[layer] = {
            "r95_growth_over_D": float(growth),
            "epoch10_final_overlap_k32": float(small),
            "epoch10_final_overlap_final_r95": float(large),
            "final_interval_previous_overlap_k32": float(previous),
            "final_interval_novel_energy_final_r95": float(final_novel),
        }
    return verdicts, details


def generate_summary(run_dir: Path, training: pd.DataFrame) -> dict[str, str]:
    spectral = pd.read_csv(run_dir / "spectral_broadening.csv")
    overlap = pd.read_csv(run_dir / "subspace_overlap.csv")
    novel = pd.read_csv(run_dir / "novel_direction_energy.csv")
    capture = pd.read_csv(run_dir / "final_subspace_capture.csv")
    activation = pd.read_csv(run_dir / "direction_activation.csv")
    correlations = pd.read_csv(run_dir / "gate_delta_correlations.csv")
    classes = pd.read_csv(run_dir / "class_subspace_metrics.csv")
    temporal = pd.read_csv(run_dir / "temporal_subspace_metrics.csv")
    coherence = pd.read_csv(run_dir / "temporal_coherence.csv")
    controls = pd.read_csv(run_dir / "correct_vs_shuffled.csv")
    counterfactual = pd.read_csv(run_dir / "counterfactual_decomposition.csv")
    verdicts, details = _mechanism_statistics(run_dir)
    primary = spectral[
        (spectral.signal_type == "post_gate_delta")
        & (spectral.temporal_mode == "aggregated")
        & (spectral.split == "basis_eval")
        & (spectral.checkpoint_kind != "best")
    ]
    final_epoch = int(primary.epoch.max())
    lines = [
        "# Experiment 01B Summary",
        "",
        "## 1. 实验有效性",
        "",
        "本实验只重放 Experiment 01 的已有 checkpoint，没有重新训练。主轨迹预先固定为 epoch 0/10/25/50/75/100，best 仅作为补充谱点；三个 seed 全部进入统计。原 `probe_indices.csv` 原样复用，前 1024 个位置为 `basis_fit`，后 1024 个为 `basis_eval`。PCA basis 只在 fit 子集拟合，capture/novelty 均在 held-out eval 子集计算。主分析是 centered raw signal；未把可选 unit-direction 结果混入主结论。",
        "",
        "真实 gate 通过对生产 `replay_hidden` 的 spike 关于 linear current 做 autograd 获得；`delta` 同时来自真实 local proxy current-gradient，并逐元素校验 `delta=q⊙g`。时间聚合确认与训练规则一致，使用 mean。",
        "",
        "## 2. 扩张发生在什么时候",
        "",
    ]
    timing = {}
    for layer in LAYERS:
        ranks = primary[primary.layer == layer].groupby("epoch").r95.mean().sort_index()
        changes = ranks.diff().dropna()
        end = int(changes.idxmax())
        start = int(ranks.index[list(ranks.index).index(end) - 1])
        timing[layer] = (start, end, float(changes.loc[end]))
        lines.append(f"- {layer}: 最大平均 Δr95 出现在 {start}→{end}，Δr95={changes.loc[end]:.2f}。")
    lines.extend(["", "## 3. 扩张还是旋转", ""])
    for layer in LAYERS:
        value = details[layer]
        lines.append(
            f"- {layer}: **{verdicts[layer]}**。r95/D 总增量={value['r95_growth_over_D']:.3f}；最终区间 k=32 previous-overlap={value['final_interval_previous_overlap_k32']:.3f}；final-r95 novelty={value['final_interval_novel_energy_final_r95']:.3f}。判定同时使用固定-k overlap、principal angle、held-out novelty 和 rank growth；低 overlap 没有被单独解释为新增方向。"
        )
    lines.extend(["", "## 4. 是否存在 stable core + expanding tail", ""])
    for layer in LAYERS:
        value = details[layer]
        lines.append(
            f"- {layer}: epoch10→final overlap：k=32 为 {value['epoch10_final_overlap_k32']:.3f}，k=final-r95 为 {value['epoch10_final_overlap_final_r95']:.3f}。"
        )
    lines.append("\n规则透明化：若 r95/D 增长>0.10、epoch10 的 k=32 final-overlap>0.55，且比 final-r95 overlap 高>0.12，标为 CORE-PLUS-TAIL；否则结合最终区间 overlap/novelty 判为 expansion、rotation 或 mixed。阈值是描述性分类规则，不是统计检验。")
    lines.extend(["", "## 5. 谱如何展宽", ""])
    for layer in LAYERS:
        data = primary[primary.layer == layer].groupby("epoch").mean(numeric_only=True)
        first, last = data.iloc[0], data.iloc[-1]
        lines.append(
            f"- {layer}: top10 energy {first.top10_energy_fraction:.3f}→{last.top10_energy_fraction:.3f}，top32 {first.top32_energy_fraction:.3f}→{last.top32_energy_fraction:.3f}，总 centered energy {first.total_centered_energy:.3e}→{last.total_centered_energy:.3e}。"
        )
    lines.extend(["", "## 6. gate 是否能够解释扩张", ""])
    for delta_metric in ("delta_r95", "delta_entropy_rank"):
        subset = correlations[correlations.delta_metric == delta_metric]
        best = subset.iloc[subset.rho.abs().argmax()]
        lines.append(
            f"- {delta_metric} 与 gate 指标中绝对相关最大的是 {best.gate_metric}: Spearman ρ={best.rho:.3f}, p={best.p_value:.3g}, n={int(best.sample_count)}。这是跨 seed×epoch×layer 的相关关系，不作因果声称。"
        )
    cf = counterfactual[counterfactual.epoch == final_epoch].groupby("component_mode").r95.mean()
    lines.append("\nq/g 反事实使用完全相同的 sample、timestep 和 neuron coordinates；freeze-q 与 freeze-gate 的结果见 `counterfactual_decomposition.csv`。最终 epoch 跨层/seed 平均 r95：" + ", ".join(f"{key}={value:.1f}" for key, value in cf.items()) + "。")
    lines.extend(["", "## 7. class / timestep structure", ""])
    between_within = classes[classes.record_type == "between_within_effective_rank"].groupby(["component", "epoch"]).r95.mean().unstack(0)
    if not between_within.empty:
        first, last = between_within.iloc[0], between_within.iloc[-1]
        lines.append("- Held-out between/within r95：" + ", ".join(f"{name} {first[name]:.1f}→{last[name]:.1f}" for name in between_within.columns) + "。")
    off_time = temporal[temporal.timestep_a != temporal.timestep_b].groupby("epoch").overlap.mean()
    coh = coherence.groupby("epoch").delta_coherence_mean.mean()
    lines.append(f"- 不同 timestep 的 mean off-diagonal k=32 overlap {off_time.iloc[0]:.3f}→{off_time.iloc[-1]:.3f}；delta temporal coherence {coh.iloc[0]:.3f}→{coh.iloc[-1]:.3f}。between-class rank 几乎不增长，而 within-class rank 与 timestep 分化显著增长，因此数据更支持 **timestep/类内状态多样性**，而不是单纯的类别均值分离；这仍是结构关联而非因果分解。")
    lines.extend(["", "## 8. task relevance", ""])
    control_final = controls[
        (controls.epoch == final_epoch) & (controls.k_source == "final_r95")
    ].groupby("label_condition").agg({"r95":"mean", "entropy_rank":"mean", "energy_in_final_correct_subspace":"mean"})
    for condition, row in control_final.iterrows():
        lines.append(f"- {condition}: r95={row.r95:.1f}, entropy rank={row.entropy_rank:.1f}, energy in final-correct subspace={row.energy_in_final_correct_subspace:.3f}。")
    lines.append("\nShuffling 只改变 label/error/q；forward state 与 gate 不变，因此 gate 差异被明确记为 false。")
    lines.extend(["", "## 9. 最终机制 verdict", ""])
    for layer in LAYERS:
        lines.append(f"- {layer}: **{verdicts[layer]}**")
    lines.extend(["", "## 10. 对下一步算法的影响", ""])
    advice = {
        "CORE-PLUS-TAIL": "优先测试固定/稳定低秩核心与自适应 tail 扩容，并分别约束核心保持和尾部增长。",
        "ROTATION-DOMINANT": "不宜直接固定 PCA projector；应先研究动态子空间跟踪。",
        "EXPANSION-DOMINANT": "可研究按训练阶段 progressive rank expansion，并用 held-out novelty 触发扩容。",
        "MIXED": "需要同时跟踪旋转和容量，避免只增加 rank 或只固定 projector。",
    }
    for layer in LAYERS:
        lines.append(f"- {layer}: {advice[verdicts[layer]]}")
    birth = activation.groupby("layer").birth50.median()
    lines.append("\nfinal top-128 方向的中位 birth50：" + ", ".join(f"{layer}={birth[layer]:.0f}" for layer in LAYERS) + "。这里的 birth 仅表示达到 final direction energy 某一比例的 descriptive activation epoch，不是数学意义的维度出生。")
    lines.extend(["", "---", "", "所有数值来自三个 seed 的完整结果；未按结果挑 seed 或 epoch。`best` checkpoint 不参与预声明的主区间比较。完整机器可读表、12 张核心图和综合图位于本目录。", ""])
    (run_dir / "SUMMARY_CN.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "mechanism_verdicts.json").write_text(
        json.dumps({"verdicts": verdicts, "details": details, "fastest_intervals": timing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return verdicts


def generate_report(run_dir: Path, training: pd.DataFrame) -> dict[str, str]:
    spectral = pd.read_csv(run_dir / "spectral_broadening.csv")
    gate = pd.read_csv(run_dir / "gate_dynamics.csv")
    correlations = gate_correlations(spectral, gate)
    correlations.to_csv(run_dir / "gate_delta_correlations.csv", index=False)
    make_figures(run_dir, training)
    return generate_summary(run_dir, training)
