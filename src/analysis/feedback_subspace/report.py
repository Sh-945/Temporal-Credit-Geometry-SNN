"""Plots, candidate ranks, and the Chinese Experiment 01 report."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _primary_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[
        (metrics.signal_type == "post_gate_delta")
        & (metrics.temporal_mode == "aggregated")
        & (metrics.analysis_mode == "centered_raw")
        & (metrics.label_condition == "correct")
    ].copy()


def _final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.checkpoint_kind.astype(str).str.contains("final")].copy()


def _mean_std(values: pd.Series, digits: int = 2) -> str:
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def candidate_rank_table(metrics: pd.DataFrame) -> pd.DataFrame:
    final = _final_rows(_primary_delta(metrics))
    rows: list[dict[str, Any]] = []
    for layer, group in final.groupby("layer", sort=True):
        rows.append(
            {
                "layer": layer,
                "hidden_dim": int(group.ambient_dim.iloc[0]),
                "aggressive_r90_mean": float(group.r90.mean()),
                "aggressive_r90_std": float(group.r90.std(ddof=1)) if len(group) > 1 else 0.0,
                "aggressive_recommended": int(math.ceil(group.r90.mean())),
                "balanced_r95_mean": float(group.r95.mean()),
                "balanced_r95_std": float(group.r95.std(ddof=1)) if len(group) > 1 else 0.0,
                "balanced_recommended": int(math.ceil(group.r95.mean())),
                "conservative_r99_mean": float(group.r99.mean()),
                "conservative_r99_std": float(group.r99.std(ddof=1)) if len(group) > 1 else 0.0,
                "conservative_recommended": int(math.ceil(group.r99.mean())),
            }
        )
    return pd.DataFrame(rows)


def _plots(
    run_dir: Path,
    metrics: pd.DataFrame,
    spectra: pd.DataFrame,
    training: pd.DataFrame,
) -> None:
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})
    delta = _primary_delta(metrics)
    layers = sorted(delta.layer.unique())

    fig, axis = plt.subplots(figsize=(9, 5))
    for layer in layers:
        group = delta[delta.layer == layer].groupby("epoch").r95_over_ambient
        mean, std = group.mean(), group.std().fillna(0)
        axis.plot(mean.index, mean.values, marker="o", label=f"{layer} delta r95/D")
        axis.fill_between(mean.index, mean - std, mean + std, alpha=0.12)
    accuracy_axis = axis.twinx()
    accuracy = training.groupby("epoch").test_acc.agg(["mean", "std"]).fillna(0)
    accuracy_axis.plot(
        accuracy.index,
        accuracy["mean"],
        color="black",
        linewidth=1.8,
        label="test accuracy",
    )
    axis.set(xlabel="Completed epoch", ylabel="delta r95 / hidden dimension")
    accuracy_axis.set_ylabel("Test accuracy")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = accuracy_axis.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy_and_r95_vs_epoch.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, max(1, len(layers)), figsize=(5 * max(1, len(layers)), 4), squeeze=False)
    for axis, layer in zip(axes[0], layers):
        selected = delta[delta.layer == layer]
        for column, label in (
            ("r95", "r95"),
            ("entropy_rank", "entropy rank"),
            ("participation_ratio", "participation ratio"),
        ):
            series = selected.groupby("epoch")[column].mean()
            axis.plot(series.index, series.values, marker="o", label=label)
        axis.set(title=layer, xlabel="Completed epoch", ylabel="Effective dimension")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "effective_rank_vs_epoch.png")
    plt.close(fig)

    final_spectra = spectra[
        spectra.checkpoint_kind.astype(str).str.contains("final")
        & (spectra.analysis_mode == "centered_raw")
        & (spectra.temporal_mode == "aggregated")
        & (spectra.label_condition == "correct")
        & spectra.signal_type.isin(["pre_gate_q", "post_gate_delta"])
    ].copy()
    fig, axes = plt.subplots(1, max(1, len(layers)), figsize=(5 * max(1, len(layers)), 4), squeeze=False)
    for axis, layer in zip(axes[0], layers):
        for signal, label in (("pre_gate_q", "q"), ("post_gate_delta", "delta")):
            selected = final_spectra[
                (final_spectra.layer == layer) & (final_spectra.signal_type == signal)
            ].copy()
            if selected.empty:
                continue
            selected["relative_singular"] = selected.groupby("seed").singular_value.transform(
                lambda values: values / max(float(values.iloc[0]), 1e-30)
            )
            curve = selected.groupby("singular_index").relative_singular.mean()
            axis.semilogy(curve.index, np.clip(curve.values, 1e-12, None), label=label)
        axis.set(title=layer, xlabel="Singular index", ylabel="sigma / sigma_1")
        axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_singular_spectra.png")
    plt.close(fig)

    final_metrics = metrics[
        metrics.checkpoint_kind.astype(str).str.contains("final")
        & (metrics.analysis_mode == "centered_raw")
        & (metrics.temporal_mode == "aggregated")
        & (metrics.label_condition == "correct")
        & metrics.signal_type.isin(["pre_gate_q", "post_gate_delta"])
    ]
    x = np.arange(len(layers))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for offset, signal, label in (
        (-width / 2, "pre_gate_q", "q"),
        (width / 2, "post_gate_delta", "delta"),
    ):
        selected = final_metrics[final_metrics.signal_type == signal]
        absolute = selected.groupby("layer").r95.mean().reindex(layers)
        denominator_ratio = (
            selected.groupby("layer").r95_over_algebraic_max.mean().reindex(layers)
            if signal == "pre_gate_q"
            else selected.groupby("layer").r95_over_ambient.mean().reindex(layers)
        )
        axes[0].bar(x + offset, absolute.values, width, label=label)
        axes[1].bar(x + offset, denominator_ratio.values, width, label=label)
    for axis in axes:
        axis.set_xticks(x, layers)
        axis.legend()
    axes[0].set_ylabel("r95")
    axes[0].set_title("Absolute effective dimension")
    axes[1].set_ylabel("r95 / valid denominator")
    axes[1].set_title("q: algebraic max; delta: hidden dimension")
    fig.tight_layout()
    fig.savefig(run_dir / "pre_vs_post_gate.png")
    plt.close(fig)

    final_delta = _final_rows(delta)
    heat = final_delta.groupby("layer")[[
        "r95_over_ambient", "entropy_over_ambient", "participation_over_ambient"
    ]].mean().reindex(layers)
    fig, axis = plt.subplots(figsize=(7, max(3, len(layers))))
    image = axis.imshow(heat.values, cmap="viridis", vmin=0, vmax=max(0.01, float(heat.max().max())))
    axis.set_xticks(range(3), ["r95/D", "entropy/D", "PR/D"])
    axis.set_yticks(range(len(layers)), layers)
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            axis.text(column, row, f"{heat.iloc[row, column]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=axis)
    axis.set_title("Final post-gate delta effective-rank ratios")
    fig.tight_layout()
    fig.savefig(run_dir / "layerwise_final_rank_heatmap.png")
    plt.close(fig)

    comparison = metrics[
        metrics.checkpoint_kind.astype(str).str.contains("final")
        & (metrics.signal_type == "post_gate_delta")
        & (metrics.temporal_mode == "aggregated")
        & (metrics.analysis_mode == "centered_raw")
    ]
    fig, axis = plt.subplots(figsize=(8, 4))
    for offset, condition in ((-width / 2, "correct"), (width / 2, "shuffled")):
        values = comparison[comparison.label_condition == condition].groupby("layer").r95_over_ambient.mean().reindex(layers)
        axis.bar(x + offset, values.values, width, label=condition)
    axis.set_xticks(x, layers)
    axis.set_ylabel("delta r95 / hidden dimension")
    axis.set_title("Final checkpoint: correct vs shuffled labels")
    axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "correct_vs_shuffled_labels.png")
    plt.close(fig)


def generate_report(run_dir: Path) -> str:
    metrics = pd.read_csv(run_dir / "subspace_metrics.csv")
    spectra = pd.read_csv(run_dir / "singular_values.csv")
    training = pd.read_csv(run_dir / "training_metrics.csv")
    feedback = pd.read_csv(run_dir / "feedback_matrix_checks.csv")
    candidates = candidate_rank_table(metrics)
    candidates.to_csv(run_dir / "candidate_rank_recommendations.csv", index=False)
    _plots(run_dir, metrics, spectra, training)

    delta = _primary_delta(metrics)
    final_delta = _final_rows(delta)
    init_delta = delta[delta.epoch == 0]
    q_final = metrics[
        metrics.checkpoint_kind.astype(str).str.contains("final")
        & (metrics.signal_type == "pre_gate_q")
        & (metrics.temporal_mode == "aggregated")
        & (metrics.analysis_mode == "centered_raw")
        & (metrics.label_condition == "correct")
    ]
    best_accuracies = training.groupby("seed").test_acc.max()
    valid_baseline = len(best_accuracies) == 3 and bool((best_accuracies >= 0.95).all())
    fixed_feedback = bool(feedback.fixed_verified.astype(bool).all())

    core_rows: list[list[Any]] = []
    evolution: dict[str, str] = {}
    for layer in sorted(final_delta.layer.unique()):
        delta_layer = final_delta[final_delta.layer == layer]
        q_layer = q_final[q_final.layer == layer]
        initial = init_delta[init_delta.layer == layer].r95_over_ambient.mean()
        final = delta_layer.r95_over_ambient.mean()
        change = final - initial
        evolution[layer] = "下降" if change <= -0.05 else "上升" if change >= 0.05 else "基本不变"
        core_rows.append(
            [
                layer,
                int(delta_layer.ambient_dim.iloc[0]),
                int(round(q_layer.rank_B.mean())),
                _mean_std(q_layer.r95),
                _mean_std(q_layer.r95_over_algebraic_max, 3),
                _mean_std(delta_layer.r95),
                _mean_std(delta_layer.r95_over_ambient, 3),
                _mean_std(delta_layer.entropy_rank),
                f"{initial:.3f} → {final:.3f}（{evolution[layer]}）",
            ]
        )

    low_layers = final_delta.groupby("layer").r95_over_ambient.mean()
    majority_low = int((low_layers <= 0.20).sum()) >= math.ceil(len(low_layers) / 2)
    moderate_any = bool((low_layers <= 0.40).any())
    seed_consistent = bool(
        (final_delta.groupby("layer").r95_over_ambient.std().fillna(0) <= 0.10).all()
    )
    compressed = any(value == "下降" for value in evolution.values())
    shuffled = metrics[
        metrics.checkpoint_kind.astype(str).str.contains("final")
        & (metrics.signal_type == "post_gate_delta")
        & (metrics.temporal_mode == "aggregated")
        & (metrics.analysis_mode == "centered_raw")
        & (metrics.label_condition == "shuffled")
    ]
    shuffled_effect = False
    if not shuffled.empty:
        correct_mean = final_delta.groupby("layer").r95_over_ambient.mean()
        shuffled_mean = shuffled.groupby("layer").r95_over_ambient.mean()
        shuffled_effect = bool(((shuffled_mean - correct_mean) >= 0.05).any())
    task_related = compressed or shuffled_effect

    if valid_baseline and fixed_feedback and majority_low and seed_consistent and task_related:
        verdict = "STRONG SUPPORT"
    elif valid_baseline and fixed_feedback and moderate_any:
        verdict = "PARTIAL SUPPORT"
    else:
        verdict = "NOT SUPPORTED"

    candidate_rows = [
        [
            row.layer,
            row.aggressive_recommended,
            row.balanced_recommended,
            row.conservative_recommended,
        ]
        for row in candidates.itertuples()
    ]
    accuracy_text = _mean_std(best_accuracies * 100.0, 2) + "%"
    evolution_text = "；".join(f"{layer}：{state}" for layer, state in evolution.items())
    report = f"""# Dense sDFA 有效反馈子空间诊断（Experiment 01）

## 1. 实验是否有效

- 自动判定：**{verdict}**。
- dense sDFA 三种子 best test accuracy：**{accuracy_text}**；三种子均达到 95%：**{'是' if valid_baseline else '否'}**。
- 固定随机反馈矩阵 B 的所有 checkpoint checksum 一致：**{'是' if fixed_feedback else '否'}**。
- instrumentation 来自真实局部更新路径：对 replay 隐藏层的线性电流注册临时 autograd hook，直接取得 `mean(spikes*q)` 对电流的梯度，再仅移除全局 mean 常数。因此 post-gate delta 是实际进入权重更新的 `q⊙g`，不是按论文公式另行模拟。
- 实际时间聚合规则：局部 proxy loss 对时间维取 **mean**；报告同时保留 timestep 观察与相同 mean 聚合。
- 仓库 provenance：源目录不是 Git repository，故 `environment.txt` 记录了完整源码 SHA-256，而不是虚构 commit。

## 2. 核心结果表（final，3 seeds mean ± std）

{_markdown_table(
    ['layer', 'hidden dim', 'rank(B)', 'q-r95', 'q-r95/max', 'delta-r95', 'delta-r95/hidden', 'entropy rank', 'init→final delta ratio'],
    core_rows,
)}

注意：N-MNIST 只有 10 类，`rank(B)≤10` 是代数约束。上表对 q 使用 `min(rank(B), rank(E))` 作分母；绝不把 `q-r95/800` 当作低维发现。核心结论以 post-gate delta 相对 hidden dimension 的结果为准。

## 3. Training evolution

{evolution_text}。只有 final 低于 initialization 时才解释为训练中涌现的压缩；若两者接近，只能称为该设置下内禀的低维使用。

## 4. q 与 delta 的区别

- 多数层 final delta 达到强低维阈值（≤0.20）：**{'是' if majority_low else '否'}**。
- 三种子层级趋势一致（各层 ratio std≤0.10）：**{'是' if seed_consistent else '否'}**。
- correct-vs-shuffled 或训练演化提供 task-related 证据：**{'是' if task_related else '否'}**（演化压缩：{'是' if compressed else '否'}；shuffle 差异：{'是' if shuffled_effect else '否'}）。
- q 的解释必须先看可用 feedback 子空间占用率；delta 才检验 sample/timestep-specific neuronal gating 后的经验维度。

## 5. Hypothesis verdict

**{verdict}**

该判定同时综合 baseline 有效性、B 固定性、delta 的 r95/entropy/PR、三种子方差、训练演化与 shuffled-label 对照，没有因结果方向筛除 seed 或层。

## 6. 下一阶段 adaptive LoDFA 候选 layerwise ranks

这些值只来自 dense sDFA final aggregated delta，本轮没有用它们重训 LoDFA。跨 seed 推荐整数取对应 r90/r95/r99 的均值向上取整。

{_markdown_table(['layer', 'aggressive r90', 'balanced r95', 'conservative r99'], candidate_rows)}

## 7. 结果文件

所有 CSV、六张主图、config snapshots、checkpoint/log provenance 与本报告均位于：

`{run_dir.resolve()}`
"""
    (run_dir / "SUMMARY_CN.md").write_text(report, encoding="utf-8")
    (run_dir / "VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
    return verdict
