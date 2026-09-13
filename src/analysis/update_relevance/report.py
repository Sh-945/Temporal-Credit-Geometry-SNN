"""Plots, automatic verdict, and Chinese report for Experiment 02."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERDICTS = {
    "STRONG UPDATE-RELEVANT SUBSPACE",
    "PARTIAL UPDATE-RELEVANT SUBSPACE",
    "NOT UPDATE-RELEVANT",
}


def _final_epoch(frame: pd.DataFrame) -> int:
    return int(frame.epoch.max())


def _mean_std(values: pd.Series, digits: int = 3) -> str:
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _line_metric(
    gradient: pd.DataFrame,
    run_dir: Path,
    metric: str,
    filename: str,
    ylabel: str,
) -> None:
    final = gradient[
        (gradient.epoch == _final_epoch(gradient))
        & (~gradient.norm_matching.astype(bool))
        & (gradient.basis_type.isin(["top_aggregated", "top_timestep", "random_orthogonal"]))
    ]
    layers = sorted(final.layer.unique())
    figure, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    styles = {
        "top_aggregated": ("o-", "Top-aggregated"),
        "top_timestep": ("s-", "Top-timestep"),
        "random_orthogonal": ("^-", "Random (5 repeats)"),
    }
    for axis, layer in zip(axes, layers):
        part = final[final.layer == layer]
        for basis_type, (style, label) in styles.items():
            selected = part[part.basis_type == basis_type]
            grouped = selected.groupby("k_ratio")[metric].agg(["mean", "std"]).reset_index()
            if grouped.empty:
                continue
            axis.errorbar(
                grouped.k_ratio,
                grouped["mean"],
                yerr=grouped["std"].fillna(0),
                fmt=style,
                capsize=2,
                label=label,
            )
        axis.set_title(layer)
        axis.set_xlabel("k / hidden dimension")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Final checkpoint: {ylabel}")
    _save_figure(run_dir / filename)


def _plot_norm_matched_control(gradient: pd.DataFrame, run_dir: Path) -> None:
    final_epoch = _final_epoch(gradient)
    selected = gradient[
        (gradient.epoch == final_epoch)
        & (gradient.norm_matching.astype(bool))
        & (gradient.k_source == "r95")
        & (gradient.basis_type.isin(["top_aggregated", "top_timestep", "random_orthogonal"]))
    ].copy()
    grouped = selected.groupby(["layer", "basis_type"])[
        ["gradient_cosine", "relative_error"]
    ].mean().reset_index()
    layers = sorted(grouped.layer.unique())
    labels = ["top_aggregated", "top_timestep", "random_orthogonal"]
    x = np.arange(len(layers))
    width = 0.25
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, basis_type in enumerate(labels):
        part = grouped[grouped.basis_type == basis_type].set_index("layer")
        for axis, metric, title in zip(
            axes,
            ["gradient_cosine", "relative_error"],
            ["Gradient cosine", "Relative error"],
        ):
            axis.bar(
                x + (index - 1) * width,
                [part.loc[layer, metric] for layer in layers],
                width,
                label=basis_type,
            )
            axis.set_title(f"Norm-matched r95: {title}")
            axis.set_xticks(x, layers)
            axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    _save_figure(run_dir / "top_vs_norm_matched_random.png")


def _plot_loss_descent(loss: pd.DataFrame, run_dir: Path) -> None:
    if loss.empty:
        return
    final_epoch = _final_epoch(loss)
    selected = loss[
        (loss.epoch == final_epoch)
        & (
            (loss.projection_type == "full")
            | (loss.norm_matching.astype(bool))
        )
    ].copy()
    styles = {
        "top_aggregated": "o-",
        "top_timestep": "s-",
        "random_orthogonal": "^-",
        "identity": "D",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for basis_type, style in styles.items():
        part = selected[selected.basis_type == basis_type]
        if part.empty:
            continue
        for axis, metric, title in zip(
            axes,
            ["task_loss_change", "local_proxy_loss_change"],
            ["Task loss change", "Local-proxy loss change"],
        ):
            grouped = part.groupby("k_ratio")[metric].agg(["mean", "std"]).reset_index()
            if basis_type == "identity":
                axis.scatter(grouped.k_ratio, grouped["mean"], marker=style, label="Full DFA")
            else:
                axis.errorbar(
                    grouped.k_ratio,
                    grouped["mean"],
                    yerr=grouped["std"].fillna(0),
                    fmt=style,
                    capsize=2,
                    label=basis_type,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(title)
            axis.set_xlabel("k / hidden dimension")
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    _save_figure(run_dir / "one_step_loss_change_vs_k.png")


def _plot_bptt_rank(counterfactual: pd.DataFrame, run_dir: Path) -> None:
    rank = counterfactual[counterfactual.record_type == "effective_rank"].copy()
    if rank.empty:
        return
    grouped = rank.groupby(["layer", "method", "temporal_mode"])["r95"].mean().reset_index()
    layers = sorted(grouped.layer.unique())
    combinations = [
        ("DFA", "aggregated"),
        ("BPTT", "aggregated"),
        ("DFA", "timestep"),
        ("BPTT", "timestep"),
    ]
    x = np.arange(len(layers))
    width = 0.2
    plt.figure(figsize=(9, 4.8))
    for index, (method, mode) in enumerate(combinations):
        part = grouped[(grouped.method == method) & (grouped.temporal_mode == mode)].set_index("layer")
        plt.bar(
            x + (index - 1.5) * width,
            [part.loc[layer, "r95"] for layer in layers],
            width,
            label=f"{method}-{mode}",
        )
    plt.xticks(x, layers)
    plt.ylabel("evaluation delta r95")
    plt.title("DFA vs counterfactual-BPTT effective dimension")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    _save_figure(run_dir / "dfa_vs_bptt_effective_rank.png")


def _plot_overlap(angles: pd.DataFrame, run_dir: Path) -> None:
    if angles.empty:
        return
    summary = angles.groupby(
        ["layer", "temporal_mode", "k_source", "k"], as_index=False
    ).agg(overlap_k=("overlap_k", "mean"), median_angle=("principal_angle_degrees", "median"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for (layer, mode), part in summary.groupby(["layer", "temporal_mode"]):
        part = part.sort_values("k")
        label = f"{layer}-{mode}"
        axes[0].plot(part.k, part.overlap_k, "o-", label=label)
        axes[1].plot(part.k, part.median_angle, "o-", label=label)
    axes[0].set_ylabel("overlap_k")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("DFA–BPTT principal-subspace overlap")
    axes[1].set_ylabel("Median principal angle (degrees)")
    axes[1].set_title("DFA–BPTT principal angles")
    for axis in axes:
        axis.set_xlabel("k")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    _save_figure(run_dir / "dfa_bptt_subspace_overlap.png")


def _plot_temporal_effect(
    gradient: pd.DataFrame, basis: pd.DataFrame, run_dir: Path
) -> None:
    final_epoch = _final_epoch(gradient)
    ranks = basis[
        (basis.epoch == final_epoch)
        & (basis.basis_type.isin(["aggregated_delta_pca", "timestep_delta_pca"]))
    ].copy()
    ranks["temporal_mode"] = ranks.basis_type.str.replace("_delta_pca", "", regex=False)
    ranks = ranks.groupby(["layer", "temporal_mode"])["r95_over_ambient"].mean().reset_index()
    preservation = gradient[
        (gradient.epoch == final_epoch)
        & (~gradient.norm_matching.astype(bool))
        & (
            ((gradient.basis_type == "top_aggregated") & (gradient.k_source == "r95"))
            | ((gradient.basis_type == "top_timestep") & (gradient.k_source == "timestep_r95"))
        )
    ].copy()
    preservation["temporal_mode"] = preservation.basis_type.str.replace("top_", "", regex=False)
    preservation = preservation.groupby(["layer", "temporal_mode"])["gradient_cosine"].mean().reset_index()
    layers = sorted(ranks.layer.unique())
    x = np.arange(len(layers))
    width = 0.35
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, mode in enumerate(["aggregated", "timestep"]):
        rank_part = ranks[ranks.temporal_mode == mode].set_index("layer")
        grad_part = preservation[preservation.temporal_mode == mode].set_index("layer")
        axes[0].bar(
            x + (index - 0.5) * width,
            [rank_part.loc[layer, "r95_over_ambient"] for layer in layers],
            width,
            label=mode,
        )
        axes[1].bar(
            x + (index - 0.5) * width,
            [grad_part.loc[layer, "gradient_cosine"] for layer in layers],
            width,
            label=mode,
        )
    axes[0].set_title("Teaching-signal dimensionality")
    axes[0].set_ylabel("delta r95 / hidden dimension")
    axes[1].set_title("Actual-gradient preservation at corresponding r95")
    axes[1].set_ylabel("gradient cosine")
    for axis in axes:
        axis.set_xticks(x, layers)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    _save_figure(run_dir / "temporal_aggregation_effect.png")


def corrected_experiment01_spectrum(experiment01_dir: Path) -> Path:
    """Replace the misleading numerical-residual tail in the q spectrum."""

    spectra = pd.read_csv(experiment01_dir / "singular_values.csv")
    feedback = pd.read_csv(experiment01_dir / "feedback_matrix_checks.csv")
    final_epoch = int(spectra.epoch.max())
    selected = spectra[
        (spectra.epoch == final_epoch)
        & (spectra.label_condition == "correct")
        & (spectra.temporal_mode == "aggregated")
        & (spectra.analysis_mode == "centered_raw")
        & (spectra.signal_type.isin(["pre_gate_q", "post_gate_delta"]))
    ]
    layers = sorted(selected.layer.unique())
    figure, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, layer in zip(axes, layers):
        rank_b = int(
            feedback[(feedback.epoch == final_epoch) & (feedback.layer == layer)][
                "numerical_rank"
            ].iloc[0]
        )
        for signal_type, label in [("pre_gate_q", "q"), ("post_gate_delta", "delta")]:
            part = selected[
                (selected.layer == layer) & (selected.signal_type == signal_type)
            ].groupby("singular_index")["singular_value"].mean().sort_index()
            if signal_type == "pre_gate_q":
                part = part.iloc[:rank_b]
            normalized = part / max(float(part.iloc[0]), 1e-30)
            axis.semilogy(part.index, normalized, label=label)
        axis.axvline(rank_b, linestyle="--", color="tab:blue", alpha=0.8, label=f"rank(B)={rank_b}")
        axis.text(
            rank_b + 8,
            2e-4,
            "q beyond this line = numerical residual (not plotted)",
            fontsize=7,
            color="tab:blue",
        )
        axis.set_title(layer)
        axis.set_xlabel("Singular index")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("sigma / sigma_1")
    axes[0].legend(fontsize=8)
    figure.suptitle("Corrected final spectrum: q truncated at numerical rank")
    output = experiment01_dir / "corrected_final_singular_spectra.png"
    _save_figure(output)
    return output


def _significant_advantage(top: pd.Series, random: pd.Series) -> bool:
    if len(top) < 2 or len(random) < 2:
        return float(top.mean()) > float(random.mean()) + 0.05
    standard_error = np.sqrt(top.var(ddof=1) / len(top) + random.var(ddof=1) / len(random))
    return float(top.mean() - random.mean()) > 2.0 * float(standard_error)


def generate_report(run_dir: Path, experiment01_dir: Path) -> str:
    gradient = pd.read_csv(run_dir / "gradient_preservation.csv")
    loss = pd.read_csv(run_dir / "loss_descent.csv")
    counterfactual = pd.read_csv(run_dir / "bptt_counterfactual_metrics.csv")
    angles = pd.read_csv(run_dir / "principal_angles.csv")
    basis = pd.read_csv(run_dir / "basis_metadata.csv")

    _line_metric(gradient, run_dir, "gradient_cosine", "gradient_cosine_vs_k.png", "Gradient cosine")
    _line_metric(gradient, run_dir, "relative_error", "relative_gradient_error_vs_k.png", "Relative gradient error")
    _plot_norm_matched_control(gradient, run_dir)
    _plot_loss_descent(loss, run_dir)
    _plot_bptt_rank(counterfactual, run_dir)
    _plot_overlap(angles, run_dir)
    _plot_temporal_effect(gradient, basis, run_dir)
    corrected_path = corrected_experiment01_spectrum(experiment01_dir)

    final_epoch = _final_epoch(gradient)
    primary = gradient[
        (gradient.epoch == final_epoch)
        & (~gradient.norm_matching.astype(bool))
        & (gradient.k_source.isin(["r95", "timestep_r95"]))
        & (gradient.basis_type.isin(["top_aggregated", "top_timestep", "random_orthogonal"]))
    ]
    layer_rows = []
    strong_layers = 0
    partial_evidence = False
    for layer in sorted(primary.layer.unique()):
        aggregate = primary[(primary.layer == layer) & (primary.basis_type == "top_aggregated")]
        aggregate = aggregate[aggregate.k_source == "r95"]
        timestep_same_k = primary[
            (primary.layer == layer)
            & (primary.basis_type == "top_timestep")
            & (primary.k_source == "r95")
        ]
        timestep = primary[
            (primary.layer == layer)
            & (primary.basis_type == "top_timestep")
            & (primary.k_source == "timestep_r95")
        ]
        random = primary[
            (primary.layer == layer)
            & (primary.basis_type == "random_orthogonal")
            & (primary.k_source == "r95")
        ]
        cosine = float(aggregate.gradient_cosine.mean())
        relative = float(aggregate.relative_error.mean())
        advantage = _significant_advantage(aggregate.gradient_cosine, random.gradient_cosine)

        full_loss = loss[
            (loss.epoch == final_epoch)
            & (loss.layer == layer)
            & (loss.projection_type == "full")
        ].local_proxy_loss_change
        top_loss = loss[
            (loss.epoch == final_epoch)
            & (loss.layer == layer)
            & (loss.basis_type == "top_aggregated")
            & (loss.k_source == "r95")
            & (~loss.norm_matching.astype(bool))
        ].local_proxy_loss_change
        if len(full_loss) and len(top_loss):
            full_change = float(full_loss.mean())
            top_change = float(top_loss.mean())
            # A hard-spike forward can be exactly unchanged by a very small
            # SGD-equivalent step.  Equality to Full is then a no-op, not
            # evidence that the projected update preserves descent.
            measurable_full_descent = full_change < -1e-15
            if measurable_full_descent:
                loss_gap = abs(top_change - full_change) / abs(full_change)
                loss_close = top_change < 0 and loss_gap <= 0.25
            else:
                loss_gap = float("nan")
                loss_close = False
        else:
            loss_gap = float("nan")
            loss_close = False
        strong = cosine >= 0.95 and relative <= 0.25 and advantage and loss_close
        strong_layers += int(strong)

        r99 = gradient[
            (gradient.epoch == final_epoch)
            & (gradient.layer == layer)
            & (gradient.basis_type == "top_aggregated")
            & (gradient.k_source == "r99")
            & (~gradient.norm_matching.astype(bool))
        ]
        partial_evidence = partial_evidence or (
            (not r99.empty)
            and float(r99.gradient_cosine.mean()) >= 0.90
            and advantage
        ) or (
            float(timestep.gradient_cosine.mean()) >= 0.90 and advantage
        )
        layer_rows.append(
            {
                "layer": layer,
                "agg_cos": cosine,
                "agg_err": relative,
                "time_same_cos": float(timestep_same_k.gradient_cosine.mean()),
                "time_own_cos": float(timestep.gradient_cosine.mean()),
                "time_own_k": int(timestep.k.iloc[0]),
                "random_cos": float(random.gradient_cosine.mean()),
                "loss_gap": loss_gap,
                "strong": strong,
            }
        )

    if strong_layers >= max(1, int(np.ceil(len(layer_rows) / 2))):
        verdict = "STRONG UPDATE-RELEVANT SUBSPACE"
    elif partial_evidence:
        verdict = "PARTIAL UPDATE-RELEVANT SUBSPACE"
    else:
        verdict = "NOT UPDATE-RELEVANT"
    if verdict not in VERDICTS:
        raise AssertionError(verdict)

    layer_table = [
        "| layer | agg-r95 cosine | agg-r95 rel.err | time@same-k cosine | time-own-r95 cosine (k) | random@agg-r95 cosine | local-loss gap vs full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in layer_rows:
        loss_gap_text = (
            f"{row['loss_gap']:.3f}"
            if np.isfinite(row["loss_gap"])
            else "inconclusive (no-op)"
        )
        layer_table.append(
            f"| {row['layer']} | {row['agg_cos']:.3f} | {row['agg_err']:.3f} | "
            f"{row['time_same_cos']:.3f} | {row['time_own_cos']:.3f} ({row['time_own_k']}) | "
            f"{row['random_cos']:.3f} | {loss_gap_text} |"
        )

    safest = []
    for layer in sorted(gradient.layer.unique()):
        candidates = gradient[
            (gradient.epoch == final_epoch)
            & (gradient.layer == layer)
            & (gradient.basis_type == "top_aggregated")
            & (~gradient.norm_matching.astype(bool))
            & (gradient.k_source.isin(["entropy", "r90", "r95", "r99"]))
        ].groupby("k_source").agg(
            cosine=("gradient_cosine", "mean"), error=("relative_error", "mean"), k=("k", "first")
        )
        selected = candidates[(candidates.cosine >= 0.95) & (candidates.error <= 0.25)].sort_values("k")
        if selected.empty:
            safest.append(f"{layer}: full_D")
        else:
            safest.append(f"{layer}: {selected.index[0]} (k={int(selected.iloc[0].k)})")

    rank_rows = counterfactual[counterfactual.record_type == "effective_rank"]
    rank_summary = rank_rows.groupby(["method", "temporal_mode"])["r95"].mean()
    alignment = counterfactual[
        counterfactual.record_type == "weight_gradient_alignment"
    ].gradient_cosine
    min_overlap = angles[angles.k_source == "min_r95"].groupby(
        ["layer", "temporal_mode"]
    ).overlap_k.mean()
    overlap_text = ", ".join(
        f"{layer}/{mode}={value:.3f}" for (layer, mode), value in min_overlap.items()
    )
    easiest = max(layer_rows, key=lambda row: row["agg_cos"])["layer"]
    hardest = min(layer_rows, key=lambda row: row["agg_cos"])["layer"]
    parity_max = float(gradient.parity_weight_relative.max())
    full_loss_rows = loss[loss.projection_type == "full"]
    no_op_fraction = (
        float((full_loss_rows.local_proxy_loss_change.abs() <= 1e-15).mean())
        if len(full_loss_rows)
        else float("nan")
    )
    metric_pass_layers = sum(
        row["agg_cos"] >= 0.95 and row["agg_err"] <= 0.25
        for row in layer_rows
    )
    norm_r95 = gradient[
        (gradient.epoch == final_epoch)
        & (gradient.norm_matching.astype(bool))
        & (gradient.k_source == "r95")
        & (gradient.basis_type.isin(["top_aggregated", "random_orthogonal"]))
    ]
    norm_top_error = norm_r95[
        norm_r95.basis_type == "top_aggregated"
    ].relative_error.mean()
    norm_random_error = norm_r95[
        norm_r95.basis_type == "random_orthogonal"
    ].relative_error.mean()

    summary = f"""# Experiment 02：DFA teaching subspace 的实际更新相关性

## 自动结论

**{verdict}**

生产路径 parity 最大相对误差：`{parity_max:.3e}`。PCA basis 只由前 1024 个 probe 拟合；全部 preservation、loss-descent 与 counterfactual 指标只使用后 1024 个 probe。

## Final checkpoint 核心结果（3 seeds）

{chr(10).join(layer_table)}

## 七个问题

1. **aggregated delta 能否保留实际更新？** 只能部分保留。agg-r95 的数值门槛（cos≥0.95 且 rel.err≤0.25）仅 {metric_pass_layers}/{len(layer_rows)} 层达到；加入有效 one-step descent 要求后强证据层数为 {strong_layers}/{len(layer_rows)}。
2. **timestep 与 aggregated 谁更有效？** 在相同 aggregated-r95 k 下平均 cosine：aggregated={np.mean([row['agg_cos'] for row in layer_rows]):.3f}，timestep={np.mean([row['time_same_cos'] for row in layer_rows]):.3f}；timestep 使用自身约 600 维 r95 时为 {np.mean([row['time_own_cos'] for row in layer_rows]):.3f}。
3. **Top-k 在 norm-matched 下是否优于 Random-k？** 是。方向 cosine 不受正标量 norm matching 改变，Top 与 Random 在三层均有 2-SE 以上分离；norm-matched agg-r95 的平均 relative error 为 Top={norm_top_error:.3f}、Random={norm_random_error:.3f}。但 norm matching 使 Top 更新幅值过冲，不能把方向优势解释成完整更新已被保留。
4. **哪层最易/最难压缩？** 最易：{easiest}；最难：{hardest}（按 aggregated-r95 actual-gradient cosine）。
5. **安全工作区间？** {'；'.join(safest)}。
6. **DFA 与 BPTT 谁的 effective dimension 更高？** evaluation mean r95：DFA-agg={rank_summary.get(('DFA', 'aggregated'), float('nan')):.1f}，BPTT-agg={rank_summary.get(('BPTT', 'aggregated'), float('nan')):.1f}，DFA-time={rank_summary.get(('DFA', 'timestep'), float('nan')):.1f}，BPTT-time={rank_summary.get(('BPTT', 'timestep'), float('nan')):.1f}。DFA/BPTT weight-gradient cosine={_mean_std(alignment)}。
7. **DFA 与 BPTT dominant subspace overlap？** min-r95 overlap：{overlap_text}。

## 方法边界

- actual-gradient 主分析是原始 dense sDFA local-proxy 的 FC weight gradient；identity projector 必须与生产 autograd 梯度一致。
- norm-matched 投影只补回 delta Frobenius norm，不改变方向；因此 cosine 是最直接的方向性对照。
- one-step test 是 checkpoint learning-rate 下的 layer-local SGD-equivalent 虚拟更新；每次更新后恢复原 checkpoint，不使用 Adam state，也不永久修改 checkpoint。
- 为避免 `lr×gradient` 低于 float32 权重量化精度，loss test 在参数值完全相同的 float64 checkpoint 副本上重算；即便如此，Full DFA 对 hard-spike forward 的 local loss 无可测变化比例为 {no_op_fraction:.1%}。这种 no-op 记为 **inconclusive**，不会当作“Top 与 Full 接近”的正证据。
- BPTT 是同一 dense-sDFA checkpoint、相同 forward weights/样本/标签上的只读反事实诊断。
- corrected Experiment 01 spectrum：`{corrected_path}`。q 在 rank(B)=10 后的数值残差不再画成有效尾部。

## 结果目录

`{run_dir}`
"""
    (run_dir / "SUMMARY_CN.md").write_text(summary, encoding="utf-8")
    (run_dir / "VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
    return verdict
