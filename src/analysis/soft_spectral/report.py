"""Figures and pre-registered verdict for Experiment 03."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_METHODS = (
    "full_dense",
    "full_hard",
    "full_soft",
    "full_random",
    "full_bptt",
)
LABELS = {
    "full_dense": "Dense",
    "full_hard": "Hard",
    "full_soft": "Soft",
    "full_random": "Random",
    "full_bptt": "BPTT",
    "full_dense_lrmatch": "Dense-LRMatch",
    "full_soft_normmatched": "Soft-NormMatched",
}
COLORS = {
    "full_dense": "#4c78a8",
    "full_hard": "#e45756",
    "full_soft": "#54a24b",
    "full_random": "#b279a2",
    "full_bptt": "#f2cf5b",
    "full_dense_lrmatch": "#72b7b2",
    "full_soft_normmatched": "#ff9da6",
}
LAYERS = ("hidden_1", "hidden_2", "hidden_3")


def _mean_band(ax, frame: pd.DataFrame, x: str, y: str, method: str) -> None:
    values = frame[frame.method == method].groupby(x)[y].agg(["mean", "std"]).reset_index()
    if values.empty:
        return
    std = values["std"].fillna(0)
    ax.plot(values[x], values["mean"], label=LABELS.get(method, method), color=COLORS.get(method), marker="o", markersize=3)
    ax.fill_between(values[x], values["mean"] - std, values["mean"] + std, color=COLORS.get(method), alpha=0.15)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _weight_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.record_type == "weight_gradient"].copy()


def _method_epoch100_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame.epoch == 100)
        & (frame.temporal_mode == "timestep")
        & frame.method.isin(PRIMARY_METHODS)
        & frame.signal_type.isin(("filtered_dfa_delta", "trained_bptt_credit"))
    ].copy()


def compute_verdict(run_dir: Path, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tests = pd.read_csv(run_dir / "test_results.csv")
    geometry = pd.read_csv(run_dir / "spectral_geometry.csv")
    alignment = _weight_alignment(pd.read_csv(run_dir / "bp_alignment.csv"))
    best_tests = tests[tests.checkpoint == "best_validation"]
    accuracy = best_tests.groupby("method").test_accuracy.mean()
    dense_accuracy = float(accuracy["full_dense"])
    soft_accuracy = float(accuracy["full_soft"])
    hard_accuracy = float(accuracy["full_hard"])
    random_accuracy = float(accuracy["full_random"])
    margin = float(config["experiment03"]["verdict"]["accuracy_margin_pp"]) / 100.0
    accuracy_held = soft_accuracy >= dense_accuracy - margin

    final_align = alignment[alignment.epoch == 100].groupby(["method", "layer"]).gradient_cosine.mean()
    alignment_gain = {}
    random_alignment_gain = {}
    for layer in LAYERS:
        dense = float(final_align.loc[("full_dense", layer)])
        soft = float(final_align.loc[("full_soft", layer)])
        random = float(final_align.loc[("full_random", layer)])
        alignment_gain[layer] = (soft - dense) / max(abs(dense), 1e-12)
        random_alignment_gain[layer] = (soft - random) / max(abs(random), 1e-12)
    alignment_threshold = float(config["experiment03"]["verdict"]["alignment_relative_gain"])
    alignment_pass_layers = sum(value >= alignment_threshold for value in alignment_gain.values())

    final_geo = _method_epoch100_geometry(geometry).groupby(["method", "layer"]).r95.mean()
    rank_reduction = {}
    for layer in LAYERS:
        dense = float(final_geo.loc[("full_dense", layer)])
        soft = float(final_geo.loc[("full_soft", layer)])
        rank_reduction[layer] = (dense - soft) / max(dense, 1e-12)
    rank_threshold = float(config["experiment03"]["verdict"]["rank_relative_reduction"])
    rank_pass_layers = sum(value >= rank_threshold for value in rank_reduction.values())

    paired_accuracy = best_tests.pivot(index="seed", columns="method", values="test_accuracy")
    soft_random_accuracy = paired_accuracy.full_soft - paired_accuracy.full_random
    accuracy_random_advantage = (
        float(soft_random_accuracy.mean()) >= 0.001
        and int((soft_random_accuracy > 0).sum()) >= 2
    )
    paired_alignment = alignment[alignment.epoch == 100].groupby(["seed", "method", "layer"]).gradient_cosine.mean().unstack("method")
    random_layer_wins = 0
    for layer in LAYERS:
        values = paired_alignment.xs(layer, level="layer")
        relative = (values.full_soft - values.full_random) / values.full_random.abs().clip(lower=1e-12)
        if float(relative.mean()) >= 0.10 and int((relative > 0).sum()) >= 2:
            random_layer_wins += 1
    learned_beats_random = accuracy_random_advantage or random_layer_wins >= 2
    hard_not_better = soft_accuracy >= hard_accuracy - margin
    lrmatch_present = "full_dense_lrmatch" in accuracy.index
    soft_beats_lrmatch = True
    lrmatch_details: dict[str, Any] = {"run": lrmatch_present}
    if lrmatch_present:
        control_thresholds = config["experiment03"]["automatic_controls"]
        lrmatch_accuracy_threshold = float(
            control_thresholds["directional_control_accuracy_gain_pp"]
        ) / 100.0
        lrmatch_alignment_threshold = float(
            control_thresholds["directional_control_alignment_relative_gain"]
        )
        lrmatch_accuracy = float(accuracy["full_dense_lrmatch"])
        lrmatch_accuracy_advantage = soft_accuracy - lrmatch_accuracy
        lrmatch_alignment_wins = 0
        for layer in LAYERS:
            lrmatch_value = float(final_align.loc[("full_dense_lrmatch", layer)])
            soft_value = float(final_align.loc[("full_soft", layer)])
            if (soft_value - lrmatch_value) / max(abs(lrmatch_value), 1e-12) >= lrmatch_alignment_threshold:
                lrmatch_alignment_wins += 1
        paired_lr = best_tests.pivot(index="seed", columns="method", values="test_accuracy")
        paired_gap = paired_lr.full_soft - paired_lr.full_dense_lrmatch
        soft_beats_lrmatch = (
            lrmatch_accuracy_advantage >= lrmatch_accuracy_threshold
            and int((paired_gap > 0).sum()) >= 2
        ) or lrmatch_alignment_wins >= 2
        lrmatch_details = {
            "run": True,
            "test_accuracy": lrmatch_accuracy,
            "soft_minus_lrmatch_test_accuracy": lrmatch_accuracy_advantage,
            "soft_accuracy_positive_seeds": int((paired_gap > 0).sum()),
            "soft_alignment_win_layers_at_10pct": lrmatch_alignment_wins,
        }

    strong = (
        accuracy_held
        and alignment_pass_layers >= 2
        and rank_pass_layers >= 2
        and learned_beats_random
        and hard_not_better
        and soft_beats_lrmatch
    )
    geometry_improved = alignment_pass_layers >= 1 or rank_pass_layers >= 2
    if strong:
        verdict = "STRONG CAUSAL SUPPORT"
    elif accuracy_held and geometry_improved:
        verdict = "PARTIAL CAUSAL SUPPORT"
    else:
        verdict = "NOT SUPPORTED"
    details = {
        "accuracy_checkpoint": "best-validation checkpoint; selected without test",
        "dense_test_accuracy": dense_accuracy,
        "soft_test_accuracy": soft_accuracy,
        "hard_test_accuracy": hard_accuracy,
        "random_test_accuracy": random_accuracy,
        "accuracy_held": bool(accuracy_held),
        "alignment_relative_gain": alignment_gain,
        "alignment_pass_layers": alignment_pass_layers,
        "rank_relative_reduction": rank_reduction,
        "rank_pass_layers": rank_pass_layers,
        "learned_beats_random": bool(learned_beats_random),
        "soft_minus_random_test_accuracy_by_seed": soft_random_accuracy.to_dict(),
        "soft_vs_random_alignment_pass_layers": random_layer_wins,
        "hard_not_better": bool(hard_not_better),
        "soft_beats_lrmatch": bool(soft_beats_lrmatch),
        "lrmatch": lrmatch_details,
        "thresholds": config["experiment03"]["verdict"],
    }
    return verdict, details


def make_figures(run_dir: Path) -> None:
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    training = pd.read_csv(run_dir / "training_metrics.csv")
    tests = pd.read_csv(run_dir / "test_results.csv")
    geometry = pd.read_csv(run_dir / "spectral_geometry.csv")
    alignment = _weight_alignment(pd.read_csv(run_dir / "bp_alignment.csv"))
    ranks = pd.read_csv(run_dir / "rank_schedule.csv")
    rotation = pd.read_csv(run_dir / "basis_rotation.csv")

    full = training[training.method.isin(PRIMARY_METHODS)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in PRIMARY_METHODS:
        _mean_band(ax, full, "epoch", "validation_accuracy", method)
    ax.set(xlabel="Epoch", ylabel="Validation accuracy", title="Validation accuracy (selection uses validation only)")
    ax.grid(alpha=0.25); ax.legend(ncol=2)
    _save(fig, figures / "validation_accuracy_curves.png")

    best = tests[(tests.checkpoint == "best_validation") & tests.method.isin(PRIMARY_METHODS)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, method in enumerate(PRIMARY_METHODS):
        values = best[best.method == method].sort_values("seed")
        ax.scatter(np.full(len(values), index), values.test_accuracy, color=COLORS[method], alpha=0.8)
        ax.errorbar(index, values.test_accuracy.mean(), yerr=values.test_accuracy.std(), fmt="D", color="black", capsize=4)
    ax.set_xticks(range(len(PRIMARY_METHODS)), [LABELS[m] for m in PRIMARY_METHODS])
    ax.set(ylabel="Test accuracy", title="Paired test accuracy at best-validation checkpoint")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures / "test_accuracy_comparison.png")

    delta = geometry[(geometry.signal_type.isin(("filtered_dfa_delta", "trained_bptt_credit"))) & (geometry.temporal_mode == "timestep")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for method in ("full_dense", "full_soft", "full_hard", "full_bptt"):
            _mean_band(ax, delta[delta.layer == layer], "epoch", "r95", method)
        ax.set_title(layer); ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Timestep r95"); axes[-1].legend(fontsize=8)
    _save(fig, figures / "timestep_r95_vs_epoch.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for method in ("full_dense", "full_soft", "full_hard", "full_random"):
            _mean_band(ax, alignment[alignment.layer == layer], "epoch", "gradient_cosine", method)
        ax.set_title(layer); ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("DFA vs BP weight-gradient cosine"); axes[-1].legend(fontsize=8)
    _save(fig, figures / "bp_gradient_cosine_vs_epoch.png")

    pilot = training[(training.method.str.startswith("pilot_soft_")) & training.epoch.between(41, 50)]
    pilot_stats = pilot.groupby("alpha").agg(validation_accuracy=("validation_accuracy", "mean"), validation_loss=("validation_loss", "mean")).reset_index()
    pilot_align = alignment[(alignment.method.str.startswith("pilot_soft_")) & (alignment.epoch == 50)].merge(training[["method", "alpha"]].drop_duplicates(), on="method").groupby("alpha").gradient_cosine.mean()
    fig, ax = plt.subplots(figsize=(7, 5)); ax2 = ax.twinx()
    ax.plot(pilot_stats.alpha, pilot_stats.validation_accuracy, marker="o", label="Validation accuracy")
    ax2.plot(pilot_align.index, pilot_align.values, marker="s", color="#e45756", label="BP alignment (not selection)")
    ax.set(xlabel="Alpha", ylabel="Epoch 41-50 mean validation accuracy", title="Pilot alpha tradeoff")
    ax2.set_ylabel("Post-hoc mean BP alignment"); ax.grid(alpha=0.25)
    _save(fig, figures / "alpha_pilot_tradeoff.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for method in ("full_dense", "full_hard", "full_soft"):
            _mean_band(ax, ranks[ranks.layer == layer], "source_epoch", "k", method)
        ax.set_title(layer); ax.set_xlabel("Basis source epoch"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Dynamic calibration r95 k"); axes[-1].legend(fontsize=8)
    _save(fig, figures / "dynamic_k_schedule.png")

    final_alignment = alignment[alignment.epoch == 100].groupby("method").gradient_cosine.mean()
    final_rank = delta[delta.epoch == 100].groupby("method").r95.mean()
    best_acc = best.groupby("method").test_accuracy.mean()
    methods_lr = ["full_soft", "full_random"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, metric, title in zip(axes, (best_acc, final_alignment, final_rank), ("Test accuracy", "BP cosine", "Timestep r95")):
        ax.bar([LABELS[m] for m in methods_lr], [metric[m] for m in methods_lr], color=[COLORS[m] for m in methods_lr]); ax.set_title(title); ax.grid(axis="y", alpha=0.2)
    _save(fig, figures / "learned_vs_random_subspace.png")

    methods_sh = ["full_soft", "full_hard"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, metric, title in zip(axes, (best_acc, final_alignment, final_rank), ("Test accuracy", "BP cosine", "Timestep r95")):
        ax.bar([LABELS[m] for m in methods_sh], [metric[m] for m in methods_sh], color=[COLORS[m] for m in methods_sh]); ax.set_title(title); ax.grid(axis="y", alpha=0.2)
    _save(fig, figures / "soft_vs_hard.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in PRIMARY_METHODS:
        ax.scatter(final_rank[method], best_acc[method], color=COLORS[method], s=80); ax.annotate(LABELS[method], (final_rank[method], best_acc[method]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Mean timestep r95", ylabel="Test accuracy", title="Accuracy vs effective dimension"); ax.grid(alpha=0.25)
    _save(fig, figures / "accuracy_vs_effective_dimension.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in PRIMARY_METHODS:
        ax.scatter(final_alignment[method], best_acc[method], color=COLORS[method], s=80); ax.annotate(LABELS[method], (final_alignment[method], best_acc[method]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Mean BP gradient cosine", ylabel="Test accuracy", title="Accuracy vs BP alignment"); ax.grid(alpha=0.25)
    _save(fig, figures / "accuracy_vs_bp_alignment.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    final_geo = delta[delta.epoch == 100].groupby(["method", "layer"]).r95.mean()
    for ax, layer in zip(axes, LAYERS):
        vals = [final_geo.loc[("full_dense", layer)], final_geo.loc[("full_bptt", layer)]]
        ax.bar(["Dense DFA", "Trained BPTT"], vals, color=[COLORS["full_dense"], COLORS["full_bptt"]]); ax.set_title(layer); ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Timestep credit r95")
    _save(fig, figures / "dfa_vs_trained_bptt_geometry.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for method in ("full_dense", "full_soft"):
            data = rotation[(rotation.layer == layer) & (rotation.k_source == "fixed_32")]
            _mean_band(ax, data, "current_source_epoch", "overlap", method)
        ax.set_title(layer); ax.set_xlabel("Current basis source epoch"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Consecutive overlap k=32"); axes[-1].legend(fontsize=8)
    _save(fig, figures / "basis_rotation_dense_vs_soft.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ambient = 800.0
    annotation_offsets = {
        "full_dense": (6, -12),
        "full_hard": (6, 6),
        "full_soft": (6, 6),
        "full_random": (6, 6),
        "full_bptt": (6, 6),
    }
    for method in PRIMARY_METHODS:
        x = final_rank[method] / ambient
        y = best_acc[method]
        color_value = final_alignment[method]
        point = ax.scatter(x, y, c=[color_value], cmap="viridis", vmin=0, vmax=1, s=130, edgecolor="black")
        ax.annotate(
            LABELS[method],
            (x, y),
            xytext=annotation_offsets[method],
            textcoords="offset points",
        )
    fig.colorbar(point, ax=ax, label="Mean BP gradient cosine")
    ax.set(xlabel="Mean timestep effective-rank ratio", ylabel="Test accuracy", title="Performance–geometry Pareto (all methods retained)"); ax.grid(alpha=0.25)
    _save(fig, figures / "performance_geometry_pareto.png")


def generate_summary(run_dir: Path, config: dict[str, Any], verdict: str, details: dict[str, Any]) -> None:
    selection = json.loads((run_dir / "pilot_selection.json").read_text(encoding="utf-8"))
    tests = pd.read_csv(run_dir / "test_results.csv")
    geometry = pd.read_csv(run_dir / "spectral_geometry.csv")
    alignment = _weight_alignment(pd.read_csv(run_dir / "bp_alignment.csv"))
    gate = pd.read_csv(run_dir / "gate_dynamics.csv")
    rotation = pd.read_csv(run_dir / "basis_rotation.csv")
    decisions_path = run_dir / "automatic_control_decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.is_file() else {"controls_run": []}
    best = tests[tests.checkpoint == "best_validation"]
    acc = best.groupby("method").test_accuracy.agg(["mean", "std"])
    geo = _method_epoch100_geometry(geometry).groupby(["method", "layer"]).r95.mean()
    align = alignment[alignment.epoch == 100].groupby(["method", "layer"]).gradient_cosine.mean()
    bptt_ranks = [geo.loc[("full_bptt", layer)] for layer in LAYERS]
    lines = [
        "# Experiment 03 Summary",
        "",
        "## 1. Pilot alpha selection",
        "",
        f"best alpha = **{selection['selected_alpha']:.2f}**。选择标准严格为 epochs 41–50 mean validation accuracy；差距 <0.05 percentage point 时依次使用 mean validation loss 和更大的 alpha。test 与 BP alignment 均未参与选择。",
        "",
        "## 2. Accuracy",
        "",
    ]
    for method in PRIMARY_METHODS:
        lines.append(f"- {LABELS[method]}: best-validation checkpoint test accuracy = {acc.loc[method, 'mean']*100:.3f} ± {acc.loc[method, 'std']*100:.3f}%")
    lines.extend(["", "## 3. Effective dimensionality", ""])
    for layer in LAYERS:
        lines.append(f"- {layer}: Dense {geo.loc[('full_dense', layer)]:.1f} → Soft {geo.loc[('full_soft', layer)]:.1f} timestep r95；relative reduction={details['rank_relative_reduction'][layer]*100:.1f}%。")
    lines.extend(["", "## 4. DFA–BP alignment", ""])
    for layer in LAYERS:
        lines.append(f"- {layer}: Dense {align.loc[('full_dense', layer)]:.4f} → Soft {align.loc[('full_soft', layer)]:.4f}；relative gain={details['alignment_relative_gain'][layer]*100:.1f}%。")
    lrmatch_details = details["lrmatch"]
    if lrmatch_details.get("run", False):
        lrmatch_line = (
            "Soft beats LRMatch directional-control rule: "
            f"**{details['soft_beats_lrmatch']}**；"
            f"details={json.dumps(lrmatch_details, ensure_ascii=False)}。"
        )
    else:
        lrmatch_line = "LRMatch 未触发、未运行，因此不做 Soft vs LRMatch 比较。"
    lines.extend(["", "## 5. Learned vs Random", "", f"Learned basis beats Random under the pre-registered stability rule: **{details['learned_beats_random']}**。Soft/Random test means are {details['soft_test_accuracy']*100:.3f}% / {details['random_test_accuracy']*100:.3f}%。", "", "## 6. Soft vs Hard", "", f"Hard does not significantly outperform Soft: **{details['hard_not_better']}**。Hard/Soft test means are {details['hard_test_accuracy']*100:.3f}% / {details['soft_test_accuracy']*100:.3f}%。", "", "## 7. Gradient norm / effective-LR controls", "", f"Triggered controls: {', '.join(decisions.get('controls_run', [])) or 'none'}。Soft median weight-gradient norm ratio s_global={decisions.get('s_global_median_weight_gradient_norm_ratio', float('nan')):.4f}。所有条件阈值在 Stage B 前写入 config。", lrmatch_line])
    if "full_soft_normmatched" in acc.index:
        lines.append(f"Soft-NormMatched best-validation test accuracy={acc.loc['full_soft_normmatched', 'mean']*100:.3f} ± {acc.loc['full_soft_normmatched', 'std']*100:.3f}%。")
    lines.extend(["", "## 8. Independently trained BPTT geometry", "", "Matched BPTT-trained timestep r95: " + " / ".join(f"{value:.1f}" for value in bptt_ranks) + "。", "", "## 9. Gate diversity and rotation", ""])
    for layer in LAYERS:
        dense_gate = gate[(gate.method == "full_dense") & (gate.layer == layer) & (gate.epoch == 100)].temporal_gate_cosine.mean()
        soft_gate = gate[(gate.method == "full_soft") & (gate.layer == layer) & (gate.epoch == 100)].temporal_gate_cosine.mean()
        dense_rot = rotation[(rotation.method == "full_dense") & (rotation.layer == layer) & (rotation.k_source == "fixed_32")].overlap.mean()
        soft_rot = rotation[(rotation.method == "full_soft") & (rotation.layer == layer) & (rotation.k_source == "fixed_32")].overlap.mean()
        lines.append(f"- {layer}: temporal gate cosine Dense/Soft={dense_gate:.3f}/{soft_gate:.3f}；mean consecutive k32 overlap Dense/Soft={dense_rot:.3f}/{soft_rot:.3f}。")
    lines.extend(["", "## 10. Final verdict and next decision", "", f"# {verdict}", ""])
    if verdict == "STRONG CAUSAL SUPPORT":
        lines.append("建议下一步只做 N-Caltech101 validation，并在之后比较 continuous spectral weighting 或 online incremental tracker；本实验不自动实施。")
    elif verdict == "PARTIAL CAUSAL SUPPORT":
        lines.append("主要瓶颈按数据在 alpha selection、one-epoch basis lag、two-band weighting、rotation、layer heterogeneity、N-MNIST ceiling 中定位；下一步最多建议两个针对性实验，不自动运行。")
    else:
        lines.append("当前 spectral-regulation hypothesis 不受支持；停止该路线，不继续调 alpha、换 seed 或增加 regularizer。")
    lines.extend(["", "---", "", "模型选择严格只依据 validation；test 仅在 final 与 best-validation checkpoint 评估。没有使用 future basis、BP projector、test tuning 或 seed filtering。", ""])
    (run_dir / "SUMMARY_CN.md").write_text("\n".join(lines), encoding="utf-8")


def generate_report(run_dir: Path, config: dict[str, Any]) -> str:
    verdict, details = compute_verdict(run_dir, config)
    (run_dir / "verdict_details.json").write_text(
        json.dumps({"verdict": verdict, "details": details}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(run_dir)
    generate_summary(run_dir, config, verdict, details)
    return verdict
