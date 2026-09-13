"""Create the Paper 1 Experiment 04 manifest, evidence matrix, and Chinese summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260830, 20260831, 20260901)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values: pd.Series, *, percent: bool = False, digits: int = 3) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{values.mean()*scale:.{digits}f}±{values.std(ddof=1)*scale:.{digits}f}{suffix}"


def environment_text() -> str:
    try:
        nvidia = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except Exception as error:  # pragma: no cover
        nvidia = f"nvidia-smi unavailable: {error}"
    return "\n".join(
        [
            f"utc_time: {datetime.now(timezone.utc).isoformat()}",
            f"os: {platform.platform()}", f"python: {sys.version}",
            f"torch: {torch.__version__}", f"torch_cuda: {torch.version.cuda}",
            f"cudnn: {torch.backends.cudnn.version()}",
            f"cuda_available: {torch.cuda.is_available()}",
            f"seeds: {list(SEEDS)}", "", "[nvidia-smi]", nvidia,
        ]
    )


def evidence_matrix(verdict: dict, width: dict) -> str:
    rows = [
        ("DVS full layer-wise replication", "Experiment 03 preregistered criterion", "FAILED (preserved)", "H1/H2 did not show strong expansion"),
        ("Layer-selective temporal differentiation → lifting", "3 seeds × 6 epochs × 3 layers", "DESCRIPTIVE SUPPORT", "H3 negative cosine–GE association; H1/H2 boundary"),
        ("Gate geometry contributes to H3 lifting", "Final-checkpoint offline MeanGate", "PARTIAL SUPPORT" if verdict["criteria"]["mean_gate_compression"] else "NOT SUPPORTED", "MeanGate compresses but does not eliminate lifting; same forward checkpoint"),
        ("DVS H3 diversity mechanism", "Actual vs H3-only MeanGate, 3 paired seeds", verdict["verdict"], "Geometry compression plus mean learning effect; H1/H2 retained"),
        ("DVS H3 alignment effect on accuracy", "Actual vs H3-only ShuffledGate, 3 paired seeds", "NOT STABLE", "Credit direction changes strongly, but accuracy effect is small and seed-inconsistent"),
        ("N-MNIST width robustness", "D=400/800/1200, 3 seeds", "SUPPORTED" if width["robust"] else "LIMITATION", "Fallback used because no tested alternative surrogate exists"),
        ("Universal expansion across all SNN layers", "H1/H2 negative result", "REJECTED", "Claim is explicitly conditional, not universal"),
    ]
    lines = [
        "# Experiment 04 Evidence Matrix", "",
        "| Claim | Evidence | Status | Boundary / note |", "|---|---|---|---|",
    ]
    lines.extend(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
    lines.extend(
        [
            "", "## Claim hierarchy", "",
            "- **Claim A — Strong:** On N-MNIST, temporal gate diversity and gate-time alignment strongly shape DFA credit geometry.",
            f"- **Claim B — DVS follow-up:** {verdict['verdict']} for the layer-conditional diversity mechanism; alignment-related accuracy replication is not stable. This does not overwrite the failed full cross-dataset replication.",
            "- **Claim C — Explicit boundary:** Credit-space lifting is layer-selective and depends on local temporal-state differentiation.",
            "- Forbidden overclaim: **All SNN layers exhibit credit-space expansion.**",
        ]
    )
    return "\n".join(lines) + "\n"


def summary(run_dir: Path) -> str:
    trajectory = pd.read_csv(run_dir / "dvs_temporal_lifting_trajectory.csv")
    corr = pd.read_csv(run_dir / "dvs_trajectory_correlations.csv")
    counter = pd.read_csv(run_dir / "dvs_counterfactual_gate.csv")
    causal = pd.read_csv(run_dir / "dvs_h3_causal_results.csv")
    width = pd.read_csv(run_dir / "width_robustness.csv")
    verdict = json.loads((run_dir / "stageB_mechanism_verdict.json").read_text(encoding="utf-8"))
    width_verdict = json.loads((run_dir / "stageC_width_verdict.json").read_text(encoding="utf-8"))

    final = trajectory[trajectory.epoch.eq(100)]
    h3_by_epoch = trajectory[trajectory.layer.eq("hidden_3")].groupby("epoch").mean(numeric_only=True)
    h3_corr = corr[
        corr.layer.eq("hidden_3") & corr.seed_scope.eq("pooled")
        & corr.x_metric.eq("gate_temporal_cosine")
    ].iloc[0].spearman_rho
    causal_means = causal.groupby("Method").mean(numeric_only=True)
    counter_means = counter.groupby(["variant", "layer"]).mean(numeric_only=True)
    width_means = width.groupby(["width", "layer"]).mean(numeric_only=True)

    def final_value(layer: str, column: str) -> str:
        return mean_std(final[final.layer.eq(layer)][column])

    actual = causal[causal.Method.eq("Actual")]
    mean_gate = causal[causal.Method.eq("H3-MeanGate")]
    shuffled = causal[causal.Method.eq("H3-ShuffledGate")]
    acc_line = (
        f"Actual {mean_std(actual['Best-val test accuracy'], percent=True)}；"
        f"H3-MeanGate {mean_std(mean_gate['Best-val test accuracy'], percent=True)}；"
        f"H3-ShuffledGate {mean_std(shuffled['Best-val test accuracy'], percent=True)}。"
    )
    h3_line = (
        f"Actual q/delta/GE={causal_means.loc['Actual', 'H3 q r95']:.2f}/"
        f"{causal_means.loc['Actual', 'H3 delta timestep r95']:.2f}/"
        f"{causal_means.loc['Actual', 'H3 GE']:.2f}；Mean="
        f"{causal_means.loc['H3-MeanGate', 'H3 q r95']:.2f}/"
        f"{causal_means.loc['H3-MeanGate', 'H3 delta timestep r95']:.2f}/"
        f"{causal_means.loc['H3-MeanGate', 'H3 GE']:.2f}；Shuffled="
        f"{causal_means.loc['H3-ShuffledGate', 'H3 q r95']:.2f}/"
        f"{causal_means.loc['H3-ShuffledGate', 'H3 delta timestep r95']:.2f}/"
        f"{causal_means.loc['H3-ShuffledGate', 'H3 GE']:.2f}。"
    )
    offline_actual = counter_means.loc[("Actual", "hidden_3")]
    offline_mean = counter_means.loc[("MeanGate", "hidden_3")]
    offline_reduction = 100.0 * (
        1.0 - offline_mean.delta_timestep_r95 / offline_actual.delta_timestep_r95
    )
    mean_accuracy_effect = verdict["mean_gate_accuracy_effect_actual_minus_intervention_pp"]
    shuffle_accuracy_effect = verdict["shuffled_gate_accuracy_effect_actual_minus_intervention_pp"]
    shuffle_delta_cosine = causal_means.loc[
        "H3-ShuffledGate", "H3 actual-vs-intervened delta cosine"
    ]
    stop = verdict["verdict"] != "NOT REPLICATED"

    lines = [
        "# Paper 1 — Experiment 04 中文总结", "",
        f"**最终 DVS follow-up verdict：{verdict['verdict']}。** 原 Experiment 03 的 `full cross-dataset replication failed` 保持不变。",
        "", "## A. Stage A：DVS 时序轨迹", "",
        "| Layer | epoch 0 GE | epoch 100 GE | epoch 100 gate cosine |",
        "|---|---:|---:|---:|",
    ]
    for layer in ("hidden_1", "hidden_2", "hidden_3"):
        initial = trajectory[(trajectory.layer.eq(layer)) & (trajectory.epoch.eq(0))]
        lines.append(
            f"| {layer.replace('hidden_', 'H')} | {initial['ge'].mean():.3f} | "
            f"{final_value(layer, 'ge')} | {final_value(layer, 'gate_temporal_cosine')} |"
        )
    lines.extend(
        [
            "", f"H3 在 epoch 10 已跃迁到 GE={h3_by_epoch.loc[10, 'ge']:.2f}，之后维持在约 27–31；因此它是早期快速形成并稳定保持，而不是缓慢单调增长。H3 gate cosine 同期从 {h3_by_epoch.loc[0, 'gate_temporal_cosine']:.4f} 降到 {h3_by_epoch.loc[10, 'gate_temporal_cosine']:.4f}，epoch 100 为 {h3_by_epoch.loc[100, 'gate_temporal_cosine']:.4f}。描述性 pooled Spearman ρ={h3_corr:.3f}；不报告 p-value。",
            "", "## B. Stage B：H3-only 因果训练", "", acc_line, "", h3_line,
            "", "## C. 必答问题", "",
            f"1. **DVS H3 lifting 是否随训练逐渐形成？** 不是缓慢单调形成；它从 epoch 0 的 {h3_by_epoch.loc[0, 'ge']:.2f} 在 epoch 10 快速跃迁到 {h3_by_epoch.loc[10, 'ge']:.2f}，随后稳定保持。",
            f"2. **H3 lifting 与 temporal gate differentiation 是否同步？** 是，跃迁同期 gate cosine 从 {h3_by_epoch.loc[0, 'gate_temporal_cosine']:.4f} 降到 {h3_by_epoch.loc[10, 'gate_temporal_cosine']:.4f}；跨全部 seed×epoch 的方向为 ρ={h3_corr:.3f}，但这是依赖 epoch 的描述分析。",
            f"3. **为什么 H1/H2 不 lifting？** H1/H2 的 gate cosine 到 epoch 100 仍分别为 {final_value('hidden_1', 'gate_temporal_cosine')} 和 {final_value('hidden_2', 'gate_temporal_cosine')}，局部 gate 几乎不随时间分化；对应 GE 为 {final_value('hidden_1', 'ge')} 和 {final_value('hidden_2', 'ge')}。这符合 span(D_tQ) 在 D_t 近似不变时不扩张的条件机制。",
            f"4. **offline MeanGate 是否能解释 H3 lifting 来源？** 部分可以。Actual H3 delta r95={offline_actual.delta_timestep_r95:.2f}，未做 norm matching 的 offline MeanGate 为 {offline_mean.delta_timestep_r95:.2f}，压缩 {offline_reduction:.1f}%（GE {offline_actual['ge']:.2f}→{offline_mean['ge']:.2f}）；但仍残留明显 lifting，因此 gate differentiation 是重要来源而不是唯一来源。",
            f"5. **H3 MeanGate training 如何改变 geometry 和 accuracy？** {h3_line} Best-val→test accuracy：Actual {causal_means.loc['Actual', 'Best-val test accuracy']*100:.3f}%，Mean {causal_means.loc['H3-MeanGate', 'Best-val test accuracy']*100:.3f}%，平均下降 {mean_accuracy_effect:.3f} pp，3 个 seed 中 {verdict['mean_gate_degradation_seed_count']} 个下降。",
            f"6. **H3 ShuffledGate 是否在维持较高维度时仍破坏学习？** 它保持 H3 delta r95={causal_means.loc['H3-ShuffledGate', 'H3 delta timestep r95']:.2f}，并将 actual-vs-intervened delta cosine 降至 {shuffle_delta_cosine:.3f}；但 accuracy 仅平均下降 {shuffle_accuracy_effect:.3f} pp，且只在 {verdict['shuffled_gate_degradation_seed_count']}/3 seed 下降、方向不一致。因此 alignment 明显改变 credit 方向，却没有稳定复现 accuracy 损害。",
            f"7. **N-MNIST diversity + alignment 是否在 DVS H3 得到 causal replication？** 只能作有边界的回答：总体 verdict 为 **{verdict['verdict']}**；diversity 的几何压缩与学习效应得到支持，但 alignment 的 accuracy replication 不稳定，不能声称双机制完整复现。",
            "8. **原 full cross-dataset replication failed 是否仍成立？** **YES。** H1/H2 negative result 全部保留，新实验不改写原预注册 verdict。",
            f"9. **新的更准确 claim 是什么？** Credit-space lifting 不是每个 SNN 层的无条件属性；其幅度依赖局部 temporal-state differentiation。DVS 的证据集中在深层 H3。",
            f"10. **surrogate/width robustness 是否成立？** 仓库没有第二个成熟测试 surrogate，故按预案执行 D=400/800/1200 width fallback；结论为 **{'成立' if width_verdict['robust'] else '未成立/存在宽度依赖限制'}**。",
            "11. **最危险的 reviewer objection？** H3 是观察到强 expansion 后选择的 targeted follow-up，且只有 3 个配对 seed；因此它支持条件机制，但不能被包装成新的预注册全数据集复现。训练干预同时改变 H3 局部梯度，学习效应仍可能包含层功能敏感性。",
            f"12. **是否建议停止实验、进入写作？** {'建议。当前定向问题已闭环，应停止继续碰数据集并进入写作。' if stop else '仍应停止本轮，按 NOT REPLICATED 如实写 limitation；不建议追加救结果实验。'}",
            "", "## D. Width robustness 摘要", "",
            "| Width | Layer | delta r95 | delta r95 / D | GE | gate cosine | accuracy |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for (width_value, layer), row in width_means.iterrows():
        lines.append(
            f"| {width_value} | {layer.replace('hidden_', 'H')} | {row.delta_r95:.2f} | "
            f"{row.delta_r95_over_width:.3f} | {row['ge']:.2f} | {row.gate_temporal_cosine:.3f} | "
            f"{row.best_val_test_accuracy*100:.3f}% |"
        )
    lines.extend(
        [
            "", "## E. Paper-ready 文件", "",
            "- `dvs_gate_diversity_vs_lifting.png`", "- `dvs_layerwise_trajectory.png`",
            "- `dvs_h3_causal_validation.png`", "- `dvs_temporal_lifting_trajectory.csv`",
            "- `dvs_counterfactual_gate.csv`", "- `dvs_h3_causal_results.csv`",
            "- `width_robustness.csv`", "- `EXPERIMENT04_EVIDENCE_MATRIX.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir, source_run = args.run_dir.resolve(), args.source_run.resolve()
    verdict = json.loads((run_dir / "stageB_mechanism_verdict.json").read_text(encoding="utf-8"))
    width = json.loads((run_dir / "stageC_width_verdict.json").read_text(encoding="utf-8"))
    (run_dir / "EXPERIMENT04_EVIDENCE_MATRIX.md").write_text(
        evidence_matrix(verdict, width), encoding="utf-8"
    )
    (run_dir / "SUMMARY_CN.md").write_text(summary(run_dir), encoding="utf-8")
    (run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")

    sources = [
        source_run / "stageD_cross_dataset" / "geometry.csv",
        source_run / "stageA_bptt" / "geometry.csv",
        source_run / "stageA_bptt" / "test_results.csv",
    ]
    code_files = [
        ROOT / "analysis/paper1_geometry/training.py",
        ROOT / "analysis/paper1_geometry/diagnostics.py",
        ROOT / "methods/gate_intervention/gates.py",
        ROOT / "scripts/diagnostics/paper1_worker.py",
        ROOT / "scripts/diagnostics/paper1_experiment04_stageA.py",
        ROOT / "scripts/diagnostics/paper1_experiment04_stageB_run.py",
        ROOT / "scripts/diagnostics/paper1_experiment04_aggregate.py",
        ROOT / "scripts/diagnostics/paper1_experiment04_stageC_width.py",
        ROOT / "scripts/diagnostics/paper1_experiment04_finalize.py",
    ]
    required_outputs = [
        "SUMMARY_CN.md", "dvs_temporal_lifting_trajectory.csv",
        "dvs_gate_diversity_vs_lifting.png", "dvs_layerwise_trajectory.png",
        "dvs_counterfactual_gate.csv", "dvs_h3_causal_results.csv",
        "dvs_h3_causal_validation.png", "width_robustness.csv",
        "EXPERIMENT04_EVIDENCE_MATRIX.md", "run_manifest.json", "environment.txt",
    ]
    manifest = {
        "experiment": "Paper 1 Experiment 04 — Conditional Generalization of Temporal Credit-Space Lifting",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment03": str(source_run),
        "seeds": list(SEEDS), "diagnostic_epochs": [0, 10, 25, 50, 75, 100],
        "stage_order": ["A_existing_checkpoint_trajectory", "A_offline_counterfactual", "B_H3_targeted_causal", "C_width_fallback"],
        "original_full_cross_dataset_replication_failed": True,
        "stageB_verdict": verdict, "stageC_width_verdict": width,
        "alternative_surrogate_available": False,
        "stageC_fallback": "N-MNIST hidden widths 400 and 1200; width 800 reused",
        "test_used_for_selection": False,
        "training_pairing": "same seed, initialization rule, feedback B rule, split, optimizer, schedule, batch size, and T",
        "resource_event": "Two concurrent DVS diagnostic replays OOMed before training; diagnostic concurrency was reduced to one and completed items were reused. No scientific setting changed.",
        "checkpoint_storage": "run-dir checkpoints symlinked to the system disk to preserve data-disk headroom",
        "source_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sources
        ],
        "code_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in code_files
        ],
        "required_outputs": {name: (run_dir / name).is_file() for name in required_outputs if name != "run_manifest.json"},
        "forbidden_work_not_run": ["LoDFA", "low-rank B", "spectral regulation/filtering", "new optimization", "new dataset", "T sweep", "new neuron", "hardware/FPGA"],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "verdict": verdict["verdict"], "width_robust": width["robust"]}))


if __name__ == "__main__":
    main()
