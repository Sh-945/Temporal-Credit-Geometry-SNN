"""Aggregate the temporal ANN control and matched SNN comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.feedback_expansion.core import principal_subspace_metrics


EPOCHS = (0, 10, 25, 50, 75, 100)


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
    standard = float(np.std(difference, ddof=1)) if len(difference) > 1 else float("nan")
    mean = float(np.mean(difference))
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
    stage = run_dir / "stageB_ann"
    diagnostic_root = stage / "diagnostics" / "temporal_ann_dfa"
    geometry = collect(diagnostic_root, "geometry.csv")
    for filename in (
        "weight_updates.csv",
        "subspace_comparisons.csv",
        "parity.csv",
        "basis_generalization.csv",
    ):
        collect(diagnostic_root, filename).to_csv(stage / filename, index=False)
    geometry.to_csv(stage / "geometry.csv", index=False)

    checkpoint_root = run_dir / "checkpoints" / "stageB_ann" / "temporal_ann_dfa"
    training_rows = []
    summaries = []
    for seed_dir in sorted(checkpoint_root.glob("seed_*")):
        training_rows.extend(json.loads((seed_dir / "training_metrics.json").read_text()))
        summaries.append(json.loads((seed_dir / "variant_summary.json").read_text()))
    training = pd.DataFrame(training_rows)
    training.to_csv(stage / "training_metrics.csv", index=False)
    tests = []
    for summary in summaries:
        for checkpoint, key in (("best_validation", "best_validation_test"), ("final", "final_test")):
            tests.append(
                {
                    "method": "temporal_ann_dfa",
                    "seed": summary["seed"],
                    "checkpoint": checkpoint,
                    "test_accuracy": summary[key]["accuracy"],
                    "test_loss": summary[key]["loss"],
                    "best_validation_epoch": summary["best_validation_epoch"],
                    "convergence_epoch_99pct_own_best": summary["convergence_epoch_99pct_own_best"],
                }
            )
    tests = pd.DataFrame(tests)
    tests.to_csv(stage / "test_results.csv", index=False)

    primary = geometry[
        (geometry.probe_split == "basis_eval")
        & (geometry.residualization == "raw")
    ]
    expansion_rows = []
    for (seed, layer, mode), group in primary[
        primary.signal_type.isin(["ann_dfa_actual", "ann_dfa_q"])
    ].groupby(["seed", "layer", "temporal_mode"]):
        delta = group[group.signal_type == "ann_dfa_actual"].set_index("epoch")
        q = group[group.signal_type == "ann_dfa_q"].set_index("epoch")
        if 0 not in delta.index or 100 not in delta.index or 100 not in q.index:
            continue
        expansion_rows.append(
            {
                "model": "temporal_ann",
                "seed": seed,
                "layer": layer,
                "temporal_mode": mode,
                "init_r95": int(delta.loc[0].r95),
                "final_r95": int(delta.loc[100].r95),
                "final_q_r95": int(q.loc[100].r95),
                "expansion_ratio_ER": float(delta.loc[100].r95 / max(delta.loc[0].r95, 1)),
                "gate_expansion_factor_GE": float(delta.loc[100].r95 / max(q.loc[100].r95, 1)),
                "final_temporal_coherence": float(delta.loc[100].temporal_coherence),
                "final_mean_temporal_cosine": float(delta.loc[100].mean_temporal_cosine),
            }
        )
    ann_expansion = pd.DataFrame(expansion_rows)
    ann_expansion.to_csv(stage / "ann_expansion_factors.csv", index=False)

    snn = pd.read_csv(run_dir / "stageA_bptt" / "geometry.csv")
    snn_primary = snn[
        (snn.method == "dfa_trained")
        & (snn.signal_type == "dfa_actual")
        & (snn.probe_split == "basis_eval")
        & (snn.residualization == "raw")
    ]
    compare_rows = []
    effects = []
    for seed in sorted(geometry.seed.unique()):
        for layer in ("hidden_1", "hidden_2", "hidden_3"):
            for mode in ("timestep", "aggregated"):
                ann_rows = primary[
                    (primary.seed == seed)
                    & (primary.layer == layer)
                    & (primary.temporal_mode == mode)
                    & (primary.signal_type == "ann_dfa_actual")
                ].set_index("epoch")
                snn_rows = snn_primary[
                    (snn_primary.seed == seed)
                    & (snn_primary.layer == layer)
                    & (snn_primary.temporal_mode == mode)
                ].set_index("epoch")
                ann_gate = primary[
                    (primary.seed == seed)
                    & (primary.layer == layer)
                    & (primary.temporal_mode == mode)
                    & (primary.signal_type == "ann_relu_gate")
                ].set_index("epoch")
                snn_gate = snn[
                    (snn.method == "dfa_trained")
                    & (snn.seed == seed)
                    & (snn.layer == layer)
                    & (snn.temporal_mode == mode)
                    & (snn.signal_type == "gate_actual")
                    & (snn.probe_split == "basis_eval")
                    & (snn.residualization == "raw")
                ].set_index("epoch")
                compare_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "temporal_mode": mode,
                        "snn_init_r95": int(snn_rows.loc[0].r95),
                        "snn_final_r95": int(snn_rows.loc[100].r95),
                        "ann_init_r95": int(ann_rows.loc[0].r95),
                        "ann_final_r95": int(ann_rows.loc[100].r95),
                        "snn_ER": float(snn_rows.loc[100].r95 / max(snn_rows.loc[0].r95, 1)),
                        "ann_ER": float(ann_rows.loc[100].r95 / max(ann_rows.loc[0].r95, 1)),
                        "snn_temporal_coherence": float(snn_rows.loc[100].temporal_coherence),
                        "ann_temporal_coherence": float(ann_rows.loc[100].temporal_coherence),
                        "snn_temporal_cosine": float(snn_rows.loc[100].mean_temporal_cosine),
                        "ann_temporal_cosine": float(ann_rows.loc[100].mean_temporal_cosine),
                        "snn_gate_temporal_coherence": float(snn_gate.loc[100].temporal_coherence),
                        "ann_gate_temporal_coherence": float(ann_gate.loc[100].temporal_coherence),
                        "snn_gate_temporal_cosine": float(snn_gate.loc[100].mean_temporal_cosine),
                        "ann_gate_temporal_cosine": float(ann_gate.loc[100].mean_temporal_cosine),
                    }
                )
    comparison = pd.DataFrame(compare_rows)
    comparison.to_csv(stage / "snn_vs_temporal_ann.csv", index=False)
    for layer in ("hidden_1", "hidden_2", "hidden_3"):
        selected = comparison[(comparison.layer == layer) & (comparison.temporal_mode == "timestep")]
        for first, second, metric in (
            ("snn_final_r95", "ann_final_r95", "final_r95"),
            ("snn_ER", "ann_ER", "ER"),
            ("snn_temporal_coherence", "ann_temporal_coherence", "temporal_coherence"),
            ("snn_temporal_cosine", "ann_temporal_cosine", "temporal_cosine"),
            ("snn_gate_temporal_coherence", "ann_gate_temporal_coherence", "gate_temporal_coherence"),
            ("snn_gate_temporal_cosine", "ann_gate_temporal_cosine", "gate_temporal_cosine"),
        ):
            effects.append(
                {
                    "comparison": "snn_minus_temporal_ann",
                    "layer": layer,
                    "metric": metric,
                    **paired_effect((selected[first] - selected[second]).to_numpy(float)),
                }
            )
    pd.DataFrame(effects).to_csv(stage / "paired_effects.csv", index=False)

    overlap_rows = []
    snn_diag = run_dir / "stageA_bptt" / "diagnostics" / "dfa_trained"
    ann_diag = diagnostic_root
    for seed in sorted(geometry.seed.unique()):
        for epoch in EPOCHS:
            snn_path = snn_diag / f"seed_{seed}" / f"epoch_{epoch:03d}"
            ann_path = ann_diag / f"seed_{seed}" / f"epoch_{epoch:03d}"
            snn_bases = torch.load(snn_path / "bases.pt", weights_only=False)
            ann_bases = torch.load(ann_path / "bases.pt", weights_only=False)
            for layer in ("hidden_1", "hidden_2", "hidden_3"):
                for mode in ("timestep", "aggregated"):
                    snn_key = f"dfa_actual|{layer}|basis_fit|{mode}"
                    ann_key = f"ann_dfa_actual|{layer}|basis_fit|{mode}"
                    snn_r95 = int(
                        snn[
                            (snn.method == "dfa_trained")
                            & (snn.seed == seed)
                            & (snn.epoch == epoch)
                            & (snn.layer == layer)
                            & (snn.signal_type == "dfa_actual")
                            & (snn.probe_split == "basis_fit")
                            & (snn.temporal_mode == mode)
                            & (snn.residualization == "raw")
                        ].iloc[0].r95
                    )
                    ann_r95 = int(
                        geometry[
                            (geometry.seed == seed)
                            & (geometry.epoch == epoch)
                            & (geometry.layer == layer)
                            & (geometry.signal_type == "ann_dfa_actual")
                            & (geometry.probe_split == "basis_fit")
                            & (geometry.temporal_mode == mode)
                            & (geometry.residualization == "raw")
                        ].iloc[0].r95
                    )
                    overlap_rows.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "layer": layer,
                            "temporal_mode": mode,
                            "snn_r95_fit": snn_r95,
                            "ann_r95_fit": ann_r95,
                            **principal_subspace_metrics(
                                snn_bases[snn_key], ann_bases[ann_key], max(1, min(snn_r95, ann_r95))
                            ),
                        }
                    )
    pd.DataFrame(overlap_rows).to_csv(stage / "snn_ann_subspace_overlap.csv", index=False)

    timestep = comparison[comparison.temporal_mode == "timestep"]
    consistent_rank = all(
        (group.snn_final_r95 > group.ann_final_r95).all()
        for _layer, group in timestep.groupby("layer")
    )
    consistent_temporal = all(
        (group.snn_temporal_coherence < group.ann_temporal_coherence).all()
        for _layer, group in timestep.groupby("layer")
    )
    consistent_gate_diversity = all(
        (group.snn_gate_temporal_cosine < group.ann_gate_temporal_cosine).all()
        for _layer, group in timestep.groupby("layer")
    )
    if consistent_rank and consistent_temporal and consistent_gate_diversity:
        specificity = "STRONG SNN-SPECIFICITY"
    elif sum((consistent_rank, consistent_temporal, consistent_gate_diversity)) >= 1:
        specificity = "PARTIAL SNN-SPECIFICITY"
    else:
        specificity = "NOT SNN-SPECIFIC"
    summary = {
        "status": "complete",
        "diagnostic_jobs": len(list(diagnostic_root.glob("**/complete.json"))),
        "best_validation_test_accuracy": tests[tests.checkpoint == "best_validation"].test_accuracy.agg(["mean", "std"]).to_dict(),
        "specificity_verdict": specificity,
        "verdict_rule": "strong requires all-seed higher SNN final r95, lower delta temporal coherence, and lower gate temporal cosine in every layer; partial requires at least one; otherwise not specific",
    }
    (stage / "stageB_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
