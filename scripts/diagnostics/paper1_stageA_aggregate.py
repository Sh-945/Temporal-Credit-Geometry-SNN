"""Aggregate Stage A without importing excluded Experiment 03 variants."""

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


METHOD_MAP = {"full_dense": "dfa_trained", "full_bptt": "bptt_trained"}
SIGNAL_MAP = {"dfa_trained": "dfa_actual", "bptt_trained": "bptt_actual"}
EPOCHS = (0, 10, 25, 50, 75, 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--legacy-run", required=True)
    return parser.parse_args()


def collect(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in root.glob(f"**/{filename}"):
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_effect(first: pd.Series, second: pd.Series) -> dict[str, float | int]:
    difference = first.to_numpy(float) - second.to_numpy(float)
    std = float(np.std(difference, ddof=1)) if len(difference) > 1 else float("nan")
    return {
        "seed_count": len(difference),
        "paired_difference_mean": float(np.mean(difference)),
        "paired_difference_std": std,
        "paired_effect_dz": float(np.mean(difference) / std) if std > 0 else float("nan"),
        "seed_consistency": float(np.mean(np.sign(difference) == np.sign(np.mean(difference)))),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    legacy = Path(args.legacy_run).resolve()
    stage = run_dir / "stageA_bptt"
    diagnostics = stage / "diagnostics"
    geometry = collect(diagnostics, "geometry.csv")
    updates = collect(diagnostics, "weight_updates.csv")
    subspaces = collect(diagnostics, "subspace_comparisons.csv")
    parity = collect(diagnostics, "parity.csv")
    heldout = collect(diagnostics, "basis_generalization.csv")
    for name, frame in (
        ("geometry.csv", geometry),
        ("weight_updates.csv", updates),
        ("within_model_subspaces.csv", subspaces),
        ("parity.csv", parity),
        ("basis_generalization.csv", heldout),
    ):
        frame.to_csv(stage / name, index=False)

    training = pd.read_csv(legacy / "training_metrics.csv")
    training = training[training.method.isin(METHOD_MAP)].copy()
    training["method"] = training.method.map(METHOD_MAP)
    training.to_csv(stage / "training_metrics.csv", index=False)
    tests = pd.read_csv(legacy / "test_results.csv")
    tests = tests[tests.method.isin(METHOD_MAP)].copy()
    tests["method"] = tests.method.map(METHOD_MAP)
    tests.to_csv(stage / "test_results.csv", index=False)

    lookup = training.set_index(["method", "seed", "epoch"])["validation_accuracy"]
    geometry["validation_accuracy"] = [
        lookup.get((row.method, row.seed, row.epoch), np.nan)
        for row in geometry.itertuples()
    ]
    geometry.to_csv(stage / "geometry.csv", index=False)

    delta_rows = []
    primary = geometry[
        (geometry.probe_split == "basis_eval")
        & (geometry.temporal_mode == "timestep")
        & (geometry.residualization == "raw")
    ]
    for (method, seed, layer, signal), group in primary.groupby(
        ["method", "seed", "layer", "signal_type"]
    ):
        values = group.set_index("epoch").r95
        for start, end in zip(EPOCHS[:-1], EPOCHS[1:]):
            if start in values and end in values:
                delta_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "layer": layer,
                        "signal_type": signal,
                        "interval": f"{start}->{end}",
                        "delta_r95": int(values[end] - values[start]),
                    }
                )
    pd.DataFrame(delta_rows).to_csv(stage / "delta_r95_intervals.csv", index=False)

    matched_rows = []
    for seed in sorted(training.seed.unique()):
        for threshold in (.95, .97):
            for method in ("dfa_trained", "bptt_trained"):
                candidates = training[
                    (training.seed == seed)
                    & (training.method == method)
                    & (training.epoch.isin(EPOCHS[1:]))
                    & (training.validation_accuracy >= threshold)
                ].sort_values("epoch")
                if candidates.empty:
                    continue
                epoch = int(candidates.iloc[0].epoch)
                selected = geometry[
                    (geometry.method == method)
                    & (geometry.seed == seed)
                    & (geometry.epoch == epoch)
                    & (geometry.signal_type == SIGNAL_MAP[method])
                    & (geometry.probe_split == "basis_eval")
                    & (geometry.residualization == "raw")
                ]
                for row in selected.itertuples():
                    matched_rows.append(
                        {
                            "threshold": threshold,
                            "selection_rule": "first diagnostic checkpoint reaching validation threshold",
                            "method": method,
                            "seed": seed,
                            "epoch": epoch,
                            "validation_accuracy": float(candidates.iloc[0].validation_accuracy),
                            "layer": row.layer,
                            "temporal_mode": row.temporal_mode,
                            "r95": row.r95,
                            "entropy_rank": row.entropy_rank,
                            "temporal_coherence": row.temporal_coherence,
                        }
                    )
    pd.DataFrame(matched_rows).to_csv(stage / "accuracy_matched_geometry.csv", index=False)

    independent_rows = []
    for seed in sorted(geometry.seed.unique()):
        for epoch in EPOCHS:
            for layer in ("hidden_1", "hidden_2", "hidden_3"):
                paths = {
                    method: diagnostics / method / f"seed_{seed}" / f"epoch_{epoch:03d}"
                    for method in ("dfa_trained", "bptt_trained")
                }
                if not all((path / "bases.pt").is_file() for path in paths.values()):
                    continue
                dfa_bases = torch.load(paths["dfa_trained"] / "bases.pt", weights_only=False)
                bp_bases = torch.load(paths["bptt_trained"] / "bases.pt", weights_only=False)
                for mode in ("timestep", "aggregated"):
                    first_key = f"dfa_actual|{layer}|basis_fit|{mode}"
                    second_key = f"bptt_actual|{layer}|basis_fit|{mode}"
                    dfa_row = geometry[
                        (geometry.method == "dfa_trained")
                        & (geometry.seed == seed)
                        & (geometry.epoch == epoch)
                        & (geometry.layer == layer)
                        & (geometry.signal_type == "dfa_actual")
                        & (geometry.probe_split == "basis_fit")
                        & (geometry.temporal_mode == mode)
                        & (geometry.residualization == "raw")
                    ].iloc[0]
                    bp_row = geometry[
                        (geometry.method == "bptt_trained")
                        & (geometry.seed == seed)
                        & (geometry.epoch == epoch)
                        & (geometry.layer == layer)
                        & (geometry.signal_type == "bptt_actual")
                        & (geometry.probe_split == "basis_fit")
                        & (geometry.temporal_mode == mode)
                        & (geometry.residualization == "raw")
                    ].iloc[0]
                    rank = max(1, min(int(dfa_row.r95), int(bp_row.r95)))
                    independent_rows.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "layer": layer,
                            "temporal_mode": mode,
                            "dfa_r95_fit": int(dfa_row.r95),
                            "bptt_r95_fit": int(bp_row.r95),
                            **principal_subspace_metrics(
                                dfa_bases[first_key], bp_bases[second_key], rank
                            ),
                        }
                    )
    independent = pd.DataFrame(independent_rows)
    independent.to_csv(stage / "independently_trained_subspace_comparison.csv", index=False)

    effects = []
    final = primary[primary.epoch == 100]
    for layer in sorted(final.layer.unique()):
        for metric in ("r95", "entropy_rank", "temporal_coherence"):
            pivot = final[
                (final.layer == layer)
                & (
                ((final.method == "dfa_trained") & (final.signal_type == "dfa_actual"))
                | ((final.method == "bptt_trained") & (final.signal_type == "bptt_actual"))
                )
            ].pivot(index="seed", columns="method", values=metric)
            effects.append(
                {
                    "comparison": "dfa_trained_minus_bptt_trained",
                    "layer": layer,
                    "metric": metric,
                    **paired_effect(
                        pivot.loc[:, "dfa_trained"], pivot.loc[:, "bptt_trained"]
                    ),
                }
            )
    pd.DataFrame(effects).to_csv(stage / "paired_effects.csv", index=False)
    summary = {
        "status": "complete",
        "diagnostic_jobs": int(len(list(diagnostics.glob("**/complete.json")))),
        "geometry_rows": len(geometry),
        "parity_max_relative_error": float(parity.production_delta_relative_error.max()),
        "best_validation_test_accuracy": tests[tests.checkpoint == "best_validation"].groupby("method").test_accuracy.agg(["mean", "std"]).to_dict("index"),
    }
    (stage / "stageA_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
