"""Replay formal Paper 1 Dense-DFA checkpoints and export Figure 1 source data.

This script never trains a model.  It invokes the existing Paper 1 diagnostic
worker for the preregistered checkpoints, then selects the unchanged held-out
``basis_eval``/``raw`` r95 rows.  The output is withheld unless the replayed
final values agree exactly with the frozen Paper 1 geometry table and its
rounded main-results summary.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260830, 20260831, 20260901)
EPOCHS = (0, 10, 25, 50, 75, 100)
LAYERS = ("hidden_1", "hidden_2", "hidden_3")
SIGNALS = (
    ("pre_gate_q", "dfa_q", "timestep", "none"),
    ("post_gate_delta", "dfa_actual", "timestep", "none"),
    ("post_gate_delta", "dfa_actual", "aggregated", "temporal_mean"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/nmnist/paper1_experiment03.yaml")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--materialized-cache")
    return parser.parse_args()


def checkpoint_name(epoch: int) -> str:
    return "init.pt" if epoch == 0 else f"epoch_{epoch:03d}.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fixed_probe(reference_run: Path) -> str:
    probe_path = reference_run / "diagnostic_probe.csv"
    split_path = reference_run / "data_split.csv"
    probe = pd.read_csv(probe_path).sort_values("probe_position").reset_index(drop=True)
    split = pd.read_csv(split_path)
    dataset_length = len(split[split.split.isin(("training", "validation"))])

    split_rng = np.random.default_rng(20260830)
    permutation = split_rng.permutation(dataset_length)
    validation_count = max(1, int(round(dataset_length * 0.10)))
    training = np.sort(permutation[validation_count:])
    expected_indices = np.random.default_rng(20260831).permutation(training)[:1024]
    expected_splits = np.array(["basis_fit"] * 512 + ["basis_eval"] * 512)

    if len(probe) != 1024:
        raise AssertionError(f"fixed probe has {len(probe)} rows, expected 1024")
    if not np.array_equal(probe.dataset_index.to_numpy(int), expected_indices):
        raise AssertionError("frozen diagnostic_probe.csv indices do not match the formal split")
    if not np.array_equal(probe.probe_split.to_numpy(str), expected_splits):
        raise AssertionError("frozen diagnostic_probe.csv fit/eval assignment changed")
    return sha256(probe_path)


def run_replays(args: argparse.Namespace, checkpoint_root: Path, replay_dir: Path) -> None:
    env = os.environ.copy()
    if args.materialized_cache:
        cache = Path(args.materialized_cache).resolve()
        if not cache.is_file():
            raise FileNotFoundError(f"materialized cache not found: {cache}")
        env["PAPER1_MATERIALIZED_CACHE"] = str(cache)

    worker = ROOT / "scripts/diagnostics/paper1_worker.py"
    for seed in SEEDS:
        seed_root = checkpoint_root / f"seed_{seed}"
        for epoch in EPOCHS:
            checkpoint = seed_root / checkpoint_name(epoch)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"formal checkpoint not found: {checkpoint}")
            output_dir = replay_dir / f"seed_{seed}" / f"epoch_{epoch:03d}"
            complete = output_dir / "complete.json"
            if complete.is_file() and (output_dir / "geometry.csv").is_file():
                continue
            command = [
                sys.executable,
                str(worker),
                "diagnose_snn",
                "--config",
                str(Path(args.config).resolve()),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--checkpoint",
                str(checkpoint.resolve()),
                "--output-dir",
                str(output_dir.resolve()),
                "--method",
                "dfa_trained",
                "--epoch",
                str(epoch),
            ]
            subprocess.run(command, cwd=ROOT, env=env, check=True)


def one_metric(
    geometry: pd.DataFrame,
    *,
    signal_type: str,
    temporal_mode: str,
    layer: str,
) -> pd.Series:
    selected = geometry[
        (geometry.signal_type == signal_type)
        & (geometry.temporal_mode == temporal_mode)
        & (geometry.layer == layer)
        & (geometry.probe_split == "basis_eval")
        & (geometry.residualization == "raw")
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"expected exactly one {signal_type}/{temporal_mode}/{layer} primary row; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def collect_replay(replay_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        for epoch in EPOCHS:
            path = replay_dir / f"seed_{seed}" / f"epoch_{epoch:03d}" / "geometry.csv"
            geometry = pd.read_csv(path)
            for layer in LAYERS:
                for stage, signal, mode, aggregation_rule in SIGNALS:
                    row = one_metric(
                        geometry,
                        signal_type=signal,
                        temporal_mode=mode,
                        layer=layer,
                    )
                    records.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "source_layer": layer,
                            "layer": f"H{LAYERS.index(layer) + 1}",
                            "stage": stage,
                            "signal_type": signal,
                            "temporal_mode": mode,
                            "aggregation_rule": aggregation_rule,
                            "ambient_dim": int(row.ambient_dim),
                            "r95": int(row.r95),
                        }
                    )
    return pd.DataFrame(records)


def reference_primary(reference_run: Path) -> pd.DataFrame:
    geometry = pd.read_csv(reference_run / "stageA_bptt/geometry.csv")
    return geometry[
        (geometry.method == "dfa_trained")
        & (geometry.probe_split == "basis_eval")
        & (geometry.residualization == "raw")
        & (geometry.epoch.isin(EPOCHS))
        & (geometry.layer.isin(LAYERS))
    ].copy()


def validate_exact_reference(replay: pd.DataFrame, reference_run: Path) -> None:
    reference = reference_primary(reference_run)
    mismatches: list[str] = []
    for row in replay.itertuples(index=False):
        source = reference[
            (reference.seed == row.seed)
            & (reference.epoch == row.epoch)
            & (reference.layer == row.source_layer)
            & (reference.signal_type == row.signal_type)
            & (reference.temporal_mode == row.temporal_mode)
        ]
        if len(source) != 1:
            mismatches.append(
                f"missing/duplicate reference row seed={row.seed} epoch={row.epoch} "
                f"layer={row.layer} signal={row.signal_type} mode={row.temporal_mode}"
            )
            continue
        expected = int(source.iloc[0].r95)
        if expected != row.r95:
            mismatches.append(
                f"seed={row.seed} epoch={row.epoch} layer={row.layer} "
                f"signal={row.signal_type} mode={row.temporal_mode}: "
                f"replay={row.r95}, reference={expected}"
            )
    if mismatches:
        raise AssertionError("replay disagrees with frozen geometry:\n" + "\n".join(mismatches))


def parse_main_metric(value: str) -> tuple[float, float, int]:
    match = re.fullmatch(
        r"\s*([0-9.]+)±([0-9.]+)\s*/\s*([0-9]+)\s*\([^)]+\)\s*", str(value)
    )
    if match is None:
        raise ValueError(f"cannot parse Paper 1 main metric: {value!r}")
    return float(match.group(1)), float(match.group(2)), int(match.group(3))


def validate_main_results(replay: pd.DataFrame, reference_run: Path) -> None:
    main = pd.read_csv(reference_run / "paper1_main_results.csv")
    main = main[
        (main["Dataset"] == "N-MNIST")
        & (main["Model"] == "FC SNN")
        & (main["Learning rule"] == "Dense DFA")
    ]
    if len(main) != 3:
        raise AssertionError(f"expected three Dense DFA main-result rows; found {len(main)}")

    checks = (
        ("timestep r95/D", "timestep"),
        ("aggregated r95/D", "aggregated"),
    )
    for main_column, mode in checks:
        for layer_index, source_layer in enumerate(LAYERS, start=1):
            values = replay[
                (replay.epoch == 100)
                & (replay.source_layer == source_layer)
                & (replay.signal_type == "dfa_actual")
                & (replay.temporal_mode == mode)
            ].sort_values("seed").r95.to_numpy(float)
            observed_mean = float(values.mean())
            observed_std = float(values.std(ddof=1))
            cell = main[main.Layer == source_layer].iloc[0][main_column]
            expected_mean, expected_std, expected_dim = parse_main_metric(cell)
            if expected_dim != 800:
                raise AssertionError(f"unexpected main-results ambient dimension: {expected_dim}")
            if round(observed_mean, 1) != expected_mean or round(observed_std, 1) != expected_std:
                raise AssertionError(
                    f"final {mode} r95 mismatch for H{layer_index}: replay "
                    f"{observed_mean:.1f}±{observed_std:.1f}, main table "
                    f"{expected_mean:.1f}±{expected_std:.1f}"
                )


def source_table(replay: pd.DataFrame, probe_sha256: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (epoch, source_layer, layer, stage, signal, mode, rule), group in replay.groupby(
        [
            "epoch",
            "source_layer",
            "layer",
            "stage",
            "signal_type",
            "temporal_mode",
            "aggregation_rule",
        ],
        sort=True,
    ):
        group = group.sort_values("seed")
        if tuple(group.seed.astype(int)) != SEEDS:
            raise AssertionError("source-data group does not contain the three formal seeds")
        dimensions = group.ambient_dim.unique()
        if len(dimensions) != 1:
            raise AssertionError("ambient dimension differs across seeds")
        dimension = int(dimensions[0])
        values = group.r95.to_numpy(float)
        ratios = values / dimension
        rows.append(
            {
                "dataset": "N-MNIST",
                "model": "FC SNN",
                "learning_rule": "Dense DFA",
                "epoch": int(epoch),
                "layer": layer,
                "source_layer": source_layer,
                "stage": stage,
                "signal_type": signal,
                "temporal_mode": mode,
                "temporal_aggregation_rule": rule,
                "probe_split": "basis_eval",
                "pca_centering": "centered_raw_signal",
                "residualization": "raw",
                "ambient_dim": dimension,
                "seed_20260830_r95": int(values[0]),
                "seed_20260831_r95": int(values[1]),
                "seed_20260901_r95": int(values[2]),
                "r95_mean": float(values.mean()),
                "r95_std": float(values.std(ddof=1)),
                "seed_20260830_r95_over_D": float(ratios[0]),
                "seed_20260831_r95_over_D": float(ratios[1]),
                "seed_20260901_r95_over_D": float(ratios[2]),
                "r95_over_D_mean": float(ratios.mean()),
                "r95_over_D_std": float(ratios.std(ddof=1)),
                "seed_count": 3,
                "diagnostic_probe_sha256": probe_sha256,
                "protocol_validation": "exact_match",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["epoch", "layer", "stage", "temporal_mode"]
    )


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    reference_run = Path(args.reference_run).resolve()
    replay_dir = Path(args.replay_dir).resolve()
    output = Path(args.output).resolve()

    probe_sha256 = validate_fixed_probe(reference_run)
    run_replays(args, checkpoint_root, replay_dir)
    replay = collect_replay(replay_dir)
    validate_exact_reference(replay, reference_run)
    validate_main_results(replay, reference_run)
    table = source_table(replay, probe_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(
        f"FIG1_SOURCE_DATA_OK rows={len(table)} output={output} "
        f"probe_sha256={probe_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
