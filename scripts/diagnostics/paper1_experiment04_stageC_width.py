"""Run and aggregate the pre-specified N-MNIST hidden-width fallback."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.soft_spectral.training import save_paired_initial_state  # noqa: E402
from training.config import load_config  # noqa: E402


SEEDS = (20260830, 20260831, 20260901)
WIDTHS = (400, 1200)
LAYERS = ("hidden_1", "hidden_2", "hidden_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument(
        "--cache",
        default=Path(tempfile.gettempdir()) / "paper1_nmnist_materialized.pt",
        type=Path,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-workers", type=int, default=2)
    return parser.parse_args()


def config_path(width: int) -> Path:
    return ROOT / "configs" / "nmnist" / f"paper1_experiment04_width{width}.yaml"


def seed_dir(run_dir: Path, width: int, seed: int) -> Path:
    return run_dir / "checkpoints" / "stageC_width" / f"width{width}" / f"seed_{seed}"


def checkpoint(root: Path, epoch: int) -> Path:
    return root / ("init.pt" if epoch == 0 else f"epoch_{epoch:03d}.pt")


def run_logged(command: list[str], *, log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            + log.read_text(encoding="utf-8", errors="replace")[-8000:]
        )


def ensure_cache(path: Path, run_dir: Path) -> None:
    if path.is_file():
        return
    run_logged(
        [
            sys.executable, str(ROOT / "scripts/diagnostics/paper1_cache_data.py"),
            "--config", "configs/nmnist/paper1_experiment03.yaml",
            "--output", str(path), "--num-workers", "10",
        ],
        log=run_dir / "logs" / "cache_nmnist.log", env=os.environ.copy(),
    )


def prepare_initial_states(run_dir: Path) -> None:
    for width in WIDTHS:
        config = load_config(config_path(width))
        snapshot = run_dir / "configs" / config_path(width).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path(width), snapshot)
        for seed in SEEDS:
            path = run_dir / "checkpoints" / "stageC_width" / "initial_states" / f"width{width}_seed_{seed}.pt"
            if path.is_file():
                continue
            selected = copy.deepcopy(config)
            selected["experiment"]["seed"] = seed
            save_paired_initial_state(config=selected, seed=seed, path=path)


def initial_state(run_dir: Path, width: int, seed: int) -> Path:
    return run_dir / "checkpoints" / "stageC_width" / "initial_states" / f"width{width}_seed_{seed}.pt"


def train_job(run_dir: Path, width: int, seed: int, *, device: str, env: dict[str, str]) -> dict:
    destination = seed_dir(run_dir, width, seed)
    if (destination / "variant_summary.json").is_file():
        return {"status": "skipped", "width": width, "seed": seed}
    run_logged(
        [
            sys.executable, str(ROOT / "scripts/diagnostics/paper1_worker.py"),
            "train_snn", "--config", str(config_path(width)), "--seed", str(seed),
            "--method", f"width{width}", "--epochs", "100",
            "--initial-state", str(initial_state(run_dir, width, seed)),
            "--output-dir", str(destination), "--device", device,
        ],
        log=run_dir / "logs" / f"train_width{width}_seed_{seed}.log", env=env,
    )
    return {"status": "complete", "width": width, "seed": seed}


def diagnose_job(run_dir: Path, width: int, seed: int, *, device: str, env: dict[str, str]) -> dict:
    destination = run_dir / "stageC_width" / "diagnostics" / f"width{width}" / f"seed_{seed}"
    if (destination / "complete.json").is_file():
        return {"status": "skipped", "width": width, "seed": seed}
    run_logged(
        [
            sys.executable, str(ROOT / "scripts/diagnostics/paper1_worker.py"),
            "diagnose_snn", "--seed", str(seed), "--method", f"width{width}",
            "--checkpoint", str(checkpoint(seed_dir(run_dir, width, seed), 100)),
            "--output-dir", str(destination), "--epoch", "100",
            "--gate-mode", "actual", "--device", device,
        ],
        log=run_dir / "logs" / f"diag_width{width}_seed_{seed}.log", env=env,
    )
    return {"status": "complete", "width": width, "seed": seed}


def parallel(jobs, worker, max_workers: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, *job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


def metric(frame: pd.DataFrame, signal: str, layer: str) -> pd.Series:
    selected = frame[
        frame.signal_type.eq(signal) & frame.layer.eq(layer)
        & frame.probe_split.eq("basis_eval") & frame.temporal_mode.eq("timestep")
        & frame.residualization.eq("raw")
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one {signal}/{layer} row, got {len(selected)}")
    return selected.iloc[0]


def aggregate(run_dir: Path, source_run: Path) -> pd.DataFrame:
    source_geometry = pd.read_csv(source_run / "stageA_bptt" / "geometry.csv")
    source_geometry = source_geometry[
        source_geometry.method.eq("dfa_trained") & source_geometry.epoch.eq(100)
    ]
    source_test = pd.read_csv(source_run / "stageA_bptt" / "test_results.csv")
    rows: list[dict] = []
    for width in (400, 800, 1200):
        for seed in SEEDS:
            if width == 800:
                geometry = source_geometry[source_geometry.seed.eq(seed)]
                test_row = source_test[
                    source_test.method.eq("dfa_trained")
                    & source_test.seed.eq(seed)
                    & source_test.checkpoint.eq("best_validation")
                ].iloc[0]
                test_accuracy = float(test_row.test_accuracy)
                final_test = float(
                    source_test[
                        source_test.method.eq("dfa_trained")
                        & source_test.seed.eq(seed)
                        & source_test.checkpoint.eq("final")
                    ].iloc[0].test_accuracy
                )
            else:
                geometry = pd.read_csv(
                    run_dir / "stageC_width" / "diagnostics" / f"width{width}" / f"seed_{seed}" / "geometry.csv"
                )
                summary = json.loads(
                    (seed_dir(run_dir, width, seed) / "variant_summary.json").read_text(encoding="utf-8")
                )
                test_accuracy = float(summary["best_validation_test"]["accuracy"])
                final_test = float(summary["final_test"]["accuracy"])
            for layer in LAYERS:
                q = metric(geometry, "dfa_q", layer)
                delta = metric(geometry, "dfa_actual", layer)
                gate = metric(geometry, "gate_actual", layer)
                rows.append(
                    {
                        "width": width, "seed": seed, "layer": layer,
                        "test_accuracy": final_test,
                        "best_val_test_accuracy": test_accuracy,
                        "q_r95": int(q.r95), "delta_r95": int(delta.r95),
                        "delta_r95_over_width": float(delta.r95 / width),
                        "ge": float(delta.r95 / max(float(q.r95), 1.0)),
                        "gate_temporal_cosine": float(gate.mean_temporal_cosine),
                        "credit_temporal_coherence": float(delta.temporal_coherence),
                        "probe_split": "basis_eval", "residualization": "raw",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir, source_run = args.run_dir.resolve(), args.source_run.resolve()
    stage_b = run_dir / "stageB_execution_summary.json"
    if not stage_b.is_file() or json.loads(stage_b.read_text(encoding="utf-8")).get("status") != "complete":
        raise RuntimeError("Stage B must complete before Stage C")
    ensure_cache(args.cache, run_dir)
    prepare_initial_states(run_dir)
    env = os.environ.copy()
    env["PAPER1_MATERIALIZED_CACHE"] = str(args.cache)
    for width in WIDTHS:
        parallel(
            [(run_dir, width, seed) for seed in SEEDS],
            lambda run, selected_width, seed: train_job(
                run, selected_width, seed, device=args.device, env=env
            ),
            3,
        )
    parallel(
        [(run_dir, width, seed) for width in WIDTHS for seed in SEEDS],
        lambda run, selected_width, seed: diagnose_job(
            run, selected_width, seed, device=args.device, env=env
        ),
        max(1, args.diagnostic_workers),
    )
    frame = aggregate(run_dir, source_run)
    frame.to_csv(run_dir / "width_robustness.csv", index=False)
    summary = frame.groupby(["width", "layer"]).agg(
        accuracy_mean=("best_val_test_accuracy", "mean"),
        delta_r95_mean=("delta_r95", "mean"),
        delta_r95_over_width_mean=("delta_r95_over_width", "mean"),
        ge_mean=("ge", "mean"),
        gate_cosine_mean=("gate_temporal_cosine", "mean"),
    ).reset_index()
    robust = bool(
        (summary.groupby("width").ge_mean.max() >= 5.0).all()
        and (summary.groupby("width").delta_r95_mean.max() >= 100).all()
    )
    result = {
        "status": "complete", "robust": robust,
        "interpretation": "qualitative width robustness" if robust else "width-dependent limitation",
        "rows": len(frame), "widths": [400, 800, 1200],
    }
    (run_dir / "stageC_width_verdict.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
