"""Finish Experiment 03 after independently scheduled base workers.

The scientific work remains in ``soft_spectral_training.py``.  This helper only
waits for the preregistered base cache files, evaluates the preregistered
automatic-control rules, runs any required controls with at most three workers,
and finally invokes the canonical serial orchestrator to assemble the report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.soft_spectral_training import _conditional_decisions  # noqa: E402
from training.config import load_config  # noqa: E402


BASE_METHODS = (
    "full_dense",
    "full_hard",
    "full_soft",
    "full_random",
    "full_bptt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/nmnist/experiment03_soft_spectral.yaml"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout-hours", type=float, default=8.0)
    return parser.parse_args()


def absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def cache_path(run_dir: Path, stage: str, method: str, seed: int) -> Path:
    return run_dir / "partial" / f"{stage}_{method}_seed{seed}.pt"


def wait_for_base(run_dir: Path, seeds: list[int], poll: float, timeout: float) -> None:
    expected = [
        cache_path(run_dir, "full", method, seed)
        for method in BASE_METHODS
        for seed in seeds
    ]
    deadline = time.monotonic() + timeout
    previous_missing = None
    previous_sizes: tuple[int, ...] | None = None
    while True:
        missing = [path for path in expected if not path.is_file()]
        if not missing:
            sizes = tuple(path.stat().st_size for path in expected)
            if all(size > 0 for size in sizes) and sizes == previous_sizes:
                print(
                    json.dumps({"status": "base_complete", "files": len(expected)}),
                    flush=True,
                )
                return
            if previous_sizes is None:
                print(
                    json.dumps(
                        {
                            "status": "waiting_for_stable_base_files",
                            "files": len(expected),
                        }
                    ),
                    flush=True,
                )
            previous_sizes = sizes
        else:
            previous_sizes = None
        if len(missing) != previous_missing:
            print(
                json.dumps(
                    {
                        "status": "waiting_for_base",
                        "complete": len(expected) - len(missing),
                        "expected": len(expected),
                        "missing": [path.name for path in missing],
                    }
                ),
                flush=True,
            )
            previous_missing = len(missing)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {len(missing)} base cache files")
        time.sleep(poll)


def base_decisions(
    run_dir: Path, seeds: list[int], config: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    summaries = []
    alignment = []
    norms = []
    for method in BASE_METHODS:
        for seed in seeds:
            result = torch.load(
                cache_path(run_dir, "full", method, seed),
                map_location="cpu",
                weights_only=False,
            )
            summaries.append(result["summary"])
            alignment.extend(result.get("bp_alignment", []))
            norms.extend(result.get("gradient_norms", []))
    decisions = _conditional_decisions(summaries, alignment, config)
    ratios = [
        float(row["weight_gradient_norm_ratio"])
        for row in norms
        if row.get("method") == "full_soft"
        and row.get("record_type") == "training_batch"
    ]
    if not ratios:
        raise RuntimeError("full_soft training-batch norm ratios are missing")
    scale = float(np.median(ratios))
    decisions["s_global_median_weight_gradient_norm_ratio"] = scale
    return decisions, scale


def run_control_wave(
    *,
    method: str,
    seeds: list[int],
    selected_alpha: float,
    learning_rate_scale: float,
    config_path: Path,
    run_dir: Path,
    device: str,
) -> None:
    pending = [
        seed
        for seed in seeds
        if not cache_path(run_dir, "control", method, seed).is_file()
    ]
    if not pending:
        print(json.dumps({"status": "control_cached", "method": method}), flush=True)
        return
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }
    )
    processes: list[tuple[int, subprocess.Popen, Any]] = []
    for seed in pending:
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "diagnostics" / "soft_spectral_worker.py"),
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--device",
            device,
            "--stage",
            "control",
            "--variant",
            method,
            "--seed",
            str(seed),
            "--selected-alpha",
            str(selected_alpha),
        ]
        if method == "full_dense_lrmatch":
            command.extend(["--learning-rate-scale", str(learning_rate_scale)])
        log_path = run_dir.parent / f"{run_dir.name}_control_{method}_{seed}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((seed, process, log_handle))
    print(
        json.dumps({"status": "control_started", "method": method, "seeds": pending}),
        flush=True,
    )
    failures = []
    for seed, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0 or not cache_path(run_dir, "control", method, seed).is_file():
            failures.append({"seed": seed, "return_code": return_code})
    if failures:
        raise RuntimeError(f"control wave {method} failed: {failures}")
    print(json.dumps({"status": "control_complete", "method": method}), flush=True)


def main() -> None:
    args = parse_args()
    config_path = absolute(args.config)
    run_dir = absolute(args.run_dir)
    config = load_config(config_path)
    experiment = config["experiment03"]
    seeds = [int(value) for value in experiment["full_seeds"]]
    selected_alpha = float(
        json.loads((run_dir / "pilot_selection.json").read_text(encoding="utf-8"))[
            "selected_alpha"
        ]
    )
    wait_for_base(
        run_dir,
        seeds,
        args.poll_seconds,
        args.wait_timeout_hours * 60.0 * 60.0,
    )
    decisions, scale = base_decisions(run_dir, seeds, config)
    (run_dir / "automatic_control_decisions_preliminary.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "control_decisions", **decisions}), flush=True)
    if decisions["trigger_lrmatch"]:
        run_control_wave(
            method="full_dense_lrmatch",
            seeds=seeds,
            selected_alpha=selected_alpha,
            learning_rate_scale=scale,
            config_path=config_path,
            run_dir=run_dir,
            device=args.device,
        )
    if decisions["trigger_normmatched"]:
        run_control_wave(
            method="full_soft_normmatched",
            seeds=seeds,
            selected_alpha=selected_alpha,
            learning_rate_scale=scale,
            config_path=config_path,
            run_dir=run_dir,
            device=args.device,
        )
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "diagnostics" / "soft_spectral_training.py"),
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
        "--device",
        args.device,
        "--stage",
        "full",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({"status": "experiment03_finalized", "run_dir": str(run_dir)}), flush=True)


if __name__ == "__main__":
    main()
