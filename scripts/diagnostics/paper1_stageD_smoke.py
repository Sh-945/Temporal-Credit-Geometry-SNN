"""Wait for Stage C, then run only the pre-registered DVS-Gesture smoke audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260830


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--cache",
        default=str(Path(tempfile.gettempdir()) / "paper1_dvs_materialized.pt"),
    )
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--wait-for-stage-c", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=43200)
    return parser.parse_args()


def run(command: list[str], *, log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env or os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{tail}")


def wait_for_stage_c(run_dir: Path, timeout: int) -> None:
    expected = run_dir / "stageC_gate" / "stageC_summary.json"
    start = time.monotonic()
    while not expected.is_file():
        print(json.dumps({"status": "waiting_for_stage_c", "complete": False}), flush=True)
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Stage C summary was not created within {timeout} seconds")
        time.sleep(30)
    summary = json.loads(expected.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError(f"Stage C summary is not complete: {summary}")
    print(json.dumps({"status": "waiting_for_stage_c", "complete": True}), flush=True)


def train_smoke(
    run_dir: Path,
    *,
    method: str,
    is_bptt: bool,
    epochs: int,
    device: str,
    env: dict[str, str],
) -> Path:
    output = run_dir / "stageD_cross_dataset" / "smoke" / "checkpoints" / method / f"seed_{SEED}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "diagnostics" / "paper1_worker.py"),
        "train_snn",
        "--config",
        "configs/dvs_gesture/paper1_experiment03.yaml",
        "--seed",
        str(SEED),
        "--method",
        method,
        "--epochs",
        str(epochs),
        "--initial-state",
        str(run_dir / "checkpoints" / "stageD_cross_dataset" / "initial_states" / f"seed_{SEED}.pt"),
        "--output-dir",
        str(output),
        "--device",
        device,
    ]
    if is_bptt:
        command.append("--is-bptt")
    run(command, log_path=run_dir / "logs" / f"stageD_smoke_train_{method}.log", env=env)
    return output


def diagnose_smoke(
    run_dir: Path,
    *,
    method: str,
    is_bptt: bool,
    epoch: int,
    checkpoint: Path,
    device: str,
    env: dict[str, str],
) -> None:
    output = run_dir / "stageD_cross_dataset" / "smoke" / "diagnostics" / method / f"epoch_{epoch:03d}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "diagnostics" / "paper1_worker.py"),
        "diagnose_snn",
        "--seed",
        str(SEED),
        "--method",
        method,
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--epoch",
        str(epoch),
        "--gate-mode",
        "actual",
        "--device",
        device,
    ]
    if is_bptt:
        command.append("--is-bptt")
    run(command, log_path=run_dir / "logs" / f"stageD_smoke_diag_{method}.log", env=env)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if args.wait_for_stage_c:
        wait_for_stage_c(run_dir, args.wait_timeout_seconds)

    stage = run_dir / "stageD_cross_dataset"
    stage.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "diagnostics" / "paper1_stageD_prepare.py"),
            "--config",
            "configs/dvs_gesture/paper1_experiment03.yaml",
            "--run-dir",
            str(run_dir),
            "--full-data-checksum",
        ],
        log_path=run_dir / "logs" / "stageD_prepare.log",
    )

    cache_path = Path(args.cache)
    if not cache_path.is_file():
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "diagnostics" / "paper1_cache_data.py"),
                "--config",
                "configs/dvs_gesture/paper1_experiment03.yaml",
                "--output",
                str(cache_path),
                "--num-workers",
                str(args.cache_workers),
            ],
            log_path=run_dir / "logs" / "stageD_cache_dvs.log",
        )
    env = os.environ.copy()
    env["PAPER1_MATERIALIZED_CACHE"] = str(cache_path)

    outputs = {}
    for method, is_bptt in (("dvs_dfa_smoke", False), ("dvs_bptt_smoke", True)):
        output = train_smoke(
            run_dir,
            method=method,
            is_bptt=is_bptt,
            epochs=args.epochs,
            device=args.device,
            env=env,
        )
        diagnose_smoke(
            run_dir,
            method=method,
            is_bptt=is_bptt,
            epoch=args.epochs,
            checkpoint=output / "last.pt",
            device=args.device,
            env=env,
        )
        outputs[method] = output

    summaries = {
        method: json.loads((output / "variant_summary.json").read_text(encoding="utf-8"))
        for method, output in outputs.items()
    }
    smoke = {}
    for method, output in outputs.items():
        metrics = json.loads((output / "training_metrics.json").read_text(encoding="utf-8"))
        start, final = metrics[0], metrics[-1]
        smoke[method] = {
            "epochs": len(metrics),
            "initial_train_loss": start["train_loss"],
            "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
            "final_validation_accuracy": final["validation_accuracy"],
            "loss_decreased": final["train_loss"] < start["train_loss"],
            "above_chance_validation": final["validation_accuracy"] > 1 / 11,
            "best_validation_test": summaries[method]["best_validation_test"],
            "geometry_complete": (
                stage / "smoke" / "diagnostics" / method / f"epoch_{args.epochs:03d}" / "complete.json"
            ).is_file(),
        }
    reliable = all(
        row["loss_decreased"] and row["above_chance_validation"] and row["geometry_complete"]
        for row in smoke.values()
    )
    result = {
        "status": "complete",
        "dataset": "DVS-Gesture",
        "seed": SEED,
        "smoke_reliable": reliable,
        "results": smoke,
    }
    (stage / "smoke" / "smoke_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
