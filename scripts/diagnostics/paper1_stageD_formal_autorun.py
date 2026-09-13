"""Finish formal Stage D after a validated first wave, then aggregate and report."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (0, 10, 25, 50, 75, 100)


@dataclass(frozen=True)
class Variant:
    method: str
    seed: int
    is_bptt: bool


FIRST_WAVE = (
    Variant("dvs_dfa", 20260830, False),
    Variant("dvs_bptt", 20260830, True),
    Variant("dvs_bptt", 20260831, True),
)
SECOND_WAVE = (
    Variant("dvs_dfa", 20260831, False),
    Variant("dvs_dfa", 20260901, False),
    Variant("dvs_bptt", 20260901, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--cache",
        default=str(Path(tempfile.gettempdir()) / "paper1_dvs_materialized.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-workers", type=int, default=2)
    parser.add_argument("--wait-timeout-seconds", type=int, default=14400)
    return parser.parse_args()


def output_dir(run_dir: Path, variant: Variant) -> Path:
    return (
        run_dir
        / "checkpoints"
        / "stageD_cross_dataset"
        / variant.method
        / f"seed_{variant.seed}"
    )


def summary_path(run_dir: Path, variant: Variant) -> Path:
    return output_dir(run_dir, variant) / "variant_summary.json"


def wait_for_first_wave(run_dir: Path, timeout_seconds: int) -> None:
    start = time.monotonic()
    while True:
        complete = sum(summary_path(run_dir, variant).is_file() for variant in FIRST_WAVE)
        print(json.dumps({"status": "waiting_for_first_wave", "complete": complete, "total": 3}), flush=True)
        if complete == 3:
            return
        if time.monotonic() - start > timeout_seconds:
            missing = [str(summary_path(run_dir, variant)) for variant in FIRST_WAVE if not summary_path(run_dir, variant).is_file()]
            raise TimeoutError("first wave timed out; missing:\n" + "\n".join(missing))
        time.sleep(30)


def run_logged(command: list[str], *, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{tail}")


def train_variant(run_dir: Path, variant: Variant, *, device: str, env: dict[str, str]) -> dict[str, object]:
    destination = output_dir(run_dir, variant)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "diagnostics" / "paper1_worker.py"),
        "train_snn",
        "--config",
        "configs/dvs_gesture/paper1_experiment03.yaml",
        "--seed",
        str(variant.seed),
        "--method",
        variant.method,
        "--epochs",
        "100",
        "--initial-state",
        str(run_dir / "checkpoints" / "stageD_cross_dataset" / "initial_states" / f"seed_{variant.seed}.pt"),
        "--output-dir",
        str(destination),
        "--device",
        device,
    ]
    if variant.is_bptt:
        command.append("--is-bptt")
    run_logged(
        command,
        log_path=run_dir / "logs" / f"stageD_train_{variant.method}_seed_{variant.seed}.log",
        env=env,
    )
    summary = json.loads((destination / "variant_summary.json").read_text(encoding="utf-8"))
    return {
        "status": "complete",
        "method": variant.method,
        "seed": variant.seed,
        "test_accuracy": summary["best_validation_test"]["accuracy"],
    }


def checkpoint(destination: Path, epoch: int) -> Path:
    return destination / ("init.pt" if epoch == 0 else f"epoch_{epoch:03d}.pt")


def diagnose_variant(
    run_dir: Path,
    variant: Variant,
    epoch: int,
    *,
    device: str,
    env: dict[str, str],
) -> dict[str, object]:
    destination = (
        run_dir
        / "stageD_cross_dataset"
        / "diagnostics"
        / variant.method
        / f"seed_{variant.seed}"
        / f"epoch_{epoch:03d}"
    )
    if (destination / "complete.json").is_file():
        return {"status": "skipped", "method": variant.method, "seed": variant.seed, "epoch": epoch}
    command = [
        sys.executable,
        str(ROOT / "scripts" / "diagnostics" / "paper1_worker.py"),
        "diagnose_snn",
        "--seed",
        str(variant.seed),
        "--method",
        variant.method,
        "--checkpoint",
        str(checkpoint(output_dir(run_dir, variant), epoch)),
        "--output-dir",
        str(destination),
        "--epoch",
        str(epoch),
        "--gate-mode",
        "actual",
        "--device",
        device,
    ]
    if variant.is_bptt:
        command.append("--is-bptt")
    run_logged(
        command,
        log_path=run_dir / "logs" / f"stageD_diag_{variant.method}_seed_{variant.seed}_epoch_{epoch:03d}.log",
        env=env,
    )
    return {"status": "complete", "method": variant.method, "seed": variant.seed, "epoch": epoch}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cache = Path(args.cache)
    if not cache.is_file():
        raise FileNotFoundError(cache)
    env = os.environ.copy()
    env["PAPER1_MATERIALIZED_CACHE"] = str(cache)
    wait_for_first_wave(run_dir, args.wait_timeout_seconds)

    free_bytes = shutil.disk_usage(run_dir).free
    if free_bytes < 10 * 1024**3:
        raise RuntimeError(f"less than 10 GiB free before second wave: {free_bytes}")
    pending_variants = [
        variant for variant in SECOND_WAVE if not summary_path(run_dir, variant).is_file()
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(train_variant, run_dir, variant, device=args.device, env=env)
            for variant in pending_variants
        ]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    if not pending_variants:
        print(json.dumps({"status": "training_already_complete", "jobs": 6}), flush=True)

    variants = FIRST_WAVE + SECOND_WAVE
    jobs = [(variant, epoch) for variant in variants for epoch in EPOCHS]
    finished = 0
    with ThreadPoolExecutor(max_workers=max(1, args.diagnostic_workers)) as executor:
        futures = {
            executor.submit(
                diagnose_variant,
                run_dir,
                variant,
                epoch,
                device=args.device,
                env=env,
            ): (variant, epoch)
            for variant, epoch in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            finished += 1
            print(json.dumps({**result, "finished": finished, "total": len(jobs)}), flush=True)

    for script in ("paper1_stageD_aggregate.py", "paper1_finalize.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "diagnostics" / script), "--run-dir", str(run_dir)],
            cwd=ROOT,
            env=env,
            check=True,
        )
    result = {
        "status": "complete",
        "training_jobs": 6,
        "diagnostic_jobs": len(jobs),
        "report": str(run_dir / "SUMMARY_CN.md"),
    }
    (run_dir / "stageD_cross_dataset" / "formal_autorun_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
