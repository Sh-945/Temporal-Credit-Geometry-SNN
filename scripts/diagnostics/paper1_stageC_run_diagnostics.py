"""Run all pre-registered Stage C checkpoint diagnostics with bounded concurrency."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (0, 10, 25, 50, 75, 100)
SEEDS = (20260830, 20260831, 20260901)


@dataclass(frozen=True)
class Job:
    method: str
    source_method: str
    gate_mode: str
    seed: int
    epoch: int
    checkpoint: Path
    output_dir: Path
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-timeout-seconds", type=int, default=21600)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def checkpoint_path(root: Path, epoch: int) -> Path:
    return root / ("init.pt" if epoch == 0 else f"epoch_{epoch:03d}.pt")


def build_jobs(run_dir: Path) -> list[Job]:
    jobs: list[Job] = []
    variants = (
        ("mean_gate", "mean_gate", "temporal_mean_normmatched"),
        ("mean_gate_unmatched_offline", "mean_gate", "temporal_mean_unmatched"),
        ("shuffled_gate", "shuffled_gate", "timestep_shuffled"),
    )
    for method, source_method, gate_mode in variants:
        for seed in SEEDS:
            source = run_dir / "checkpoints" / "stageC_gate" / source_method / f"seed_{seed}"
            for epoch in EPOCHS:
                output = (
                    run_dir
                    / "stageC_gate"
                    / "diagnostics"
                    / method
                    / f"seed_{seed}"
                    / f"epoch_{epoch:03d}"
                )
                jobs.append(
                    Job(
                        method=method,
                        source_method=source_method,
                        gate_mode=gate_mode,
                        seed=seed,
                        epoch=epoch,
                        checkpoint=checkpoint_path(source, epoch),
                        output_dir=output,
                        log_path=run_dir / "logs" / f"stageC_diag_{method}_seed{seed}_epoch{epoch:03d}.log",
                    )
                )
    return jobs


def training_summaries(run_dir: Path) -> list[Path]:
    return [
        run_dir
        / "checkpoints"
        / "stageC_gate"
        / method
        / f"seed_{seed}"
        / "variant_summary.json"
        for method in ("mean_gate", "shuffled_gate")
        for seed in SEEDS
    ]


def wait_for_training(run_dir: Path, *, poll_seconds: int, timeout_seconds: int) -> None:
    expected = training_summaries(run_dir)
    start = time.monotonic()
    while True:
        complete = sum(path.is_file() for path in expected)
        print(json.dumps({"status": "waiting_for_training", "complete": complete, "total": len(expected)}), flush=True)
        if complete == len(expected):
            return
        if time.monotonic() - start > timeout_seconds:
            missing = [str(path) for path in expected if not path.is_file()]
            raise TimeoutError("training wait timed out; missing:\n" + "\n".join(missing))
        time.sleep(max(5, min(60, int(poll_seconds))))


def run_job(job: Job, device: str) -> dict[str, object]:
    complete = job.output_dir / "complete.json"
    if complete.is_file():
        return {"status": "skipped", "method": job.method, "seed": job.seed, "epoch": job.epoch}
    if not job.checkpoint.is_file():
        raise FileNotFoundError(job.checkpoint)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "diagnostics" / "paper1_worker.py"),
        "diagnose_snn",
        "--seed",
        str(job.seed),
        "--method",
        job.method,
        "--checkpoint",
        str(job.checkpoint),
        "--output-dir",
        str(job.output_dir),
        "--epoch",
        str(job.epoch),
        "--gate-mode",
        job.gate_mode,
        "--device",
        device,
    ]
    with job.log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not complete.is_file():
        tail = job.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"diagnostic failed: {job.method} seed={job.seed} epoch={job.epoch}\n{tail}"
        )
    return {"status": "complete", "method": job.method, "seed": job.seed, "epoch": job.epoch}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if args.wait_for_training:
        wait_for_training(
            run_dir,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.wait_timeout_seconds,
        )
    jobs = build_jobs(run_dir)
    missing = [str(job.checkpoint) for job in jobs if not job.checkpoint.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage C training is incomplete; missing checkpoints:\n" + "\n".join(missing)
        )
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "jobs": len(jobs), "workers": args.max_workers}))
        return
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        pending = {executor.submit(run_job, job, args.device): job for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(json.dumps({**result, "finished": len(results), "total": len(jobs)}), flush=True)
    completed = sum(row["status"] == "complete" for row in results)
    skipped = sum(row["status"] == "skipped" for row in results)
    if args.aggregate:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "diagnostics" / "paper1_stageC_aggregate.py"),
                "--run-dir",
                str(run_dir),
            ],
            cwd=ROOT,
            check=True,
        )
    print(json.dumps({"status": "complete", "jobs": len(jobs), "completed": completed, "skipped": skipped}))


if __name__ == "__main__":
    main()
