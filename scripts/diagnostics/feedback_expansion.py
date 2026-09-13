"""Run Experiment 01B from immutable Experiment 01 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.feedback_expansion.counterfactual import counterfactual_decomposition  # noqa: E402
from analysis.feedback_expansion.diagnostic import (  # noqa: E402
    CheckpointDiagnostics,
    diagnose_checkpoint,
    save_basis_manifest,
)
from analysis.feedback_expansion.report import generate_report  # noqa: E402
from analysis.feedback_expansion.trajectory import (  # noqa: E402
    analyze_seed_trajectory,
    enrich_shuffled_control,
)
from training.config import load_config  # noqa: E402
from training.data import build_datasets  # noqa: E402
from training.seed import seed_everything  # noqa: E402


DEFAULT_SEEDS = [20260830, 20260831, 20260901]
SCHEDULED_EPOCHS = [0, 10, 25, 50, 75, 100]
DETAILED_EPOCHS = {0, 10, 50, 100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        default="results/feedback_subspace_diagnostic/experiment01_20260830_server",
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    return parser.parse_args()


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(ROOT.rglob("*.py")):
        if "results" not in path.parts and "__pycache__" not in path.parts:
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_environment(run_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        f"timestamp={datetime.now().astimezone().isoformat()}",
        f"command={' '.join(sys.argv)}",
        f"source_run={_absolute(args.source_run)}",
        f"source_tree_sha256={_source_hash()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        "nvidia_smi=",
        _command_output(["nvidia-smi"]),
    ]
    (run_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def checkpoint_entries(source_run: Path, seed: int) -> list[dict[str, Any]]:
    directory = source_run / "checkpoints" / f"seed_{seed}"
    candidates: list[dict[str, Any]] = []
    init = directory / "init.pt"
    if init.is_file():
        candidates.append({"epoch": 0, "kind": "initialization", "path": init})
    for epoch in SCHEDULED_EPOCHS[1:]:
        path = directory / f"epoch_{epoch:03d}.pt"
        if not path.is_file() and epoch == 100:
            path = directory / "last.pt"
        if path.is_file():
            candidates.append({"epoch": epoch, "kind": "scheduled", "path": path})
    best = directory / "best.pt"
    if best.is_file():
        state = torch.load(best, map_location="cpu", weights_only=False)
        epoch = int(state["epoch"]) + 1
        if epoch not in {entry["epoch"] for entry in candidates}:
            candidates.append({"epoch": epoch, "kind": "best", "path": best})
        else:
            for entry in candidates:
                if entry["epoch"] == epoch:
                    entry["kind"] += "+best"
    candidates.sort(key=lambda item: (item["epoch"], item["kind"] == "best"))
    missing = set(SCHEDULED_EPOCHS).difference(entry["epoch"] for entry in candidates)
    if missing:
        raise FileNotFoundError(f"seed {seed} missing scheduled checkpoints: {sorted(missing)}")
    return candidates


def probe_loaders(
    source_run: Path,
    config: dict[str, Any],
    run_dir: Path,
    batch_size: int,
    smoke: bool,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    frame = pd.read_csv(source_run / "probe_indices.csv").sort_values("probe_position")
    if len(frame) != 2048 and not smoke:
        raise ValueError(f"Experiment 01B expects the original 2048 probe rows, found {len(frame)}")
    midpoint = len(frame) // 2
    fit_frame = frame.iloc[:midpoint].copy()
    eval_frame = frame.iloc[midpoint:].copy()
    if smoke:
        fit_frame = fit_frame.iloc[:8].copy()
        eval_frame = eval_frame.iloc[:8].copy()
    fit_frame.insert(0, "split_position", range(len(fit_frame)))
    eval_frame.insert(0, "split_position", range(len(eval_frame)))
    fit_frame.to_csv(run_dir / "basis_fit_indices.csv", index=False)
    eval_frame.to_csv(run_dir / "basis_eval_indices.csv", index=False)
    train_dataset, _test_dataset = build_datasets(config)

    def materialize(selected: pd.DataFrame) -> TensorDataset:
        samples = []
        labels = []
        for row in selected.itertuples(index=False):
            sample, label = train_dataset[int(row.dataset_index)]
            if int(label) != int(row.correct_label):
                raise AssertionError(
                    f"probe label mismatch at dataset index {row.dataset_index}"
                )
            samples.append(sample.to(torch.uint8))
            labels.append(int(label))
        return TensorDataset(torch.stack(samples), torch.tensor(labels, dtype=torch.long))

    fit_dataset = materialize(fit_frame)
    eval_dataset = materialize(eval_frame)
    common = {
        "batch_size": min(batch_size, len(fit_dataset)),
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": bool(config["data"].get("pin_memory", False)),
    }
    fit_loader = DataLoader(fit_dataset, **common)
    common["batch_size"] = min(batch_size, len(eval_dataset))
    eval_loader = DataLoader(eval_dataset, **common)
    fit_shuffled = torch.tensor(fit_frame.shuffled_label.to_numpy(), dtype=torch.long)
    eval_shuffled = torch.tensor(eval_frame.shuffled_label.to_numpy(), dtype=torch.long)
    return fit_loader, eval_loader, fit_shuffled, eval_shuffled


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    source_run = _absolute(args.source_run)
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = _absolute(args.run_dir) if args.run_dir else (
        ROOT / "results" / "feedback_subspace_expansion" / f"experiment01B_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bases").mkdir(exist_ok=True)
    (run_dir / "configs").mkdir(exist_ok=True)
    write_environment(run_dir, args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = args.seeds[:1] if args.smoke else args.seeds

    all_spectral: list[dict[str, Any]] = []
    all_gate: list[dict[str, Any]] = []
    all_class: list[dict[str, Any]] = []
    all_temporal: list[dict[str, Any]] = []
    all_coherence: list[dict[str, Any]] = []
    all_control: list[dict[str, Any]] = []
    all_sanity: list[dict[str, Any]] = []
    all_overlap: list[dict[str, Any]] = []
    all_novel: list[dict[str, Any]] = []
    all_capture: list[dict[str, Any]] = []
    all_activation: list[dict[str, Any]] = []
    all_direction_energy: list[dict[str, Any]] = []
    all_counterfactual: list[dict[str, Any]] = []
    all_bases = []
    completed: dict[str, list[dict[str, Any]]] = {}

    for seed_index, seed in enumerate(seeds):
        config_path = source_run / "configs" / f"seed_{seed}.yaml"
        config = load_config(config_path)
        shutil.copy2(config_path, run_dir / "configs" / config_path.name)
        entries = checkpoint_entries(source_run, seed)
        if args.smoke:
            entries = [entries[0], next(entry for entry in entries if entry["epoch"] == 10)]
        fit_loader, eval_loader, fit_shuffled, eval_shuffled = probe_loaders(
            source_run, config, run_dir, args.batch_size, args.smoke
        )
        seed_everything(seed)
        seed_results: dict[int, CheckpointDiagnostics] = {}
        path_map: dict[int, Path] = {}
        for entry in entries:
            epoch = int(entry["epoch"])
            detailed = epoch in DETAILED_EPOCHS
            result = diagnose_checkpoint(
                config=config,
                checkpoint_path=Path(entry["path"]),
                checkpoint_kind=str(entry["kind"]),
                epoch=epoch,
                seed=seed,
                fit_loader=fit_loader,
                eval_loader=eval_loader,
                fit_shuffled_labels=fit_shuffled,
                eval_shuffled_labels=eval_shuffled,
                device=device,
                bases_dir=run_dir / "bases",
                detailed=detailed,
                sanity=(seed_index == 0 and epoch == 0),
            )
            seed_results[epoch] = result
            path_map[epoch] = Path(entry["path"])
            all_spectral.extend(result.spectral_rows)
            all_gate.extend(result.gate_rows)
            all_class.extend(result.class_rows)
            all_temporal.extend(result.temporal_rows)
            all_coherence.extend(result.coherence_rows)
            all_sanity.extend(result.sanity_rows)
            all_bases.extend(result.bases.values())
            print(
                json.dumps(
                    {
                        "status": "checkpoint_complete",
                        "seed": seed,
                        "epoch": epoch,
                        "kind": entry["kind"],
                        "detailed": detailed,
                    }
                ),
                flush=True,
            )
        scheduled = [epoch for epoch in SCHEDULED_EPOCHS if epoch in seed_results]
        if args.smoke:
            scheduled = sorted(seed_results)
        trajectory = analyze_seed_trajectory(
            seed=seed,
            checkpoints=seed_results,
            scheduled_epochs=scheduled,
        )
        all_overlap.extend(trajectory["overlap"])
        all_novel.extend(trajectory["novel"])
        all_capture.extend(trajectory["capture"])
        all_activation.extend(trajectory["activation"])
        all_direction_energy.extend(trajectory["direction_energy"])
        final_epoch = max(scheduled)
        all_control.extend(
            enrich_shuffled_control(
                seed=seed, checkpoints=seed_results, final_epoch=final_epoch
            )
        )
        if not args.skip_counterfactual:
            all_counterfactual.extend(
                counterfactual_decomposition(
                    config=config,
                    checkpoint_paths=path_map,
                    scheduled_epochs=scheduled,
                    seed=seed,
                    eval_loader=eval_loader,
                    device=device,
                )
            )
        completed[str(seed)] = [
            {"epoch": int(entry["epoch"]), "kind": entry["kind"], "path": str(entry["path"])}
            for entry in entries
        ]

    _write_csv(run_dir / "spectral_broadening.csv", all_spectral)
    _write_csv(run_dir / "gate_dynamics.csv", all_gate)
    _write_csv(run_dir / "class_subspace_metrics.csv", all_class)
    _write_csv(run_dir / "temporal_subspace_metrics.csv", all_temporal)
    _write_csv(run_dir / "temporal_coherence.csv", all_coherence)
    _write_csv(run_dir / "subspace_overlap.csv", all_overlap)
    _write_csv(run_dir / "novel_direction_energy.csv", all_novel)
    _write_csv(run_dir / "final_subspace_capture.csv", all_capture)
    _write_csv(run_dir / "direction_activation.csv", all_activation)
    _write_csv(run_dir / "direction_energy.csv", all_direction_energy)
    _write_csv(run_dir / "correct_vs_shuffled.csv", all_control)
    _write_csv(run_dir / "counterfactual_decomposition.csv", all_counterfactual)
    _write_csv(run_dir / "svd_sanity_check.csv", all_sanity)
    save_basis_manifest(all_bases, run_dir / "basis_manifest.json")
    shutil.copy2(source_run / "training_metrics.csv", run_dir / "training_metrics.csv")
    training = pd.read_csv(run_dir / "training_metrics.csv")
    manifest = {
        "status": "smoke_complete" if args.smoke else "complete",
        "source_run": str(source_run),
        "run_dir": str(run_dir),
        "retrained": False,
        "probe_source": str(source_run / "probe_indices.csv"),
        "split_rule": "first half basis_fit; second half basis_eval",
        "primary_analysis": "centered_raw",
        "optional_unit_direction_run": False,
        "seeds": seeds,
        "checkpoints": completed,
        "counterfactual_run": not args.skip_counterfactual,
        "source_tree_sha256": _source_hash(),
    }
    if not args.smoke:
        verdicts = generate_report(run_dir, training)
        manifest["verdicts"] = verdicts
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir), "verdicts": manifest.get("verdicts")}), flush=True)


if __name__ == "__main__":
    main()
