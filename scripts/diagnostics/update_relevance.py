"""Experiment 02: causal preservation of actual dense-sDFA weight updates."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.update_relevance.diagnostic import (  # noqa: E402
    evaluate_bptt_counterfactual,
    evaluate_gradient_preservation,
    fit_bptt_bases,
    fit_dfa_bases,
)
from analysis.update_relevance.report import generate_report  # noqa: E402
from methods import FeedbackBank  # noqa: E402
from models import build_model  # noqa: E402
from scripts.diagnostics.feedback_subspace import source_tree_hash  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.config import load_config  # noqa: E402
from training.data import build_datasets  # noqa: E402
from training.seed import seed_everything  # noqa: E402


DEFAULT_SEEDS = [20260830, 20260831, 20260901]
DEFAULT_EPOCHS = [10, 25, 50, 75, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment01-dir", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", nargs="+", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--bptt-epochs", nargs="+", type=int, default=[100])
    parser.add_argument("--loss-epochs", nargs="+", type=int, default=[100])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--basis-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--max-evaluation-batches", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def write_provenance(run_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        f"timestamp={datetime.now().astimezone().isoformat()}",
        f"repository={ROOT}",
        f"experiment01_dir={Path(args.experiment01_dir).resolve()}",
        "training_performed=false",
        "probe_split=first_1024_basis_fit,last_1024_evaluation",
        "projection_path=exact_production_local_proxy_current_gradient",
        "loss_descent=layer-local SGD-equivalent; no Adam state; checkpoint restored",
        f"source_tree_sha256={source_tree_hash()}",
        f"command={' '.join(sys.argv)}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"python_executable={sys.executable}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda_runtime={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        "nvidia_smi=",
        _command_output(["nvidia-smi"]),
        "pip_freeze=",
        _command_output([sys.executable, "-m", "pip", "freeze"]),
    ]
    (run_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize_probe(
    dataset: Dataset,
    indices: list[int],
    *,
    workers: int,
) -> TensorDataset:
    loader_kwargs: dict[str, Any] = {
        "batch_size": 128,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": False,
    }
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(Subset(dataset, indices), **loader_kwargs)
    frames = []
    labels = []
    for samples, target in loader:
        rounded = samples.round()
        if not torch.equal(samples, rounded):
            raise ValueError("probe frames are not losslessly representable as integers")
        frames.append(rounded.to(torch.uint8))
        labels.append(target.to(torch.long))
    return TensorDataset(torch.cat(frames), torch.cat(labels))


def _split_probe(
    experiment01_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    *,
    basis_limit: int | None,
    evaluation_limit: int | None,
) -> tuple[TensorDataset, TensorDataset]:
    probe = pd.read_csv(experiment01_dir / "probe_indices.csv").sort_values(
        "probe_position"
    )
    if len(probe) != 2048:
        raise ValueError(f"Experiment 02 requires exactly 2048 probe rows, got {len(probe)}")
    basis_frame = probe.iloc[:1024].copy()
    evaluation_frame = probe.iloc[1024:].copy()
    if basis_limit is not None:
        basis_frame = basis_frame.iloc[:basis_limit]
    if evaluation_limit is not None:
        evaluation_frame = evaluation_frame.iloc[:evaluation_limit]
    basis_frame.insert(0, "split_position", range(len(basis_frame)))
    evaluation_frame.insert(0, "split_position", range(len(evaluation_frame)))
    basis_frame.to_csv(run_dir / "basis_fit_indices.csv", index=False)
    evaluation_frame.to_csv(run_dir / "evaluation_indices.csv", index=False)
    overlap = set(basis_frame.dataset_index).intersection(evaluation_frame.dataset_index)
    if overlap:
        raise AssertionError(f"basis/evaluation data leakage: {sorted(overlap)[:5]}")

    train_dataset, _test_dataset = build_datasets(config)
    workers = int(config["data"].get("num_workers", 0))
    basis_dataset = _materialize_probe(
        train_dataset, basis_frame.dataset_index.astype(int).tolist(), workers=workers
    )
    evaluation_dataset = _materialize_probe(
        train_dataset,
        evaluation_frame.dataset_index.astype(int).tolist(),
        workers=workers,
    )
    return basis_dataset, evaluation_dataset


def _load_rank_schedule(
    experiment01_dir: Path,
    hidden_dimensions: list[int],
) -> dict[int, list[tuple[str, int]]]:
    recommendations = pd.read_csv(
        experiment01_dir / "candidate_rank_recommendations.csv"
    ).set_index("layer")
    metrics = pd.read_csv(experiment01_dir / "subspace_metrics.csv")
    final_epoch = int(metrics.epoch.max())
    common = (
        (metrics.epoch == final_epoch)
        & (metrics.label_condition == "correct")
        & (metrics.signal_type == "post_gate_delta")
        & (metrics.analysis_mode == "centered_raw")
    )
    primary = metrics[common & (metrics.temporal_mode == "aggregated")]
    timestep = metrics[common & (metrics.temporal_mode == "timestep")]
    result = {}
    for layer_index, dimension in enumerate(hidden_dimensions):
        layer = f"hidden_{layer_index + 1}"
        result[layer_index] = [
            ("entropy", int(round(primary[primary.layer == layer].entropy_rank.mean()))),
            ("r90", int(recommendations.loc[layer, "aggressive_recommended"])),
            ("r95", int(recommendations.loc[layer, "balanced_recommended"])),
            ("r99", int(recommendations.loc[layer, "conservative_recommended"])),
            (
                "timestep_r95",
                int(math.ceil(timestep[timestep.layer == layer].r95.mean())),
            ),
            ("full_D", int(dimension)),
        ]
    return result


def _checkpoint_path(experiment01_dir: Path, seed: int, epoch: int) -> Path:
    path = (
        experiment01_dir
        / "checkpoints"
        / f"seed_{seed}"
        / f"epoch_{epoch:03d}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _save_tables(
    run_dir: Path,
    gradient_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    bptt_rows: list[dict[str, Any]],
    angle_rows: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(gradient_rows).to_csv(run_dir / "gradient_preservation.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(run_dir / "loss_descent.csv", index=False)
    pd.DataFrame(bptt_rows).to_csv(
        run_dir / "bptt_counterfactual_metrics.csv", index=False
    )
    pd.DataFrame(angle_rows).to_csv(run_dir / "principal_angles.csv", index=False)
    pd.DataFrame(basis_rows).to_csv(run_dir / "basis_metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    experiment01_dir = Path(args.experiment01_dir).expanduser().resolve()
    if not (experiment01_dir / "run_manifest.json").is_file():
        raise FileNotFoundError(experiment01_dir / "run_manifest.json")
    if args.random_repeats < 5 and not args.smoke:
        raise ValueError("main Experiment 02 requires at least 5 random repeats")
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.epochs = [max(args.epochs)]
        args.bptt_epochs = list(args.epochs)
        args.loss_epochs = list(args.epochs)
        args.random_repeats = 1
        args.basis_limit = args.basis_limit or 64
        args.evaluation_limit = args.evaluation_limit or 64
        args.max_evaluation_batches = args.max_evaluation_batches or 1
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    default_run = ROOT / "results" / "update_relevance_diagnostic" / (
        f"experiment02_{timestamp}" + ("_smoke" if args.smoke else "")
    )
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else default_run
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "config_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    basis_dir = run_dir / "bases"
    basis_dir.mkdir(exist_ok=True)
    write_provenance(run_dir, args)

    configs: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        source = experiment01_dir / "configs" / f"seed_{seed}.yaml"
        configs[seed] = load_config(source)
        shutil.copy2(source, snapshot_dir / source.name)
    base_config = configs[args.seeds[0]]
    if str(base_config["method"]["name"]).lower() not in {"dfa", "sdfa"}:
        raise ValueError("Experiment 02 requires an existing dense DFA/sDFA run")
    if str(base_config["model"]["architecture"]).lower() != "fc":
        raise ValueError("Experiment 02 primary causal diagnostic requires FC SNN")

    basis_dataset, evaluation_dataset = _split_probe(
        experiment01_dir,
        run_dir,
        base_config,
        basis_limit=args.basis_limit,
        evaluation_limit=args.evaluation_limit,
    )
    pin_memory = device.type == "cuda"
    basis_loader = DataLoader(
        basis_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    if args.max_evaluation_batches is not None:
        evaluation_batches = []
        for index, batch in enumerate(evaluation_loader):
            if index >= args.max_evaluation_batches:
                break
            evaluation_batches.append(batch)
        evaluation_loader = evaluation_batches

    hidden_dimensions = [
        int(value) for value in base_config["model"]["hidden_features"]
    ]
    ranks_by_layer = _load_rank_schedule(experiment01_dir, hidden_dimensions)
    settings = {
        "experiment01_dir": str(experiment01_dir),
        "run_dir": str(run_dir),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "bptt_epochs": args.bptt_epochs,
        "loss_epochs": args.loss_epochs,
        "basis_fit_samples": len(basis_dataset),
        "evaluation_samples": len(evaluation_dataset),
        "batch_size": args.batch_size,
        "random_repeats": args.random_repeats,
        "ranks_by_layer": {
            f"hidden_{index + 1}": values
            for index, values in ranks_by_layer.items()
        },
    }
    (snapshot_dir / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "status": "running",
        "training_performed": False,
        "experiment01_dir": str(experiment01_dir),
        "run_dir": str(run_dir),
        "settings": settings,
        "completed": [],
        "source_tree_sha256": source_tree_hash(),
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    gradient_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    bptt_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        config = configs[seed]
        pointwise = config["method"].get("temporal_mode", "pointwise") == "pointwise"
        for epoch in args.epochs:
            seed_everything(seed)
            model = build_model(config).to(device)
            feedback_bank = FeedbackBank(model, config).to(device)
            checkpoint_path = _checkpoint_path(experiment01_dir, seed, epoch)
            state = load_checkpoint(
                checkpoint_path,
                model=model,
                feedback_bank=feedback_bank,
                map_location=device,
            )
            model.eval()
            feedback_bank.eval()
            learning_rate = float(state["optimizer_state"]["param_groups"][0]["lr"])

            dfa_bases, dfa_metadata = fit_dfa_bases(
                model,
                feedback_bank,
                basis_loader,
                device,
                seed=seed,
                epoch=epoch,
                pointwise=pointwise,
            )
            basis_rows.extend(dfa_metadata)
            rows, losses = evaluate_gradient_preservation(
                model,
                feedback_bank,
                evaluation_loader,
                device,
                seed=seed,
                epoch=epoch,
                bases=dfa_bases,
                ranks_by_layer=ranks_by_layer,
                random_repeats=args.random_repeats,
                pointwise=pointwise,
                learning_rate=learning_rate,
                run_loss_descent=epoch in args.loss_epochs,
            )
            gradient_rows.extend(rows)
            loss_rows.extend(losses)

            if epoch in args.bptt_epochs:
                bptt_bases, bptt_metadata = fit_bptt_bases(
                    model,
                    basis_loader,
                    device,
                    seed=seed,
                    epoch=epoch,
                )
                basis_rows.extend(bptt_metadata)
                counterfactual, angles = evaluate_bptt_counterfactual(
                    model,
                    feedback_bank,
                    evaluation_loader,
                    device,
                    seed=seed,
                    epoch=epoch,
                    dfa_fit_bases=dfa_bases,
                    dfa_fit_metadata=dfa_metadata,
                    bptt_fit_bases=bptt_bases,
                    bptt_fit_metadata=bptt_metadata,
                    pointwise=pointwise,
                )
                bptt_rows.extend(counterfactual)
                angle_rows.extend(angles)
                torch.save(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "dfa": {
                            f"hidden_{layer + 1}_{mode}": value.cpu()
                            for (layer, mode), value in dfa_bases.items()
                        },
                        "bptt": {
                            f"hidden_{layer + 1}_{mode}": value.cpu()
                            for (layer, mode), value in bptt_bases.items()
                        },
                    },
                    basis_dir / f"seed_{seed}_epoch_{epoch:03d}.pt",
                )

            manifest["completed"].append(
                {"seed": seed, "epoch": epoch, "checkpoint": str(checkpoint_path)}
            )
            _save_tables(
                run_dir,
                gradient_rows,
                loss_rows,
                bptt_rows,
                angle_rows,
                basis_rows,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "status": "checkpoint_complete",
                        "seed": seed,
                        "epoch": epoch,
                        "gradient_rows": len(gradient_rows),
                        "loss_rows": len(loss_rows),
                    }
                ),
                flush=True,
            )
            del model, feedback_bank, dfa_bases
            if device.type == "cuda":
                torch.cuda.empty_cache()

    manifest["status"] = "complete"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    verdict = generate_report(run_dir, experiment01_dir)
    print(
        json.dumps(
            {"status": "complete", "verdict": verdict, "run_dir": str(run_dir)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
