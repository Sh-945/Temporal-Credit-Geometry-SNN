"""Run dense-sDFA training and effective feedback-subspace Experiment 01."""

from __future__ import annotations

import argparse
import copy
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
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.feedback_subspace.diagnostic import (  # noqa: E402
    diagnose_checkpoint,
    verify_fixed_feedback,
)
from analysis.feedback_subspace.report import generate_report  # noqa: E402
from methods import FeedbackBank  # noqa: E402
from models import build_model  # noqa: E402
from training.checkpoint import save_checkpoint  # noqa: E402
from training.config import load_config  # noqa: E402
from training.data import build_datasets  # noqa: E402
from training.engine import Trainer  # noqa: E402
from training.seed import seed_everything  # noqa: E402


DEFAULT_SEEDS = [20260830, 20260831, 20260901]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/nmnist/feedback_subspace_4fc_sdfa.yaml",
    )
    parser.add_argument("--run-dir", help="existing/new result directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="reuse complete checkpoints already present in --run-dir",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def smoke_config(config: dict[str, Any]) -> None:
    config["data"].update(
        {
            "dataset": "synthetic",
            "root": ".",
            "time_steps": 4,
            "train_samples": 24,
            "test_samples": 12,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    )
    config["model"]["input_shape"] = [2, 8, 8]
    config["model"]["hidden_features"] = [16, 12, 10]
    config["training"].update(
        {"epochs": 2, "checkpoint_epochs": [1, 2], "lr_step": 0}
    )
    config["diagnostic"].update(
        {"probe_size": 8, "batch_size": 4, "preload_dataset": True}
    )


def source_tree_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".py", ".yaml", ".yml", ".txt"}
        and "results" not in path.parts
        and "outputs" not in path.parts
        and "__pycache__" not in path.parts
    )
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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


def write_provenance(run_dir: Path, args: argparse.Namespace) -> None:
    git_commit = _command_output(["git", "rev-parse", "HEAD"])
    git_status = _command_output(["git", "status", "--short"])
    lines = [
        f"timestamp={datetime.now().astimezone().isoformat()}",
        f"repository={ROOT}",
        f"git_commit={git_commit}",
        f"git_dirty_status={git_status}",
        f"source_tree_sha256={source_tree_hash()}",
        f"command={' '.join(sys.argv)}",
        f"seeds={','.join(map(str, args.seeds))}",
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


def cache_dataset(dataset: Dataset, name: str, workers: int) -> TensorDataset:
    """Materialize integer event frames once as uint8, preserving samples exactly."""

    loader_kwargs: dict[str, Any] = {
        "batch_size": 128,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": False,
    }
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(dataset, **loader_kwargs)
    frames: Tensor | None = None
    labels = torch.empty(len(dataset), dtype=torch.long)
    offset = 0
    for batch_index, (samples, target) in enumerate(loader):
        rounded = samples.round()
        if not torch.equal(samples, rounded) or float(samples.min()) < 0 or float(samples.max()) > 255:
            raise ValueError(f"{name} frames are not losslessly representable as uint8")
        samples_uint8 = rounded.to(torch.uint8)
        if frames is None:
            frames = torch.empty(
                (len(dataset), *samples_uint8.shape[1:]), dtype=torch.uint8
            )
        end = offset + len(target)
        frames[offset:end].copy_(samples_uint8)
        labels[offset:end].copy_(target)
        offset = end
        if batch_index % 100 == 0:
            print(json.dumps({"cache": name, "samples": offset, "total": len(dataset)}), flush=True)
    if frames is None or offset != len(dataset):
        raise RuntimeError(f"failed to cache complete {name} dataset")
    # Exactness check at deterministic endpoints.
    for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        original, target = dataset[index]
        if not torch.equal(frames[index].to(torch.float32), original):
            raise AssertionError(f"cached frame mismatch at {name}[{index}]")
        if int(labels[index]) != int(target):
            raise AssertionError(f"cached label mismatch at {name}[{index}]")
    return TensorDataset(frames, labels)


def training_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    data = config["data"]
    cached = isinstance(train_dataset, TensorDataset)
    workers = 0 if cached else int(data.get("num_workers", 0))
    common: dict[str, Any] = {
        "batch_size": int(data["batch_size"]),
        "num_workers": workers,
        "pin_memory": bool(data.get("pin_memory", False)),
    }
    if workers > 0:
        common["persistent_workers"] = bool(data.get("persistent_workers", False))
        common["prefetch_factor"] = int(data.get("prefetch_factor", 2))
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )


def seed_config(base: dict[str, Any], seed: int, run_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = f"nmnist_4fc_sdfa_feedback_subspace_seed{seed}"
    config["training"]["output_dir"] = str(run_dir / "checkpoints" / f"seed_{seed}")
    return config


def train_seed(
    config: dict[str, Any],
    train_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
) -> dict[str, Any]:
    seed = int(config["experiment"]["seed"])
    output_dir = Path(config["training"]["output_dir"])
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    seed_everything(seed)
    train_loader, test_loader = training_loaders(train_dataset, test_dataset, config)
    model = build_model(config)
    feedback_bank = FeedbackBank(model, config)
    trainer = Trainer(model, feedback_bank, config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "init.pt",
        epoch=-1,
        model=trainer.model,
        feedback_bank=trainer.feedback_bank,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        config=config,
        metrics={"stage": "initialization", "epoch": -1},
    )
    summary = trainer.fit(train_loader, test_loader)
    del trainer, model, feedback_bank, train_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def write_training_metrics(
    run_dir: Path, seeds: list[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = run_dir / "checkpoints" / f"seed_{seed}" / "metrics.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            rows.append(
                {
                    "seed": seed,
                    "epoch": int(record["epoch"]) + 1,
                    "train_loss": record["train"]["loss"],
                    "train_acc": record["train"]["accuracy"],
                    "test_loss": record["test"]["loss"],
                    "test_acc": record["test"]["accuracy"],
                    "lr": record["learning_rate"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "training_metrics.csv", index=False)
    return frame


def checkpoint_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = Path(config["training"]["output_dir"])
    epochs = int(config["training"]["epochs"])
    entries: dict[int, dict[str, Any]] = {
        0: {"epoch": 0, "path": output_dir / "init.pt", "kinds": {"initialization"}}
    }
    for epoch in config["training"].get("checkpoint_epochs", []):
        path = output_dir / f"epoch_{int(epoch):03d}.pt"
        if path.is_file():
            entries[int(epoch)] = {
                "epoch": int(epoch),
                "path": path,
                "kinds": {"scheduled"},
            }
    last_path = output_dir / "last.pt"
    if last_path.is_file():
        entries.setdefault(
            epochs, {"epoch": epochs, "path": last_path, "kinds": set()}
        )["kinds"].add("final")
    best_path = output_dir / "best.pt"
    if best_path.is_file():
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        best_epoch = int(state["epoch"]) + 1
        entries.setdefault(
            best_epoch, {"epoch": best_epoch, "path": best_path, "kinds": set()}
        )["kinds"].add("best")
    result = []
    for entry in entries.values():
        if not Path(entry["path"]).is_file():
            raise FileNotFoundError(entry["path"])
        entry["kind"] = "+".join(sorted(entry.pop("kinds")))
        result.append(entry)
    return sorted(result, key=lambda item: item["epoch"])


def fixed_probe(
    run_dir: Path,
    train_dataset: Dataset,
    config: dict[str, Any],
) -> tuple[DataLoader, Tensor]:
    diagnostic = config["diagnostic"]
    count = min(int(diagnostic["probe_size"]), len(train_dataset))
    rng = np.random.default_rng(int(diagnostic["probe_seed"]))
    indices = np.sort(rng.choice(len(train_dataset), size=count, replace=False))
    if isinstance(train_dataset, TensorDataset):
        all_frames, all_labels = train_dataset.tensors
        index_tensor = torch.from_numpy(indices).long()
        frames = all_frames.index_select(0, index_tensor)
        labels = all_labels.index_select(0, index_tensor)
    else:
        selected = [train_dataset[int(index)] for index in indices]
        frames = torch.stack([item[0] for item in selected])
        labels = torch.stack([item[1] for item in selected])
    shuffle_generator = torch.Generator().manual_seed(
        int(diagnostic["shuffled_label_seed"])
    )
    shuffled = labels[torch.randperm(len(labels), generator=shuffle_generator)]
    with (run_dir / "probe_indices.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["probe_position", "dataset_index", "correct_label", "shuffled_label"])
        for position, (index, correct, random_label) in enumerate(
            zip(indices, labels.tolist(), shuffled.tolist())
        ):
            writer.writerow([position, int(index), int(correct), int(random_label)])
    loader = DataLoader(
        TensorDataset(frames, labels),
        batch_size=int(diagnostic["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(config["data"].get("pin_memory", False)),
    )
    return loader, shuffled


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    base_config = load_config(config_path)
    if args.smoke:
        smoke_config(base_config)
        args.seeds = args.seeds[:1]
    if str(base_config["method"]["name"]).lower() != "sdfa":
        raise ValueError("Experiment 01 must use dense sDFA")
    if len(base_config["model"]["hidden_features"]) < 3:
        raise ValueError("selected FC network must contain at least three hidden layers")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    default_dir = ROOT / "results" / "feedback_subspace_diagnostic" / (
        timestamp + ("_smoke" if args.smoke else "")
    )
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else default_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(exist_ok=True)
    write_provenance(run_dir, args)
    shutil.copy2(config_path, run_dir / "configs" / "base_config.yaml")
    device = choose_device(args.device)

    # One lossless in-memory cache is reused by all seeds.  This changes only
    # storage dtype; Trainer converts every batch back to float32 before use.
    raw_train, raw_test = build_datasets(base_config)
    if bool(base_config["diagnostic"].get("preload_dataset", False)):
        workers = int(base_config["data"].get("num_workers", 0))
        train_dataset = cache_dataset(raw_train, "train", workers)
        test_dataset = cache_dataset(raw_test, "test", workers)
    else:
        train_dataset, test_dataset = raw_train, raw_test

    completed_seeds: list[int] = []
    configs: dict[int, dict[str, Any]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for seed_index, seed in enumerate(args.seeds):
        config = seed_config(base_config, seed, run_dir)
        configs[seed] = config
        with (run_dir / "configs" / f"seed_{seed}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {key: value for key, value in config.items() if not key.startswith("_")},
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        if args.skip_training:
            summary_path = Path(config["training"]["output_dir"]) / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = train_seed(config, train_dataset, test_dataset, device)
        summaries[seed] = summary
        completed_seeds.append(seed)
        if seed_index == 0 and float(summary["best_accuracy"]) < 0.95:
            print(
                json.dumps(
                    {
                        "baseline_invalid": True,
                        "seed": seed,
                        "best_accuracy": summary["best_accuracy"],
                        "action": "stop_additional_seeds_and_diagnose_first_seed",
                    }
                ),
                flush=True,
            )
            break

    training = write_training_metrics(run_dir, completed_seeds)
    if training.empty:
        raise RuntimeError("no training metrics were produced")
    probe_loader, shuffled_labels = fixed_probe(
        run_dir, train_dataset, configs[completed_seeds[0]]
    )

    all_metrics: list[dict[str, Any]] = []
    all_spectra: list[dict[str, Any]] = []
    all_feedback: list[dict[str, Any]] = []
    for seed in completed_seeds:
        config = configs[seed]
        for entry in checkpoint_entries(config):
            include_shuffle = "best" in entry["kind"] or "final" in entry["kind"]
            metrics, spectra, feedback = diagnose_checkpoint(
                config=config,
                checkpoint_path=Path(entry["path"]),
                checkpoint_kind=str(entry["kind"]),
                epoch=int(entry["epoch"]),
                seed=seed,
                probe_batches=probe_loader,
                device=device,
                shuffled_labels=shuffled_labels if include_shuffle else None,
            )
            all_metrics.extend(metrics)
            all_spectra.extend(spectra)
            all_feedback.extend(feedback)
            print(
                json.dumps(
                    {
                        "diagnosed_seed": seed,
                        "epoch": entry["epoch"],
                        "kind": entry["kind"],
                    }
                ),
                flush=True,
            )

    verify_fixed_feedback(all_feedback)
    pd.DataFrame(all_metrics).to_csv(run_dir / "subspace_metrics.csv", index=False)
    pd.DataFrame(all_spectra).to_csv(run_dir / "singular_values.csv", index=False)
    pd.DataFrame(all_feedback).to_csv(run_dir / "feedback_matrix_checks.csv", index=False)
    manifest = {
        "status": "complete",
        "run_dir": str(run_dir),
        "seeds_requested": args.seeds,
        "seeds_completed": completed_seeds,
        "summaries": summaries,
        "probe_size": len(probe_loader.dataset),
        "source_tree_sha256": source_tree_hash(),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    verdict = generate_report(run_dir)
    print(json.dumps({"status": "complete", "verdict": verdict, "run_dir": str(run_dir)}), flush=True)


if __name__ == "__main__":
    main()
