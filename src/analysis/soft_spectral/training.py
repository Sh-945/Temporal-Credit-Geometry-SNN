"""Causal training loop and deterministic data controls for Experiment 03."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from analysis.feedback_expansion.capture import capture_layer_signals
from analysis.feedback_expansion.core import principal_subspace_metrics
from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics
from analysis.soft_spectral.diagnostics import diagnose_model
from methods import DynamicBasisTracker, FeedbackBank, SoftSpectralFilter
from models import build_model
from training.checkpoint import load_checkpoint, save_checkpoint
from training.engine import Trainer
from training.seed import seed_everything


DIAGNOSTIC_EPOCHS = {0, 10, 25, 50, 75, 100}


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    alpha: float | None
    basis_kind: str
    epochs: int
    norm_matched: bool = False
    learning_rate_scale: float = 1.0

    @property
    def is_bptt(self) -> bool:
        return self.basis_kind == "bptt"


@dataclass
class DataBundle:
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    calibration_dataset: Dataset
    diagnostic_dataset: Dataset
    split_rows: list[dict[str, Any]]
    calibration_rows: list[dict[str, Any]]
    diagnostic_rows: list[dict[str, Any]]


def tensor_checksum(tensor: Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def model_checksum(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def feedback_checksum(bank) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(bank.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def materialize_dataset(
    dataset: Dataset,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
) -> TensorDataset:
    """Losslessly cache event frames as uint8 once for all paired variants."""

    worker_count = max(0, int(num_workers))
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": worker_count,
    }
    if worker_count:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_options)
    frames: Tensor | None = None
    labels = torch.empty(len(dataset), dtype=torch.long)
    offset = 0
    for samples, target in loader:
        rounded = samples.round()
        if not torch.equal(samples, rounded) or float(samples.min()) < 0 or float(samples.max()) > 255:
            raise ValueError("event frames are not losslessly representable as uint8")
        values = rounded.to(torch.uint8)
        if frames is None:
            frames = torch.empty((len(dataset), *values.shape[1:]), dtype=torch.uint8)
        end = offset + len(target)
        frames[offset:end].copy_(values)
        labels[offset:end].copy_(target)
        offset = end
    if frames is None or offset != len(dataset):
        raise RuntimeError("dataset materialization did not complete")
    return TensorDataset(frames, labels)


def deterministic_data_bundle(
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    split_seed: int,
    calibration_seed: int,
    validation_fraction: float = 0.10,
    calibration_size: int = 1024,
    diagnostic_size: int = 512,
) -> DataBundle:
    length = len(train_dataset)
    generator = np.random.default_rng(int(split_seed))
    permutation = generator.permutation(length)
    validation_count = max(1, int(round(length * validation_fraction)))
    validation_indices = np.sort(permutation[:validation_count])
    training_indices = np.sort(permutation[validation_count:])
    subset_generator = np.random.default_rng(int(calibration_seed))
    subset_order = subset_generator.permutation(training_indices)
    if calibration_size + diagnostic_size > len(subset_order):
        raise ValueError("training split too small for calibration and diagnostic subsets")
    calibration_indices = np.sort(subset_order[:calibration_size])
    diagnostic_indices = np.sort(
        subset_order[calibration_size : calibration_size + diagnostic_size]
    )
    training_set = set(map(int, training_indices))
    validation_set = set(map(int, validation_indices))
    if training_set.intersection(validation_set):
        raise AssertionError("training/validation split overlaps")
    if not set(map(int, calibration_indices)).issubset(training_set):
        raise AssertionError("calibration set is not inside training split")
    if set(map(int, calibration_indices)).intersection(map(int, diagnostic_indices)):
        raise AssertionError("calibration and diagnostic subsets overlap")
    split_rows = [
        {"dataset_index": int(index), "split": "validation"}
        for index in validation_indices
    ] + [
        {"dataset_index": int(index), "split": "training"}
        for index in training_indices
    ]
    calibration_rows = [
        {"calibration_position": position, "dataset_index": int(index)}
        for position, index in enumerate(calibration_indices)
    ]
    diagnostic_rows = [
        {"diagnostic_position": position, "dataset_index": int(index)}
        for position, index in enumerate(diagnostic_indices)
    ]
    return DataBundle(
        train_dataset=Subset(train_dataset, training_indices.tolist()),
        validation_dataset=Subset(train_dataset, validation_indices.tolist()),
        test_dataset=test_dataset,
        calibration_dataset=Subset(train_dataset, calibration_indices.tolist()),
        diagnostic_dataset=Subset(train_dataset, diagnostic_indices.tolist()),
        split_rows=split_rows,
        calibration_rows=calibration_rows,
        diagnostic_rows=diagnostic_rows,
    )


def make_loaders(
    bundle: DataBundle,
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, DataLoader]:
    batch_size = int(config["data"]["batch_size"])
    pin_memory = bool(config["data"].get("pin_memory", False))
    generator = torch.Generator().manual_seed(int(seed))
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": pin_memory,
    }
    return {
        "train": DataLoader(
            bundle.train_dataset,
            shuffle=True,
            generator=generator,
            **common,
        ),
        "validation": DataLoader(bundle.validation_dataset, shuffle=False, **common),
        "test": DataLoader(bundle.test_dataset, shuffle=False, **common),
        "calibration": DataLoader(bundle.calibration_dataset, shuffle=False, **common),
        "diagnostic": DataLoader(bundle.diagnostic_dataset, shuffle=False, **common),
    }


def save_paired_initial_state(
    *,
    config: dict[str, Any],
    seed: int,
    path: Path,
) -> dict[str, Any]:
    seed_everything(seed)
    model = build_model(config)
    bank = FeedbackBank(model, config)
    state = {
        "seed": seed,
        "model_state": copy.deepcopy(model.state_dict()),
        "feedback_state": copy.deepcopy(bank.state_dict()),
        "model_checksum": model_checksum(model),
        "feedback_checksum": feedback_checksum(bank),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return {key: value for key, value in state.items() if not key.endswith("_state")}


def load_paired_model(
    *,
    config: dict[str, Any],
    initial_state_path: Path,
    device: torch.device,
):
    state = torch.load(initial_state_path, map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(state["model_state"])
    bank = FeedbackBank(model, config)
    if str(config["method"]["name"]).lower() != "bptt":
        bank.load_state_dict(state["feedback_state"])
    model.to(device)
    bank.to(device)
    if model_checksum(model) != state["model_checksum"]:
        raise AssertionError("paired forward initialization checksum mismatch")
    if str(config["method"]["name"]).lower() != "bptt" and feedback_checksum(bank) != state["feedback_checksum"]:
        raise AssertionError("paired feedback initialization checksum mismatch")
    return model, bank, state


def baseline_parity(
    *,
    config: dict[str, Any],
    initial_state_path: Path,
    batch: tuple[Tensor, Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Prove alpha=1 is identical at delta, dW, and parameter-update levels."""

    dense_model, dense_bank, _ = load_paired_model(
        config=config, initial_state_path=initial_state_path, device=device
    )
    filtered_model, filtered_bank, _ = load_paired_model(
        config=config, initial_state_path=initial_state_path, device=device
    )
    dimensions = [int(spec["dimension"]) for spec in dense_model.hidden_specs]
    alpha_one = SoftSpectralFilter(dimensions, alpha=1.0)
    for layer_index, dimension in enumerate(dimensions):
        rank = max(1, dimension // 3)
        alpha_one.set_basis(
            layer_index, torch.eye(dimension)[:, :rank], source_epoch=0
        )
    dense_trainer = Trainer(dense_model, dense_bank, config, device)
    filtered_trainer = Trainer(
        filtered_model,
        filtered_bank,
        config,
        device,
        spectral_filter=alpha_one,
    )
    raw_samples, raw_labels = batch
    samples = raw_samples.transpose(0, 1).contiguous().to(
        device=device, dtype=torch.float32
    )
    labels = raw_labels.to(device)

    def captures(model, bank):
        with torch.no_grad():
            output, hidden_inputs, _ = model.forward_with_cache(
                samples, detach_temporal=True
            )
            desired = F.one_hot(labels, num_classes=model.num_classes).to(output.dtype)
            error = output.mean(0) - desired
        return [
            capture_layer_signals(
                model,
                bank,
                index,
                layer_input,
                error,
                detach_temporal=True,
            ).delta
            for index, layer_input in enumerate(hidden_inputs)
        ]

    dense_delta = captures(dense_model, dense_bank)
    filtered_delta = captures(filtered_model, filtered_bank)
    dense_values = dense_trainer._local_batch(samples, labels)
    filtered_values = filtered_trainer._local_batch(samples, labels)
    dense_values["total"].backward()
    filtered_values["total"].backward()
    dense_gradients = [
        layer.linear.weight.grad.detach().clone() for layer in dense_model.hidden_layers
    ]
    filtered_gradients = [
        layer.linear.weight.grad.detach().clone() for layer in filtered_model.hidden_layers
    ]
    clip = float(config["training"].get("gradient_clip", 0.0))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(dense_model.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(filtered_model.parameters(), clip)
    dense_trainer.optimizer.step()
    filtered_trainer.optimizer.step()
    rows: list[dict[str, Any]] = []
    for layer_index in range(len(dimensions)):
        delta_difference = dense_delta[layer_index] - filtered_delta[layer_index]
        gradient_difference = dense_gradients[layer_index] - filtered_gradients[layer_index]
        dense_parameter = dense_model.hidden_layers[layer_index].linear.weight.detach()
        filtered_parameter = filtered_model.hidden_layers[layer_index].linear.weight.detach()
        parameter_difference = dense_parameter - filtered_parameter
        rows.append(
            {
                "layer": f"hidden_{layer_index + 1}",
                "alpha": 1.0,
                "hook_installed": False,
                "delta_max_abs": float(delta_difference.abs().max()),
                "delta_relative_error": float(
                    torch.linalg.vector_norm(delta_difference)
                    / (torch.linalg.vector_norm(dense_delta[layer_index]) + 1e-30)
                ),
                "weight_gradient_max_abs": float(gradient_difference.abs().max()),
                "weight_gradient_relative_error": float(
                    torch.linalg.vector_norm(gradient_difference)
                    / (torch.linalg.vector_norm(dense_gradients[layer_index]) + 1e-30)
                ),
                "parameter_update_max_abs": float(parameter_difference.abs().max()),
                "parameter_update_relative_error": float(
                    torch.linalg.vector_norm(parameter_difference)
                    / (torch.linalg.vector_norm(dense_parameter) + 1e-30)
                ),
                "tolerance": 1e-7,
                "passed": bool(
                    torch.equal(dense_delta[layer_index], filtered_delta[layer_index])
                    and torch.equal(dense_gradients[layer_index], filtered_gradients[layer_index])
                    and torch.equal(dense_parameter, filtered_parameter)
                ),
            }
        )
    return rows


def select_pilot_alpha(training_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered validation-only selection over alpha 0.25/0.50/0.75."""

    candidates = []
    for alpha, method in (
        (0.25, "pilot_soft_025"),
        (0.50, "pilot_soft_050"),
        (0.75, "pilot_soft_075"),
    ):
        rows = [
            row
            for row in training_rows
            if row["method"] == method and 41 <= int(row["epoch"]) <= 50
        ]
        if len(rows) != 10 or any(bool(row["unstable"]) for row in rows):
            continue
        candidates.append(
            {
                "alpha": alpha,
                "method": method,
                "mean_validation_accuracy_41_50": float(
                    np.mean([row["validation_accuracy"] for row in rows])
                ),
                "mean_validation_loss_41_50": float(
                    np.mean([row["validation_loss"] for row in rows])
                ),
            }
        )
    if not candidates:
        raise RuntimeError("no stable soft-alpha pilot completed epochs 41-50")
    best_accuracy = max(row["mean_validation_accuracy_41_50"] for row in candidates)
    accuracy_tied = [
        row
        for row in candidates
        if best_accuracy - row["mean_validation_accuracy_41_50"] < 0.0005
    ]
    best_loss = min(row["mean_validation_loss_41_50"] for row in accuracy_tied)
    loss_tied = [
        row
        for row in accuracy_tied
        if abs(row["mean_validation_loss_41_50"] - best_loss) <= 1e-12
    ]
    selected = max(loss_tied, key=lambda row: row["alpha"])
    return {
        "criterion": "mean validation accuracy epochs 41-50; <0.05pp tie by mean validation loss; exact tie chooses larger alpha",
        "test_used_for_selection": False,
        "candidates": candidates,
        "selected_alpha": selected["alpha"],
        "selected_method": selected["method"],
    }


def rank_schedule_from_rows(
    rows: list[dict[str, Any]], method: str, seed: int
) -> dict[int, list[int]]:
    selected = [
        row for row in rows if row["method"] == method and int(row["seed"]) == seed
    ]
    result: dict[int, list[int]] = {}
    for epoch in sorted({int(row["source_epoch"]) for row in selected}):
        epoch_rows = sorted(
            (row for row in selected if int(row["source_epoch"]) == epoch),
            key=lambda row: int(str(row["layer"]).split("_")[-1]),
        )
        result[epoch] = [int(row["k"]) for row in epoch_rows]
    return result


def _calibration_covariances(
    model,
    bank,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[OnlineCovariance], float]:
    covariances = [
        OnlineCovariance(int(spec["dimension"])) for spec in model.hidden_specs
    ]
    start = time.perf_counter()
    model.eval()
    bank.eval()
    for samples, labels in loader:
        samples = samples.transpose(0, 1).contiguous().to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        labels = labels.to(device)
        with torch.no_grad():
            output, hidden_inputs, _ = model.forward_with_cache(
                samples, detach_temporal=True
            )
            desired = F.one_hot(labels, num_classes=model.num_classes).to(output.dtype)
            error = output.mean(dim=0) - desired
        for layer_index, layer_input in enumerate(hidden_inputs):
            capture = capture_layer_signals(
                model,
                bank,
                layer_index,
                layer_input,
                error,
                detach_temporal=True,
            )
            covariances[layer_index].update(capture.delta.mean(dim=0))
    return covariances, time.perf_counter() - start


def refresh_bases(
    *,
    model,
    bank,
    spectral_filter: SoftSpectralFilter,
    tracker: DynamicBasisTracker,
    calibration_loader: DataLoader,
    device: torch.device,
    seed: int,
    completed_epoch: int,
    method_key: str,
    basis_kind: str,
    forced_ranks: list[int] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    covariances, replay_seconds = _calibration_covariances(
        model, bank, calibration_loader, device
    )
    rank_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []
    pca_start = time.perf_counter()
    for layer_index, covariance in enumerate(covariances):
        learned, eigenvalues, calibration_r95 = tracker.learned_basis(
            covariance.m2, device=device
        )
        previous = spectral_filter.bases[layer_index]
        random_seed = None
        if basis_kind == "random":
            if forced_ranks is None:
                raise ValueError("random control requires learned-soft forced ranks")
            rank = int(forced_ranks[layer_index])
            random_seed = int(
                (seed * 1_000_003 + completed_epoch * 10_009 + (layer_index + 1) * 1_009)
                % (2**63 - 1)
            )
            basis = tracker.random_basis(layer_index, rank, random_seed)
            rank_source = "forced_from_paired_soft"
        else:
            basis = learned
            rank = int(calibration_r95)
            rank_source = "dynamic_calibration_r95"
        if previous is not None:
            for source, requested in (
                ("fixed_32", 32),
                ("fixed_64", 64),
                ("fixed_128", 128),
                ("dynamic_r95", min(previous.shape[1], basis.shape[1])),
            ):
                values = principal_subspace_metrics(previous, basis, requested)
                rotation_rows.append(
                    {
                        "method": method_key,
                        "seed": seed,
                        "previous_source_epoch": spectral_filter.basis_source_epochs[layer_index],
                        "current_source_epoch": completed_epoch,
                        "layer": f"hidden_{layer_index + 1}",
                        "k_source": source,
                        **values,
                    }
                )
        spectral_filter.set_basis(layer_index, basis, completed_epoch)
        metrics, _s, _e, _c = spectrum_statistics(
            covariance,
            algebraic_max_dim=covariance.ambient_dim,
            device=device,
        )
        rank_rows.append(
            {
                "method": method_key,
                "seed": seed,
                "source_epoch": completed_epoch,
                "used_from_epoch": completed_epoch + 1,
                "layer": f"hidden_{layer_index + 1}",
                "k": rank,
                "k_over_D": rank / float(covariance.ambient_dim),
                "rank_source": rank_source,
                "calibration_r95": calibration_r95,
                "energy_threshold": tracker.energy_threshold,
                "random_seed": random_seed,
                "calibration_observations": covariance.count,
                "calibration_entropy_rank": metrics["entropy_rank"],
                "calibration_participation_ratio": metrics["participation_ratio"],
            }
        )
    basis_seconds = time.perf_counter() - pca_start
    return rank_rows, rotation_rows, {
        "calibration_replay_seconds": replay_seconds,
        "basis_update_seconds": basis_seconds,
    }


def _save_filter(path: Path, spectral_filter: SoftSpectralFilter) -> None:
    torch.save(spectral_filter.state_dict(), path)


def _load_filter(path: Path, spectral_filter: SoftSpectralFilter) -> None:
    spectral_filter.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)
    )


def run_variant(
    *,
    base_config: dict[str, Any],
    variant: Variant,
    seed: int,
    initial_state_path: Path,
    bundle: DataBundle,
    run_dir: Path,
    device: torch.device,
    forced_rank_schedule: dict[int, list[int]] | None = None,
    run_test: bool,
) -> dict[str, Any]:
    """Train one paired variant with strictly lagged epoch-end bases."""

    config = copy.deepcopy(base_config)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = f"experiment03_{variant.key}_seed{seed}"
    config["method"]["name"] = "bptt" if variant.is_bptt else "sdfa"
    config["training"]["epochs"] = variant.epochs
    config["training"]["learning_rate"] = float(
        base_config["training"]["learning_rate"]
    ) * variant.learning_rate_scale
    output_dir = run_dir / "checkpoints" / variant.key / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config["training"]["output_dir"] = str(output_dir)
    model, bank, initial = load_paired_model(
        config=config,
        initial_state_path=initial_state_path,
        device=device,
    )
    dimensions = [int(spec["dimension"]) for spec in model.hidden_specs]
    spectral_filter = None
    analysis_filter = None
    tracker = None
    if not variant.is_bptt:
        analysis_filter = SoftSpectralFilter(
            dimensions,
            alpha=float(variant.alpha),
            norm_matched=variant.norm_matched,
        )
        tracker = DynamicBasisTracker(dimensions, energy_threshold=0.95)
        spectral_filter = None if variant.basis_kind == "dense" else analysis_filter
    trainer = Trainer(
        model,
        bank,
        config,
        device,
        spectral_filter=spectral_filter,
    )
    loaders = make_loaders(bundle, config, seed=seed)
    save_checkpoint(
        output_dir / "init.pt",
        epoch=-1,
        model=model,
        feedback_bank=bank,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        config=config,
        metrics={"stage": "paired_initialization"},
    )
    if analysis_filter is not None:
        _save_filter(output_dir / "init_filter.pt", analysis_filter)

    training_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    best_validation = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    unstable = False
    unstable_reason = None
    low_validation_streak = 0
    peak_memory = 0

    def append_diagnostic(epoch: int) -> None:
        values = diagnose_model(
            model=model,
            feedback_bank=bank,
            spectral_filter=analysis_filter,
            method_key=variant.key,
            seed=seed,
            epoch=epoch,
            loader=loaders["diagnostic"],
            device=device,
            is_bptt=variant.is_bptt,
        )
        geometry_rows.extend(values["spectral_geometry"])
        alignment_rows.extend(values["bp_alignment"])
        norm_rows.extend(values["gradient_norms"])
        gate_rows.extend(values["gate_dynamics"])

    append_diagnostic(0)
    wall_start = time.perf_counter()
    for epoch_index in range(variant.epochs):
        completed_epoch = epoch_index + 1
        basis_sources_used = (
            [None for _ in dimensions]
            if analysis_filter is None
            else list(analysis_filter.basis_source_epochs)
        )
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            train_metrics = trainer.train_epoch(loaders["train"])
        except FloatingPointError as error:
            unstable = True
            unstable_reason = f"nonfinite_loss: {error}"
            break
        validation_metrics = trainer.evaluate(loaders["validation"])
        epoch_seconds = time.perf_counter() - epoch_start
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
        if (
            not np.isfinite(train_metrics["loss"])
            or train_metrics["loss"] > 1_000.0
        ):
            unstable = True
            unstable_reason = "loss_explosion_threshold_1000"
        if validation_metrics["accuracy"] < 0.90:
            low_validation_streak += 1
        else:
            low_validation_streak = 0
        if low_validation_streak >= 5:
            unstable = True
            unstable_reason = "validation_accuracy_below_90pct_for_5_epochs"
        learning_rate = trainer.optimizer.param_groups[0]["lr"]
        training_rows.append(
            {
                "stage": "pilot" if variant.epochs == 50 else "full",
                "method": variant.key,
                "method_label": variant.label,
                "seed": seed,
                "epoch": completed_epoch,
                "alpha": variant.alpha,
                "norm_matched": variant.norm_matched,
                "learning_rate": learning_rate,
                "basis_source_epochs_used": json.dumps(basis_sources_used),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "unstable": unstable,
                "unstable_reason": unstable_reason,
            }
        )
        for row in trainer.last_epoch_spectral_rows:
            norm_rows.append(
                {
                    "record_type": "training_batch",
                    "method": variant.key,
                    "seed": seed,
                    "epoch": completed_epoch,
                    "layer": f"hidden_{int(row['layer_index']) + 1}",
                    **row,
                }
            )
        runtime = {
            "method": variant.key,
            "seed": seed,
            "epoch": completed_epoch,
            "training_and_validation_seconds": epoch_seconds,
            "calibration_replay_seconds": 0.0,
            "basis_update_seconds": 0.0,
            "gpu_peak_memory_bytes": peak_memory,
        }
        if completed_epoch in DIAGNOSTIC_EPOCHS:
            append_diagnostic(completed_epoch)
        if (
            validation_metrics["accuracy"] > best_validation
            or (
                validation_metrics["accuracy"] == best_validation
                and validation_metrics["loss"] <= best_validation_loss
            )
        ):
            best_validation = validation_metrics["accuracy"]
            best_validation_loss = validation_metrics["loss"]
            best_epoch = completed_epoch
            save_checkpoint(
                output_dir / "best_validation.pt",
                epoch=epoch_index,
                model=model,
                feedback_bank=bank,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                config=config,
                metrics=training_rows[-1],
            )
            if analysis_filter is not None:
                _save_filter(output_dir / "best_validation_used_filter.pt", analysis_filter)
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        if not variant.is_bptt:
            assert analysis_filter is not None and tracker is not None
            forced = None
            if variant.basis_kind == "random":
                if forced_rank_schedule is None or completed_epoch not in forced_rank_schedule:
                    raise KeyError(
                        f"missing paired soft ranks for random epoch {completed_epoch}"
                    )
                forced = forced_rank_schedule[completed_epoch]
            new_ranks, rotations, costs = refresh_bases(
                model=model,
                bank=bank,
                spectral_filter=analysis_filter,
                tracker=tracker,
                calibration_loader=loaders["calibration"],
                device=device,
                seed=seed,
                completed_epoch=completed_epoch,
                method_key=variant.key,
                basis_kind=variant.basis_kind,
                forced_ranks=forced,
            )
            rank_rows.extend(new_ranks)
            rotation_rows.extend(rotations)
            runtime.update(costs)
        runtime_rows.append(runtime)
        save_checkpoint(
            output_dir / "last.pt",
            epoch=epoch_index,
            model=model,
            feedback_bank=bank,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            config=config,
            metrics=training_rows[-1],
        )
        if analysis_filter is not None:
            _save_filter(output_dir / "last_filter.pt", analysis_filter)
        if completed_epoch in DIAGNOSTIC_EPOCHS:
            save_checkpoint(
                output_dir / f"epoch_{completed_epoch:03d}.pt",
                epoch=epoch_index,
                model=model,
                feedback_bank=bank,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                config=config,
                metrics=training_rows[-1],
            )
            if analysis_filter is not None:
                _save_filter(
                    output_dir / f"epoch_{completed_epoch:03d}_next_filter.pt",
                    analysis_filter,
                )
        if unstable:
            break

    wall_seconds = time.perf_counter() - wall_start
    completed_epochs = len(training_rows)
    final_test = best_validation_test = None
    if run_test and not unstable:
        final_test = trainer.evaluate(loaders["test"])
        best_model, best_bank, _ = load_paired_model(
            config=config,
            initial_state_path=initial_state_path,
            device=device,
        )
        load_checkpoint(
            output_dir / "best_validation.pt",
            model=best_model,
            feedback_bank=best_bank,
            map_location=device,
        )
        best_trainer = Trainer(best_model, best_bank, config, device)
        best_validation_test = best_trainer.evaluate(loaders["test"])
        del best_model, best_bank, best_trainer

    if best_epoch not in DIAGNOSTIC_EPOCHS and best_epoch > 0 and not unstable:
        load_checkpoint(
            output_dir / "best_validation.pt",
            model=model,
            feedback_bank=bank,
            map_location=device,
        )
        if analysis_filter is not None:
            _load_filter(output_dir / "best_validation_used_filter.pt", analysis_filter)
        append_diagnostic(best_epoch)

    validation_values = [row["validation_accuracy"] for row in training_rows]
    convergence_epoch = None
    if validation_values:
        threshold = 0.99 * max(validation_values)
        convergence_epoch = next(
            row["epoch"]
            for row in training_rows
            if row["validation_accuracy"] >= threshold
        )
    summary = {
        "method": variant.key,
        "method_label": variant.label,
        "seed": seed,
        "alpha": variant.alpha,
        "norm_matched": variant.norm_matched,
        "learning_rate_scale": variant.learning_rate_scale,
        "epochs_requested": variant.epochs,
        "epochs_completed": completed_epochs,
        "unstable": unstable,
        "unstable_reason": unstable_reason,
        "best_validation_accuracy": best_validation,
        "best_validation_loss": best_validation_loss,
        "best_validation_epoch": best_epoch,
        "convergence_epoch_99pct_own_best": convergence_epoch,
        "final_test": final_test,
        "best_validation_test": best_validation_test,
        "wall_seconds": wall_seconds,
        "gpu_peak_memory_bytes": peak_memory,
        "initial_model_checksum": initial["model_checksum"],
        "initial_feedback_checksum": (
            None if variant.is_bptt else initial["feedback_checksum"]
        ),
    }
    (output_dir / "variant_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "summary": summary,
        "training_metrics": training_rows,
        "rank_schedule": rank_rows,
        "basis_rotation": rotation_rows,
        "runtime_metrics": runtime_rows,
        "spectral_geometry": geometry_rows,
        "bp_alignment": alignment_rows,
        "gradient_norms": norm_rows,
        "gate_dynamics": gate_rows,
    }
