"""Checkpoint replay and sufficient-statistic collection for Experiment 01B."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from analysis.feedback_expansion.capture import capture_layer_signals, shuffled_delta
from analysis.feedback_expansion.core import (
    broadening_fields,
    effective_neuron_count,
    mean_pair_cosine,
    merge_covariances,
    pca_from_covariance,
    principal_subspace_metrics,
    temporal_coherence,
)
from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics
from methods import FeedbackBank
from models import build_model
from training.checkpoint import load_checkpoint


PRIMARY_SIGNALS = ("pre_gate_q", "gate", "post_gate_delta")
TEMPORAL_MODES = ("aggregated", "timestep")


@dataclass
class SavedBasis:
    seed: int
    epoch: int
    layer: str
    signal_type: str
    temporal_mode: str
    path: Path
    mean: Tensor
    basis: Tensor
    metrics: dict[str, Any]


@dataclass
class CheckpointDiagnostics:
    bases: dict[tuple[str, str, str], SavedBasis] = field(default_factory=dict)
    delta_eval: dict[str, OnlineCovariance] = field(default_factory=dict)
    shuffled_eval: dict[str, OnlineCovariance] = field(default_factory=dict)
    shuffled_mean: dict[str, Tensor] = field(default_factory=dict)
    spectral_rows: list[dict[str, Any]] = field(default_factory=list)
    gate_rows: list[dict[str, Any]] = field(default_factory=list)
    class_rows: list[dict[str, Any]] = field(default_factory=list)
    temporal_rows: list[dict[str, Any]] = field(default_factory=list)
    coherence_rows: list[dict[str, Any]] = field(default_factory=list)
    control_rows: list[dict[str, Any]] = field(default_factory=list)
    sanity_rows: list[dict[str, Any]] = field(default_factory=list)


def load_checkpoint_model(
    config: dict[str, Any], checkpoint_path: Path, device: torch.device
):
    model = build_model(config).to(device)
    feedback_bank = FeedbackBank(model, config).to(device)
    state = load_checkpoint(
        checkpoint_path,
        model=model,
        feedback_bank=feedback_bank,
        map_location=device,
    )
    model.eval()
    feedback_bank.eval()
    return model, feedback_bank, state


def _prepare(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    samples, labels = batch
    return (
        samples.transpose(0, 1).contiguous().to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        labels.to(device=device, non_blocking=True),
    )


def _forward_inputs(model, samples: Tensor, labels: Tensor, pointwise: bool):
    with torch.no_grad():
        output, hidden_inputs, _readout = model.forward_with_cache(
            samples, detach_temporal=pointwise
        )
        desired = F.one_hot(labels, num_classes=model.num_classes).to(output.dtype)
        error = output.mean(dim=0) - desired
    return hidden_inputs, error.detach()


def _covariance_groups(model) -> dict[tuple[str, str, str], OnlineCovariance]:
    groups: dict[tuple[str, str, str], OnlineCovariance] = {}
    for index, spec in enumerate(model.hidden_specs):
        layer = f"hidden_{index + 1}"
        dimension = int(spec["dimension"])
        for signal in PRIMARY_SIGNALS:
            for mode in TEMPORAL_MODES:
                groups[(layer, signal, mode)] = OnlineCovariance(dimension)
    return groups


def _update_groups(
    groups: dict[tuple[str, str, str], OnlineCovariance],
    layer: str,
    q: Tensor,
    gate: Tensor,
    delta: Tensor,
) -> None:
    groups[(layer, "pre_gate_q", "aggregated")].update(q)
    # q is broadcast unchanged over timesteps in the production local loss.
    groups[(layer, "pre_gate_q", "timestep")].update(
        q.unsqueeze(0).expand(gate.shape[0], -1, -1)
    )
    groups[(layer, "gate", "aggregated")].update(gate.mean(dim=0))
    groups[(layer, "gate", "timestep")].update(gate)
    groups[(layer, "post_gate_delta", "aggregated")].update(delta.mean(dim=0))
    groups[(layer, "post_gate_delta", "timestep")].update(delta)


def _pca_lowrank_basis(matrix: Tensor, rank: int, device: torch.device) -> Tensor:
    x = matrix.to(device=device, dtype=torch.float32)
    target = min(x.shape[0] - 1, x.shape[1], int(rank))
    if target < 1:
        return torch.empty(x.shape[1], 0)
    oversampled = min(x.shape[0] - 1, x.shape[1], max(target + 32, 2 * target))
    _u, _s, v = torch.pca_lowrank(x, q=oversampled, center=True, niter=5)
    return v[:, :target].detach().cpu().to(torch.float32)


def _class_details(
    *,
    seed: int,
    epoch: int,
    layer: str,
    fit_by_class: dict[int, OnlineCovariance],
    eval_by_class: dict[int, OnlineCovariance],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[int, Tensor], dict[int, Tensor]]:
    rows: list[dict[str, Any]] = []
    bases16: dict[int, Tensor] = {}
    bases64: dict[int, Tensor] = {}
    for class_id in sorted(fit_by_class):
        fit = fit_by_class[class_id]
        evaluation = eval_by_class[class_id]
        summary = pca_from_covariance(fit, device, max_basis=64)
        bases16[class_id] = summary.basis[:, :16]
        bases64[class_id] = summary.basis[:, :64]
        for split, covariance in (("basis_fit", fit), ("basis_eval", evaluation)):
            metrics, _singular, _energy, _cumulative = spectrum_statistics(
                covariance,
                algebraic_max_dim=covariance.ambient_dim,
                device=device,
            )
            rows.append(
                {
                    "record_type": "class_effective_rank",
                    "seed": seed,
                    "epoch": epoch,
                    "layer": layer,
                    "class_a": class_id,
                    "class_b": None,
                    "split": split,
                    "k": None,
                    "overlap": None,
                    **metrics,
                }
            )
    for k in (16, 32, 64):
        source = bases16 if k == 16 else bases64
        for class_a in sorted(source):
            for class_b in sorted(source):
                actual = min(k, source[class_a].shape[1], source[class_b].shape[1])
                metrics = principal_subspace_metrics(
                    source[class_a], source[class_b], actual
                )
                rows.append(
                    {
                        "record_type": "class_subspace_overlap",
                        "seed": seed,
                        "epoch": epoch,
                        "layer": layer,
                        "class_a": class_a,
                        "class_b": class_b,
                        "split": "basis_fit",
                        **metrics,
                    }
                )
    return rows, bases16, bases64


def _temporal_details(
    *,
    seed: int,
    epoch: int,
    layer: str,
    fit_signal: Tensor,
    device: torch.device,
    sanity: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Randomized per-timestep bases with one exact-vs-randomized check."""

    time_steps = fit_signal.shape[1]
    bases = [
        _pca_lowrank_basis(fit_signal[:, step], 32, device)
        for step in range(time_steps)
    ]
    rows: list[dict[str, Any]] = []
    for step_a in range(time_steps):
        for step_b in range(time_steps):
            metrics = principal_subspace_metrics(bases[step_a], bases[step_b], 32)
            rows.append(
                {
                    "record_type": "timestep_subspace_overlap",
                    "seed": seed,
                    "epoch": epoch,
                    "layer": layer,
                    "timestep_a": step_a,
                    "timestep_b": step_b,
                    "method": "torch.pca_lowrank_target32_q64_niter5",
                    **metrics,
                }
            )
    sanity_rows: list[dict[str, Any]] = []
    if sanity:
        exact_cov = OnlineCovariance(fit_signal.shape[2])
        exact_cov.update(fit_signal[:, 0])
        exact = pca_from_covariance(exact_cov, device, max_basis=32)
        comparison = principal_subspace_metrics(exact.basis, bases[0], 32)
        sanity_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "layer": layer,
                "timestep": 0,
                "exact_method": "feature_covariance_eigh",
                "approximate_method": "torch.pca_lowrank_target32_q64_niter5",
                **comparison,
            }
        )
    return rows, sanity_rows


def diagnose_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_kind: str,
    epoch: int,
    seed: int,
    fit_loader: Iterable[tuple[Tensor, Tensor]],
    eval_loader: Iterable[tuple[Tensor, Tensor]],
    fit_shuffled_labels: Tensor,
    eval_shuffled_labels: Tensor,
    device: torch.device,
    bases_dir: Path,
    detailed: bool,
    sanity: bool,
) -> CheckpointDiagnostics:
    """Replay one immutable checkpoint and persist only feature-space bases."""

    model, feedback_bank, _state = load_checkpoint_model(config, checkpoint_path, device)
    if str(config["method"].get("temporal_mode", "pointwise")) != "pointwise":
        raise ValueError("Experiment 01B exact gate audit currently expects pointwise sDFA")
    pointwise = True
    fit_groups = _covariance_groups(model)
    eval_groups = _covariance_groups(model)
    dimensions = {
        f"hidden_{i + 1}": int(spec["dimension"])
        for i, spec in enumerate(model.hidden_specs)
    }
    classes = int(model.num_classes)
    fit_class: dict[str, dict[int, OnlineCovariance]] = {
        layer: {c: OnlineCovariance(dim) for c in range(classes)}
        for layer, dim in dimensions.items()
    }
    eval_class: dict[str, dict[int, OnlineCovariance]] = {
        layer: {c: OnlineCovariance(dim) for c in range(classes)}
        for layer, dim in dimensions.items()
    }
    temporal_fit: dict[str, list[Tensor]] = defaultdict(list)
    fit_parity: dict[str, list[tuple[float, float]]] = defaultdict(list)
    shuffled_fit = {layer: OnlineCovariance(dim) for layer, dim in dimensions.items()}

    # Fit comes first so class means are available for leakage-safe eval residuals.
    fit_shuffle_offset = 0
    for batch in fit_loader:
        samples, labels = _prepare(batch, device)
        batch_size = labels.numel()
        fit_shuffled = fit_shuffled_labels[
            fit_shuffle_offset : fit_shuffle_offset + batch_size
        ].to(device)
        fit_shuffle_offset += batch_size
        hidden_inputs, error = _forward_inputs(model, samples, labels, pointwise)
        shuffled_error = None
        if detailed:
            with torch.no_grad():
                output = model(samples, detach_temporal=pointwise)
                desired_shuffled = F.one_hot(
                    fit_shuffled, num_classes=model.num_classes
                ).to(output.dtype)
                shuffled_error = output.mean(dim=0) - desired_shuffled
        for index, layer_input in enumerate(hidden_inputs):
            layer = f"hidden_{index + 1}"
            capture = capture_layer_signals(
                model,
                feedback_bank,
                index,
                layer_input,
                error,
                detach_temporal=pointwise,
            )
            _update_groups(fit_groups, layer, capture.q, capture.gate, capture.delta)
            fit_parity[layer].append((capture.parity_max_abs, capture.parity_relative))
            if detailed:
                aggregate = capture.delta.mean(dim=0)
                for class_id in range(classes):
                    mask = labels == class_id
                    if bool(mask.any()):
                        fit_class[layer][class_id].update(aggregate[mask])
                temporal_fit[layer].append(capture.delta.permute(1, 0, 2).cpu())
                assert shuffled_error is not None
                fit_shuffled_q = feedback_bank.project(index, shuffled_error).detach()
                shuffled_fit[layer].update(
                    shuffled_delta(capture, fit_shuffled_q).mean(dim=0)
                )

    if fit_shuffle_offset != len(fit_shuffled_labels):
        raise AssertionError("shuffled labels did not align with fit loader")

    class_means = {
        layer: {c: value.mean.clone() for c, value in groups.items()}
        for layer, groups in fit_class.items()
    }
    global_means = {
        layer: merge_covariances(list(groups.values())).mean
        for layer, groups in fit_class.items()
    }
    within_eval = {layer: OnlineCovariance(dim) for layer, dim in dimensions.items()}
    between_eval = {layer: OnlineCovariance(dim) for layer, dim in dimensions.items()}
    shuffled_eval = {layer: OnlineCovariance(dim) for layer, dim in dimensions.items()}
    gate_vectors: dict[str, list[Tensor]] = defaultdict(list)
    eval_labels: list[Tensor] = []
    gate_sumabs = {layer: torch.zeros(dim, dtype=torch.float64) for layer, dim in dimensions.items()}
    gate_sumsq = defaultdict(float)
    gate_active = defaultdict(int)
    gate_count = defaultdict(int)
    gate_coherence: dict[str, list[Tensor]] = defaultdict(list)
    gate_temporal_cosine: dict[str, list[Tensor]] = defaultdict(list)
    delta_coherence: dict[str, list[Tensor]] = defaultdict(list)
    delta_temporal_cosine: dict[str, list[Tensor]] = defaultdict(list)
    eval_parity: dict[str, list[tuple[float, float]]] = defaultdict(list)
    shuffle_offset = 0

    for batch in eval_loader:
        samples, labels = _prepare(batch, device)
        batch_size = labels.numel()
        shuffled_labels = eval_shuffled_labels[
            shuffle_offset : shuffle_offset + batch_size
        ].to(device)
        shuffle_offset += batch_size
        hidden_inputs, error = _forward_inputs(model, samples, labels, pointwise)
        with torch.no_grad():
            desired_shuffled = F.one_hot(
                shuffled_labels, num_classes=model.num_classes
            ).to(samples.dtype)
            output = model(samples, detach_temporal=pointwise)
            shuffled_error = output.mean(dim=0) - desired_shuffled
        eval_labels.append(labels.cpu())
        for index, layer_input in enumerate(hidden_inputs):
            layer = f"hidden_{index + 1}"
            capture = capture_layer_signals(
                model,
                feedback_bank,
                index,
                layer_input,
                error,
                detach_temporal=pointwise,
            )
            _update_groups(eval_groups, layer, capture.q, capture.gate, capture.delta)
            eval_parity[layer].append((capture.parity_max_abs, capture.parity_relative))
            gate = capture.gate
            aggregate_gate = gate.mean(dim=0)
            gate_vectors[layer].append(aggregate_gate.cpu())
            gate_sumabs[layer] += gate.abs().sum(dim=(0, 1)).cpu().to(torch.float64)
            gate_sumsq[layer] += float(gate.square().sum())
            # Fast-sigmoid surrogate derivative has a documented maximum of 1.
            gate_active[layer] += int((gate > 0.1).sum())
            gate_count[layer] += gate.numel()
            gc, gt = temporal_coherence(gate)
            dc, dt = temporal_coherence(capture.delta)
            gate_coherence[layer].append(gc)
            gate_temporal_cosine[layer].append(gt)
            delta_coherence[layer].append(dc)
            delta_temporal_cosine[layer].append(dt)
            if detailed:
                aggregate = capture.delta.mean(dim=0)
                for class_id in range(classes):
                    mask = labels == class_id
                    if bool(mask.any()):
                        eval_class[layer][class_id].update(aggregate[mask])
                means = torch.stack(
                    [class_means[layer][int(label)].to(torch.float32) for label in labels.cpu()]
                ).to(device)
                global_mean = global_means[layer].to(device=device, dtype=torch.float32)
                within_eval[layer].update(aggregate - means)
                between_eval[layer].update(means - global_mean)
                shuffled_q = feedback_bank.project(index, shuffled_error).detach()
                shuffled = shuffled_delta(capture, shuffled_q).mean(dim=0)
                shuffled_eval[layer].update(shuffled)

    if shuffle_offset != len(eval_shuffled_labels):
        raise AssertionError("shuffled labels did not align with evaluation loader")

    result = CheckpointDiagnostics()
    bases_dir.mkdir(parents=True, exist_ok=True)
    for key in sorted(fit_groups):
        layer, signal, mode = key
        algebraic_max = classes if signal == "pre_gate_q" else dimensions[layer]
        fit_summary = pca_from_covariance(
            fit_groups[key], device, max_basis=256, algebraic_max_dim=algebraic_max
        )
        eval_summary = pca_from_covariance(
            eval_groups[key], device, max_basis=1, algebraic_max_dim=algebraic_max
        )
        filename = (
            f"seed{seed}_epoch{epoch:03d}_{layer}_{signal}_{mode}.pt"
        )
        path = bases_dir / filename
        payload = {
                "seed": seed,
                "epoch": epoch,
                "checkpoint_kind": checkpoint_kind,
                "layer": layer,
                "signal_type": signal,
                "temporal_mode": mode,
                "analysis_mode": "centered_raw",
                "fit_split": "basis_fit",
                "mean": fit_groups[key].mean.to(torch.float32),
                "basis": fit_summary.basis,
                "eigenvalues": fit_summary.eigenvalues.to(torch.float32),
                "metrics": fit_summary.metrics,
            }
        if signal == "post_gate_delta" and mode == "aggregated":
            payload.update(
                {
                    "eval_count": eval_groups[key].count,
                    "eval_mean": eval_groups[key].mean.to(torch.float32),
                    "eval_m2": eval_groups[key].m2.to(torch.float32),
                }
            )
        torch.save(payload, path)
        saved = SavedBasis(
            seed=seed,
            epoch=epoch,
            layer=layer,
            signal_type=signal,
            temporal_mode=mode,
            path=path,
            mean=fit_groups[key].mean.to(torch.float32),
            basis=fit_summary.basis,
            metrics=fit_summary.metrics,
        )
        result.bases[key] = saved
        for split, covariance, summary in (
            ("basis_fit", fit_groups[key], fit_summary),
            ("basis_eval", eval_groups[key], eval_summary),
        ):
            result.spectral_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "checkpoint_kind": checkpoint_kind,
                    "layer": layer,
                    "signal_type": signal,
                    "temporal_mode": mode,
                    "analysis_mode": "centered_raw",
                    "split": split,
                    **broadening_fields(summary),
                }
            )
    result.delta_eval = {
        layer: eval_groups[(layer, "post_gate_delta", "aggregated")]
        for layer in dimensions
    }

    labels_cpu = torch.cat(eval_labels)
    for layer, dimension in dimensions.items():
        vectors = torch.cat(gate_vectors[layer])
        same, different, same_n, different_n = mean_pair_cosine(
            vectors, labels_cpu, seed=seed + epoch * 101 + int(layer[-1])
        )
        utilization = gate_sumabs[layer] / max(1, gate_count[layer] // dimension)
        neuron_entropy, effective_count = effective_neuron_count(utilization)
        gcoh = torch.cat(gate_coherence[layer])
        gtcos = torch.cat(gate_temporal_cosine[layer])
        dcoh = torch.cat(delta_coherence[layer])
        dtcos = torch.cat(delta_temporal_cosine[layer])
        parity = fit_parity[layer] + eval_parity[layer]
        result.gate_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "checkpoint_kind": checkpoint_kind,
                "layer": layer,
                "mean_absolute_gate": float(gate_sumabs[layer].sum() / gate_count[layer]),
                "rms_gate": float(np.sqrt(gate_sumsq[layer] / gate_count[layer])),
                "relative_threshold": 0.1,
                "relative_threshold_definition": "g > 0.1 * theoretical fast-sigmoid maximum (1.0)",
                "fraction_above_relative_threshold": gate_active[layer] / gate_count[layer],
                "neuron_utilization_entropy": neuron_entropy,
                "effective_neuron_count": effective_count,
                "same_class_gate_cosine": same,
                "different_class_gate_cosine": different,
                "same_class_pairs": same_n,
                "different_class_pairs": different_n,
                "mean_temporal_gate_cosine": float(gtcos.mean()),
                "mean_gate_coherence": float(gcoh.mean()),
                "gate_r95": result.bases[(layer, "gate", "timestep")].metrics["r95"],
                "gate_entropy_rank": result.bases[(layer, "gate", "timestep")].metrics["entropy_rank"],
                "gate_participation_ratio": result.bases[(layer, "gate", "timestep")].metrics["participation_ratio"],
                "delta_gate_parity_max_abs": max(value[0] for value in parity),
                "delta_gate_parity_max_relative": max(value[1] for value in parity),
            }
        )
        result.coherence_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "checkpoint_kind": checkpoint_kind,
                "layer": layer,
                "delta_coherence_mean": float(dcoh.mean()),
                "delta_coherence_std": float(dcoh.std(unbiased=True)),
                "delta_pairwise_temporal_cosine_mean": float(dtcos.mean()),
                "delta_pairwise_temporal_cosine_std": float(dtcos.std(unbiased=True)),
                "gate_coherence_mean": float(gcoh.mean()),
                "gate_pairwise_temporal_cosine_mean": float(gtcos.mean()),
            }
        )

    if detailed:
        for layer in dimensions:
            class_rows, _bases16, _bases64 = _class_details(
                seed=seed,
                epoch=epoch,
                layer=layer,
                fit_by_class=fit_class[layer],
                eval_by_class=eval_class[layer],
                device=device,
            )
            result.class_rows.extend(class_rows)
            for component, covariance in (
                ("within_class_residual", within_eval[layer]),
                ("between_class_fit_mean", between_eval[layer]),
            ):
                metrics, _singular, _energy, _cumulative = spectrum_statistics(
                    covariance,
                    algebraic_max_dim=dimensions[layer],
                    device=device,
                )
                result.class_rows.append(
                    {
                        "record_type": "between_within_effective_rank",
                        "seed": seed,
                        "epoch": epoch,
                        "layer": layer,
                        "class_a": None,
                        "class_b": None,
                        "split": "basis_eval",
                        "component": component,
                        "k": None,
                        "overlap": None,
                        **metrics,
                    }
                )
            temporal_tensor = torch.cat(temporal_fit[layer], dim=0)
            temporal_rows, sanity_rows = _temporal_details(
                seed=seed,
                epoch=epoch,
                layer=layer,
                fit_signal=temporal_tensor,
                device=device,
                sanity=sanity and layer == "hidden_1",
            )
            result.temporal_rows.extend(temporal_rows)
            result.sanity_rows.extend(sanity_rows)
            result.shuffled_eval[layer] = shuffled_eval[layer]
            result.shuffled_mean[layer] = shuffled_fit[layer].mean.to(torch.float32)
            correct = result.bases[(layer, "post_gate_delta", "aggregated")]
            for condition, covariance in (
                ("correct", result.delta_eval[layer]),
                ("shuffled", shuffled_eval[layer]),
            ):
                metrics, _s, _e, _c = spectrum_statistics(
                    covariance,
                    algebraic_max_dim=dimensions[layer],
                    device=device,
                )
                result.control_rows.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "layer": layer,
                        "label_condition": condition,
                        "split": "basis_eval",
                        "gate_changed_by_label": False,
                        "basis_reference": str(correct.path),
                        **metrics,
                    }
                )

    del model, feedback_bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_basis_manifest(bases: list[SavedBasis], path: Path) -> None:
    rows = [
        {
            "seed": item.seed,
            "epoch": item.epoch,
            "layer": item.layer,
            "signal_type": item.signal_type,
            "temporal_mode": item.temporal_mode,
            "analysis_mode": "centered_raw",
            "fit_split": "basis_fit",
            "basis_columns": item.basis.shape[1],
            "basis_path": str(item.path),
            "basis_path_relative": f"bases/{item.path.name}",
            "basis_filename": item.path.name,
            "r90": item.metrics["r90"],
            "r95": item.metrics["r95"],
            "entropy_rank": item.metrics["entropy_rank"],
        }
        for item in bases
    ]
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
