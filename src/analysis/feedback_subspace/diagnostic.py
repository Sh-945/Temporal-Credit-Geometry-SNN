"""Offline capture of the signals actually used by dense sDFA updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from analysis.feedback_subspace.metrics import (
    DualCovariance,
    spectrum_statistics,
    uncentered_numerical_rank,
)
from methods import FeedbackBank
from models import build_model
from training.checkpoint import load_checkpoint


def _tensor_checksum(tensor: Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def feedback_matrix_rows(
    feedback_bank: FeedbackBank,
    *,
    seed: int,
    epoch: int,
    checkpoint_kind: str,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scale = float(feedback_bank.feedback_config.get("scale", 1.0))
    for index, feedback in enumerate(feedback_bank.layers):
        matrix = feedback.materialize().detach().cpu().to(torch.float64)
        singular = torch.linalg.svdvals(matrix).numpy()
        positive = singular[singular > max(matrix.shape) * np.finfo(np.float32).eps * singular[0]]
        condition_number = (
            float(positive[0] / positive[-1]) if positive.size else float("nan")
        )
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "checkpoint_kind": checkpoint_kind,
                "layer": f"hidden_{index + 1}",
                "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                "numerical_rank": int(torch.linalg.matrix_rank(matrix).item()),
                "frobenius_norm": float(torch.linalg.vector_norm(matrix)),
                "spectral_norm": float(singular[0]),
                "condition_number": condition_number,
                "singular_values": json.dumps(singular.tolist()),
                "checksum": _tensor_checksum(matrix),
                "fixed_verified": None,
                "initialization_distribution": (
                    f"torch.randn(hidden_dim,error_dim)/sqrt(error_dim), scale={scale}"
                ),
                "checkpoint": str(checkpoint_path),
            }
        )
    return rows


def verify_fixed_feedback(rows: list[dict[str, Any]]) -> None:
    initial: dict[tuple[int, str], str] = {}
    for row in sorted(rows, key=lambda item: (item["seed"], item["epoch"])):
        key = (int(row["seed"]), str(row["layer"]))
        initial.setdefault(key, str(row["checksum"]))
        row["fixed_verified"] = str(row["checksum"]) == initial[key]


def _capture_post_gate_delta(
    model,
    layer_index: int,
    layer_input: Tensor,
    teaching: Tensor,
    *,
    detach_temporal: bool,
) -> Tensor:
    """Capture the exact current-gradient used by the local proxy loss.

    The production loss is ``mean(spikes * teaching)``.  A temporary forward
    hook retains the replayed linear current, and ``autograd.grad`` returns the
    actual local-loss gradient with respect to that current.  Multiplying by
    the mean denominator removes only a global scalar, yielding the exact
    per-timestep ``q * surrogate_gate`` signal.  No hook is present in normal
    training.
    """

    captured: list[Tensor] = []

    def remember_current(_module, _inputs, output: Tensor) -> None:
        captured.append(output)

    linear = model.hidden_layers[layer_index].linear
    handle = linear.register_forward_hook(remember_current)
    try:
        spikes = model.replay_hidden(
            layer_index, layer_input.detach(), detach_temporal=detach_temporal
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one replay current for hidden_{layer_index + 1}, got {len(captured)}"
        )
    local_proxy = (spikes * teaching.unsqueeze(0)).mean()
    current_gradient = torch.autograd.grad(
        local_proxy, captured[0], retain_graph=False, create_graph=False
    )[0]
    time_steps, batch_size, hidden_dim = spikes.shape
    scale = time_steps * batch_size * hidden_dim
    return current_gradient.reshape(time_steps, batch_size, hidden_dim) * scale


def _group(
    groups: dict[tuple[str, str, str, str], DualCovariance],
    condition: str,
    layer: str,
    signal_type: str,
    temporal_mode: str,
    ambient_dim: int,
) -> DualCovariance:
    key = (condition, layer, signal_type, temporal_mode)
    if key not in groups:
        groups[key] = DualCovariance(ambient_dim)
    return groups[key]


def diagnose_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_kind: str,
    epoch: int,
    seed: int,
    probe_batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    shuffled_labels: Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Diagnose one checkpoint using a fixed in-memory train probe."""

    if str(config["method"]["name"]).lower() not in {"dfa", "sdfa"}:
        raise ValueError("feedback-subspace Experiment 01 requires dense DFA/sDFA")
    if str(config["model"]["architecture"]).lower() != "fc":
        raise ValueError("Experiment 01 primary diagnostic expects the selected FC SNN")

    model = build_model(config).to(device)
    feedback_bank = FeedbackBank(model, config).to(device)
    load_checkpoint(
        checkpoint_path,
        model=model,
        feedback_bank=feedback_bank,
        map_location=device,
    )
    model.eval()
    feedback_bank.eval()
    pointwise = config["method"].get("temporal_mode", "pointwise") == "pointwise"
    time_steps = int(config["data"]["time_steps"])
    groups: dict[tuple[str, str, str, str], DualCovariance] = {}
    label_offset = 0

    for batch_samples, correct_target in probe_batches:
        batch_size = int(correct_target.numel())
        samples = batch_samples.transpose(0, 1).contiguous().to(
            device, dtype=torch.float32, non_blocking=True
        )
        correct_target = correct_target.to(device, non_blocking=True)
        with torch.no_grad():
            output, hidden_inputs, _readout_input = model.forward_with_cache(
                samples, detach_temporal=pointwise
            )
            mean_output = output.mean(dim=0)

        targets = {"correct": correct_target}
        if shuffled_labels is not None:
            targets["shuffled"] = shuffled_labels[
                label_offset : label_offset + batch_size
            ].to(device, non_blocking=True)
        label_offset += batch_size

        errors: dict[str, Tensor] = {}
        for condition, target in targets.items():
            desired = F.one_hot(
                target, num_classes=model.num_classes
            ).to(dtype=torch.float32)
            error = (mean_output - desired).detach()
            errors[condition] = error
            _group(
                groups,
                condition,
                "output",
                "output_error",
                "aggregated",
                model.num_classes,
            ).update(error)

        for layer_index, (spec, layer_input) in enumerate(
            zip(model.hidden_specs, hidden_inputs)
        ):
            layer_name = f"hidden_{layer_index + 1}"
            hidden_dim = int(spec["dimension"])
            for condition, error in errors.items():
                with torch.no_grad():
                    teaching = feedback_bank.project(layer_index, error).detach()
                _group(
                    groups,
                    condition,
                    layer_name,
                    "pre_gate_q",
                    "aggregated",
                    hidden_dim,
                ).update(teaching)
                delta = _capture_post_gate_delta(
                    model,
                    layer_index,
                    layer_input,
                    teaching,
                    detach_temporal=pointwise,
                ).detach()
                _group(
                    groups,
                    condition,
                    layer_name,
                    "post_gate_delta",
                    "timestep",
                    hidden_dim,
                ).update(delta)
                # The production proxy loss averages over time, batch, and
                # hidden dimension.  The update-relevant time rule is mean.
                _group(
                    groups,
                    condition,
                    layer_name,
                    "post_gate_delta",
                    "aggregated",
                    hidden_dim,
                ).update(delta.mean(dim=0))

    # e and q are sample-level signals broadcast unchanged to every timestep.
    for key, value in list(groups.items()):
        condition, layer, signal_type, temporal_mode = key
        if temporal_mode == "aggregated" and signal_type in {
            "output_error",
            "pre_gate_q",
        }:
            groups[(condition, layer, signal_type, "timestep")] = value.repeated(
                time_steps
            )

    b_rows = feedback_matrix_rows(
        feedback_bank,
        seed=seed,
        epoch=epoch,
        checkpoint_kind=checkpoint_kind,
        checkpoint_path=checkpoint_path,
    )
    rank_b = {str(row["layer"]): int(row["numerical_rank"]) for row in b_rows}
    error_ranks: dict[tuple[str, str], int] = {}
    for (condition, layer, signal_type, temporal_mode), dual in groups.items():
        if layer == "output" and signal_type == "output_error":
            error_ranks[(condition, temporal_mode)] = uncentered_numerical_rank(
                dual.raw, device
            )
    for condition in {key[0] for key in groups}:
        # Repeating the same sample-level error over T does not change its
        # algebraic rank; use the aggregated estimate for both representations.
        error_ranks[(condition, "timestep")] = error_ranks[
            (condition, "aggregated")
        ]

    metric_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for (condition, layer, signal_type, temporal_mode), dual in sorted(groups.items()):
        error_rank = error_ranks[(condition, temporal_mode)]
        if signal_type == "output_error":
            algebraic_max = error_rank
        elif signal_type == "pre_gate_q":
            algebraic_max = min(rank_b[layer], error_rank)
        else:
            algebraic_max = dual.raw.ambient_dim

        analyses = {
            "centered_raw": dual.raw,
            "centered_unit_direction": dual.direction,
        }
        for analysis_mode, accumulator in analyses.items():
            statistics, singular, energy, cumulative = spectrum_statistics(
                accumulator,
                algebraic_max_dim=algebraic_max,
                device=device,
            )
            common = {
                "seed": seed,
                "epoch": epoch,
                "checkpoint_kind": checkpoint_kind,
                "split": f"train_probe_{condition}",
                "label_condition": condition,
                "layer": layer,
                "signal_type": signal_type,
                "temporal_mode": temporal_mode,
                "analysis_mode": analysis_mode,
                "temporal_aggregation_rule": (
                    "mean" if temporal_mode == "aggregated" else "none"
                ),
                "output_error_rank": error_rank,
                "rank_B": rank_b.get(layer),
            }
            metric_rows.append({**common, **statistics})
            for singular_index, (value, fraction, running) in enumerate(
                zip(singular, energy, cumulative), start=1
            ):
                spectrum_rows.append(
                    {
                        **common,
                        "singular_index": singular_index,
                        "singular_value": float(value),
                        "explained_energy": float(fraction),
                        "cumulative_energy": float(running),
                    }
                )

    del model, feedback_bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_rows, spectrum_rows, b_rows
