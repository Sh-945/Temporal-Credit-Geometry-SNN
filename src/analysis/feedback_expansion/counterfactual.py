"""Same-sample q/g geometry counterfactuals for Experiment 01B."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from analysis.feedback_expansion.capture import capture_layer_signals
from analysis.feedback_expansion.diagnostic import (
    _forward_inputs,
    _prepare,
    load_checkpoint_model,
)
from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics


def counterfactual_decomposition(
    *,
    config: dict[str, Any],
    checkpoint_paths: dict[int, Path],
    scheduled_epochs: list[int],
    seed: int,
    eval_loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Freeze q or gate at initialization/final with strict sample alignment."""

    epochs = [epoch for epoch in scheduled_epochs if epoch in checkpoint_paths]
    first_epoch, final_epoch = min(epochs), max(epochs)
    reference_models = {}
    for reference_epoch in (first_epoch, final_epoch):
        reference_models[reference_epoch] = load_checkpoint_model(
            config, checkpoint_paths[reference_epoch], device
        )[:2]
    rows: list[dict[str, Any]] = []
    pointwise = True
    for epoch in epochs:
        current_model, current_feedback, _state = load_checkpoint_model(
            config, checkpoint_paths[epoch], device
        )
        accumulators: dict[tuple[str, int, str], OnlineCovariance] = {}
        for layer_index, spec in enumerate(current_model.hidden_specs):
            dimension = int(spec["dimension"])
            accumulators[("real", layer_index, "current")] = OnlineCovariance(dimension)
            for reference_epoch in (first_epoch, final_epoch):
                tag = "initial" if reference_epoch == first_epoch else "final"
                accumulators[("freeze_q", layer_index, tag)] = OnlineCovariance(dimension)
                accumulators[("freeze_gate", layer_index, tag)] = OnlineCovariance(dimension)
        parity: list[float] = []
        for batch in eval_loader:
            samples, labels = _prepare(batch, device)
            current_inputs, current_error = _forward_inputs(
                current_model, samples, labels, pointwise
            )
            reference_forward = {}
            for reference_epoch, (model, feedback) in reference_models.items():
                inputs, error = _forward_inputs(model, samples, labels, pointwise)
                reference_forward[reference_epoch] = (inputs, error, model, feedback)
            for layer_index, current_input in enumerate(current_inputs):
                current = capture_layer_signals(
                    current_model,
                    current_feedback,
                    layer_index,
                    current_input,
                    current_error,
                    detach_temporal=True,
                )
                accumulators[("real", layer_index, "current")].update(
                    current.delta.mean(dim=0)
                )
                parity.append(current.parity_relative)
                for reference_epoch, values in reference_forward.items():
                    inputs, error, model, feedback = values
                    reference = capture_layer_signals(
                        model,
                        feedback,
                        layer_index,
                        inputs[layer_index],
                        error,
                        detach_temporal=True,
                    )
                    tag = "initial" if reference_epoch == first_epoch else "final"
                    freeze_q = (
                        reference.q.unsqueeze(0) * current.gate
                    ).mean(dim=0)
                    freeze_gate = (
                        current.q.unsqueeze(0) * reference.gate
                    ).mean(dim=0)
                    accumulators[("freeze_q", layer_index, tag)].update(freeze_q)
                    accumulators[("freeze_gate", layer_index, tag)].update(freeze_gate)
        for (component, layer_index, reference), covariance in sorted(accumulators.items()):
            metrics, singular, energy, _cumulative = spectrum_statistics(
                covariance,
                algebraic_max_dim=covariance.ambient_dim,
                device=device,
            )
            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "layer": f"hidden_{layer_index + 1}",
                    "component_mode": component,
                    "reference": reference,
                    "reference_epoch": (
                        first_epoch if reference == "initial" else final_epoch if reference == "final" else epoch
                    ),
                    "split": "basis_eval",
                    "temporal_mode": "aggregated",
                    "alignment": "same probe sample, timestep, neuron, and checkpoint feedback coordinates",
                    "total_centered_energy": float((singular * singular).sum()),
                    "top10_energy_fraction": float(energy[:10].sum()),
                    "top32_energy_fraction": float(energy[:32].sum()),
                    "delta_gate_parity_max_relative": max(parity),
                    **metrics,
                }
            )
        del current_model, current_feedback
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for model, feedback in reference_models.values():
        del model, feedback
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows
