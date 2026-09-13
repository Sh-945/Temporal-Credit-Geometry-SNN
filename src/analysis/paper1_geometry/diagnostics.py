"""Matched current-level diagnostics for SNN-DFA, SNN-BPTT, and ANN-DFA."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Collection, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from analysis.feedback_expansion.core import (
    energy_in_basis,
    principal_subspace_metrics,
    scatter_about,
)
from analysis.paper1_geometry.metrics import (
    GeometryAccumulator,
    matrix_spectrum_metrics,
    tensor_cosine,
)
from analysis.update_relevance.core import capture_bptt_updates
from methods.gate_intervention import GateMode, apply_gate_intervention


def _prepare(batch, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    if len(batch) != 3:
        raise ValueError("Paper 1 diagnostic loaders must return sample, label, index")
    samples, labels, indices = batch
    return (
        samples.transpose(0, 1).contiguous().to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        labels.to(device=device, non_blocking=True),
        indices.to(device=device, non_blocking=True),
    )


def _weight_from_delta(delta: Tensor, layer_input: Tensor) -> Tensor:
    if delta.ndim != 3 or layer_input.ndim != 3:
        raise ValueError("Paper 1 FC update diagnostic expects [T,B,D]")
    return torch.einsum("tbo,tbi->oi", delta / delta.numel(), layer_input.detach())


def _capture_snn_gate(model, layer_index: int, layer_input: Tensor) -> Tensor:
    if getattr(model, "architecture", None) == "conv":
        module = (
            model.conv_layers[layer_index].conv
            if layer_index < len(model.conv_layers)
            else model.fc_hidden.linear
        )
    else:
        module = model.hidden_layers[layer_index].linear
    captured: list[Tensor] = []
    handle = module.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        spikes = model.replay_hidden(
            layer_index, layer_input.detach(), detach_temporal=True
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("expected exactly one hidden current")
    return torch.autograd.grad(
        spikes,
        captured[0],
        grad_outputs=torch.ones_like(spikes),
        retain_graph=False,
    )[0].reshape_as(spikes).detach()


def _snn_current_module_and_weight(model, layer_index: int):
    if getattr(model, "architecture", None) == "conv":
        if layer_index < len(model.conv_layers):
            module = model.conv_layers[layer_index].conv
            return module, module.weight
        return model.fc_hidden.linear, model.fc_hidden.linear.weight
    module = model.hidden_layers[layer_index].linear
    return module, module.weight


def _capture_snn_dfa_layer(
    model,
    feedback_bank,
    layer_index: int,
    layer_input: Tensor,
    output_error: Tensor,
    *,
    gate_mode: GateMode | str,
    seed: int,
    epoch: int,
    sample_indices: Tensor,
    teaching_override: Tensor | None = None,
):
    """Capture the exact production current and a gradient-only intervention."""

    teaching = (
        feedback_bank.project(layer_index, output_error).detach()
        if teaching_override is None
        else teaching_override.detach()
    )
    module, weight = _snn_current_module_and_weight(model, layer_index)
    captured: list[Tensor] = []
    handle = module.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        spikes = model.replay_hidden(
            layer_index, layer_input.detach(), detach_temporal=True
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("expected one production current")
    current = captured[0]
    current_view = current.reshape_as(spikes)
    gate = torch.autograd.grad(
        spikes,
        current,
        grad_outputs=torch.ones_like(spikes),
        retain_graph=True,
    )[0].reshape_as(spikes).detach()
    applied_gate = apply_gate_intervention(
        gate,
        gate_mode,
        seed=seed,
        epoch=epoch,
        sample_indices=sample_indices,
    )
    if spikes.ndim == 5:
        height, width = spikes.shape[-2:]
        production_teaching = teaching[:, :, None, None] / float(height * width)
    else:
        production_teaching = teaching
    expected_production_delta = production_teaching.unsqueeze(0) * gate
    production_proxy = (spikes * production_teaching.unsqueeze(0)).mean()
    production_current_gradient = torch.autograd.grad(
        production_proxy, current, retain_graph=True
    )[0].reshape_as(spikes)
    production_delta = production_current_gradient.detach() * production_current_gradient.numel()
    parity = torch.linalg.vector_norm(production_delta - expected_production_delta) / (
        torch.linalg.vector_norm(production_delta) + 1e-30
    )
    applied_delta = production_teaching.unsqueeze(0) * applied_gate
    injected_proxy = (current_view * applied_delta).mean()
    weight_gradient = torch.autograd.grad(injected_proxy, weight)[0].detach()
    return {
        "q": teaching,
        "gate": gate,
        "applied_gate": applied_gate,
        "delta": applied_delta.detach(),
        "production_delta": production_delta,
        "weight_gradient": weight_gradient,
        "parity_relative": float(parity),
    }


def _ann_task_loss(logits: Tensor, labels: Tensor, classes: int) -> Tensor:
    desired = F.one_hot(labels, num_classes=classes).to(logits.dtype)
    return 0.5 * (logits.mean(dim=0) - desired).square().sum(dim=1).mean()


def _capture_ann_bptt(model, samples: Tensor, labels: Tensor) -> dict[str, Any]:
    logits, _inputs, currents, _readout_input = model.forward_with_cache(samples)
    loss = _ann_task_loss(logits, labels, model.num_classes)
    weights = [layer.weight for layer in model.hidden_layers]
    gradients = torch.autograd.grad(loss, tuple(currents) + tuple(weights))
    count = len(currents)
    time, batch = samples.shape[:2]
    current_gradients = [
        value.detach().reshape(time, batch, -1) for value in gradients[:count]
    ]
    return {
        "delta": [value * time * batch for value in current_gradients],
        "weight_gradient": [value.detach() for value in gradients[count:]],
        "task_loss": float(loss.detach()),
    }


def _capture_ann_layer(model, bank, layer_index: int, layer_input: Tensor, error: Tensor):
    teaching = bank.project(layer_index, error).detach()
    activation, current = model.replay_hidden(layer_index, layer_input.detach())
    gate = torch.autograd.grad(
        activation,
        current,
        grad_outputs=torch.ones_like(activation),
        retain_graph=True,
    )[0].detach()
    local_proxy = (activation * teaching.unsqueeze(0)).mean()
    current_gradient = torch.autograd.grad(local_proxy, current)[0]
    delta = current_gradient.detach() * current_gradient.numel()
    parity = torch.linalg.vector_norm(delta - teaching.unsqueeze(0) * gate) / (
        torch.linalg.vector_norm(delta) + 1e-30
    )
    return teaching, gate, delta, float(parity)


def _finish(
    accumulators: dict[tuple[str, int, str], GeometryAccumulator],
    *,
    prefix: dict[str, Any],
    device: torch.device,
    gradient_sums: dict[tuple[int, str], Tensor],
    gradient_counts: dict[tuple[int, str], int],
    parity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    geometry: list[dict[str, Any]] = []
    bases: dict[str, Tensor] = {}
    grouped = defaultdict(dict)
    for (signal_type, layer_index, probe_split), accumulator in accumulators.items():
        rows, split_bases = accumulator.rows(
            device=device,
            prefix={
                **prefix,
                "signal_type": signal_type,
                "layer": f"hidden_{layer_index + 1}",
            },
            probe_split=probe_split,
        )
        geometry.extend(rows)
        for temporal_mode, basis in split_bases.items():
            key = f"{signal_type}|hidden_{layer_index + 1}|{probe_split}|{temporal_mode}"
            bases[key] = basis
        grouped[(signal_type, layer_index)][probe_split] = accumulator

    generalization: list[dict[str, Any]] = []
    for (signal_type, layer_index), splits in grouped.items():
        if "basis_fit" not in splits or "basis_eval" not in splits:
            continue
        for mode in ("timestep", "aggregated"):
            fit_acc = splits["basis_fit"].raw[mode]
            eval_acc = splits["basis_eval"].raw[mode]
            basis_key = f"{signal_type}|hidden_{layer_index + 1}|basis_fit|{mode}"
            basis = bases[basis_key]
            fit_row = next(
                row
                for row in geometry
                if row["signal_type"] == signal_type
                and row["layer"] == f"hidden_{layer_index + 1}"
                and row["probe_split"] == "basis_fit"
                and row["temporal_mode"] == mode
                and row["residualization"] == "raw"
            )
            rank = max(1, int(fit_row["r95"]))
            heldout = energy_in_basis(scatter_about(eval_acc, fit_acc.mean), basis, rank)
            generalization.append(
                {
                    **prefix,
                    "signal_type": signal_type,
                    "layer": f"hidden_{layer_index + 1}",
                    "temporal_mode": mode,
                    "fit_r95": rank,
                    "heldout_energy_at_fit_r95": heldout,
                }
            )

    updates: list[dict[str, Any]] = []
    layer_indices = sorted({layer for layer, _kind in gradient_sums})
    for layer_index in layer_indices:
        method_key = (layer_index, "method")
        bp_key = (layer_index, "bp")
        method_gradient = gradient_sums[method_key] / max(1, gradient_counts[method_key])
        bp_gradient = gradient_sums[bp_key] / max(1, gradient_counts[bp_key])
        updates.append(
            {
                **prefix,
                "layer": f"hidden_{layer_index + 1}",
                "gradient_cosine_to_bp": tensor_cosine(method_gradient, bp_gradient),
                "relative_norm_to_bp": float(
                    torch.linalg.vector_norm(method_gradient)
                    / (torch.linalg.vector_norm(bp_gradient) + 1e-30)
                ),
                "method_gradient_norm": float(torch.linalg.vector_norm(method_gradient)),
                "bp_gradient_norm": float(torch.linalg.vector_norm(bp_gradient)),
                **{f"update_{key}": value for key, value in matrix_spectrum_metrics(method_gradient).items()},
                **{f"bp_update_{key}": value for key, value in matrix_spectrum_metrics(bp_gradient).items()},
            }
        )

    comparisons: list[dict[str, Any]] = []
    signal_types = sorted({key[0] for key in grouped})
    for layer_index in layer_indices:
        for first_index, first in enumerate(signal_types):
            for second in signal_types[first_index + 1 :]:
                first_key = f"{first}|hidden_{layer_index + 1}|basis_fit|timestep"
                second_key = f"{second}|hidden_{layer_index + 1}|basis_fit|timestep"
                if first_key not in bases or second_key not in bases:
                    continue
                first_r95 = next(
                    int(row["r95"])
                    for row in geometry
                    if row["signal_type"] == first
                    and row["layer"] == f"hidden_{layer_index + 1}"
                    and row["probe_split"] == "basis_fit"
                    and row["temporal_mode"] == "timestep"
                    and row["residualization"] == "raw"
                )
                second_r95 = next(
                    int(row["r95"])
                    for row in geometry
                    if row["signal_type"] == second
                    and row["layer"] == f"hidden_{layer_index + 1}"
                    and row["probe_split"] == "basis_fit"
                    and row["temporal_mode"] == "timestep"
                    and row["residualization"] == "raw"
                )
                rank = max(1, min(first_r95, second_r95))
                comparisons.append(
                    {
                        **prefix,
                        "layer": f"hidden_{layer_index + 1}",
                        "first_signal": first,
                        "second_signal": second,
                        **principal_subspace_metrics(bases[first_key], bases[second_key], rank),
                    }
                )
    return {
        "geometry": geometry,
        "basis_generalization": generalization,
        "weight_updates": updates,
        "subspace_comparisons": comparisons,
        "parity": parity_rows,
        "bases": bases,
    }


def diagnose_snn_checkpoint(
    *,
    model,
    feedback_bank,
    loaders: dict[str, Iterable],
    device: torch.device,
    dataset: str,
    method: str,
    seed: int,
    epoch: int,
    is_bptt: bool,
    gate_mode: GateMode | str = GateMode.ACTUAL,
    intervention_layers: Collection[int] | None = None,
    q_reference_model=None,
    q_reference_feedback_bank=None,
) -> dict[str, Any]:
    """Extract actual and counterfactual signals at the same hidden current."""

    model.eval()
    feedback_bank.eval()
    if (q_reference_model is None) != (q_reference_feedback_bank is None):
        raise ValueError("q reference model and feedback bank must be provided together")
    if q_reference_model is not None:
        q_reference_model.eval()
        q_reference_feedback_bank.eval()
    prefix = {
        "dataset": dataset,
        "model": "snn",
        "method": method,
        "seed": int(seed),
        "epoch": int(epoch),
    }
    accumulators: dict[tuple[str, int, str], GeometryAccumulator] = {}
    gradient_sums: dict[tuple[int, str], Tensor] = {}
    gradient_counts: dict[tuple[int, str], int] = defaultdict(int)
    parity_rows: list[dict[str, Any]] = []

    def accumulator(signal: str, layer: int, split: str) -> GeometryAccumulator:
        key = (signal, layer, split)
        if key not in accumulators:
            accumulators[key] = GeometryAccumulator(
                int(model.hidden_specs[layer]["dimension"])
            )
        return accumulators[key]

    for probe_split in ("basis_fit", "basis_eval"):
        for batch_index, batch in enumerate(loaders[probe_split]):
            samples, labels, indices = _prepare(batch, device)
            bp = capture_bptt_updates(model, samples, labels)
            with torch.no_grad():
                output, hidden_inputs, _readout = model.forward_with_cache(
                    samples, detach_temporal=True
                )
                desired = F.one_hot(labels, num_classes=model.num_classes).to(output.dtype)
                error = output.mean(dim=0) - desired
                reference_error = None
                if q_reference_model is not None:
                    reference_output = q_reference_model(samples, detach_temporal=True)
                    reference_desired = F.one_hot(
                        labels, num_classes=q_reference_model.num_classes
                    ).to(reference_output.dtype)
                    reference_error = reference_output.mean(dim=0) - reference_desired
            batch_size = int(labels.numel())
            for layer_index, layer_input in enumerate(hidden_inputs):
                effective_gate_mode = (
                    GateMode(gate_mode)
                    if intervention_layers is None or layer_index in intervention_layers
                    else GateMode.ACTUAL
                )
                bp_delta = bp["delta"][layer_index]
                bp_signal = "bptt_actual" if is_bptt else "dfa_checkpoint_counterfactual_bp"
                accumulator(bp_signal, layer_index, probe_split).update(bp_delta, labels)
                gate = _capture_snn_gate(model, layer_index, layer_input)
                accumulator("gate_actual", layer_index, probe_split).update(gate, labels)
                if is_bptt:
                    method_gradient = bp["weight_gradient"][layer_index]
                else:
                    capture = _capture_snn_dfa_layer(
                        model,
                        feedback_bank,
                        layer_index,
                        layer_input,
                        error,
                        gate_mode=effective_gate_mode,
                        seed=seed,
                        epoch=epoch,
                        sample_indices=indices,
                        teaching_override=(
                            None
                            if reference_error is None
                            else q_reference_feedback_bank.project(
                                layer_index, reference_error
                            )
                        ),
                    )
                    applied_gate = capture["applied_gate"]
                    applied_delta = capture["delta"]
                    signal_name = (
                        "dfa_fixed_reference_q_current_gate"
                        if reference_error is not None
                        else f"dfa_{effective_gate_mode.value}"
                    )
                    accumulator(signal_name, layer_index, probe_split).update(
                        applied_delta, labels
                    )
                    q_signal_name = (
                        "dfa_reference_q" if reference_error is not None else "dfa_q"
                    )
                    accumulator(q_signal_name, layer_index, probe_split).update(
                        capture["q"].unsqueeze(0).expand(samples.shape[0], -1, -1), labels
                    )
                    accumulator("gate_applied", layer_index, probe_split).update(
                        applied_gate, labels
                    )
                    method_gradient = capture["weight_gradient"]
                    parity_rows.append(
                        {
                            **prefix,
                            "probe_split": probe_split,
                            "batch_index": batch_index,
                            "layer": f"hidden_{layer_index + 1}",
                            "gate_mode": effective_gate_mode.value,
                            "production_delta_relative_error": capture["parity_relative"],
                            "actual_vs_intervened_delta_cosine": tensor_cosine(
                                capture["production_delta"], capture["delta"]
                            ),
                            "applied_gate_norm_ratio": float(
                                torch.linalg.vector_norm(applied_gate)
                                / (torch.linalg.vector_norm(capture["gate"]) + 1e-30)
                            ),
                        }
                    )
                for kind, value in (
                    ("method", method_gradient),
                    ("bp", bp["weight_gradient"][layer_index]),
                ):
                    key = (layer_index, kind)
                    weighted = value.detach().cpu() * batch_size
                    gradient_sums[key] = gradient_sums.get(key, torch.zeros_like(weighted)) + weighted
                    gradient_counts[key] += batch_size
    return _finish(
        accumulators,
        prefix=prefix,
        device=device,
        gradient_sums=gradient_sums,
        gradient_counts=gradient_counts,
        parity_rows=parity_rows,
    )


def diagnose_ann_checkpoint(
    *,
    model,
    feedback_bank,
    loaders: dict[str, Iterable],
    device: torch.device,
    dataset: str,
    method: str,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    feedback_bank.eval()
    prefix = {
        "dataset": dataset,
        "model": "temporal_ann",
        "method": method,
        "seed": int(seed),
        "epoch": int(epoch),
    }
    accumulators: dict[tuple[str, int, str], GeometryAccumulator] = {}
    gradient_sums: dict[tuple[int, str], Tensor] = {}
    gradient_counts: dict[tuple[int, str], int] = defaultdict(int)
    parity_rows: list[dict[str, Any]] = []

    def accumulator(signal: str, layer: int, split: str) -> GeometryAccumulator:
        key = (signal, layer, split)
        if key not in accumulators:
            accumulators[key] = GeometryAccumulator(
                int(model.hidden_specs[layer]["dimension"])
            )
        return accumulators[key]

    for probe_split in ("basis_fit", "basis_eval"):
        for batch_index, batch in enumerate(loaders[probe_split]):
            samples, labels, _indices = _prepare(batch, device)
            bp = _capture_ann_bptt(model, samples, labels)
            with torch.no_grad():
                logits, hidden_inputs, _currents, _readout = model.forward_with_cache(samples)
                desired = F.one_hot(labels, num_classes=model.num_classes).to(logits.dtype)
                error = logits.mean(dim=0) - desired
            batch_size = int(labels.numel())
            for layer_index, layer_input in enumerate(hidden_inputs):
                teaching, gate, delta, parity = _capture_ann_layer(
                    model, feedback_bank, layer_index, layer_input, error
                )
                accumulator("ann_dfa_actual", layer_index, probe_split).update(delta, labels)
                accumulator("ann_dfa_q", layer_index, probe_split).update(
                    teaching.unsqueeze(0).expand(samples.shape[0], -1, -1), labels
                )
                accumulator("ann_relu_gate", layer_index, probe_split).update(gate, labels)
                accumulator("ann_counterfactual_bp", layer_index, probe_split).update(
                    bp["delta"][layer_index], labels
                )
                method_gradient = _weight_from_delta(delta, layer_input)
                for kind, value in (
                    ("method", method_gradient),
                    ("bp", bp["weight_gradient"][layer_index]),
                ):
                    key = (layer_index, kind)
                    weighted = value.detach().cpu() * batch_size
                    gradient_sums[key] = gradient_sums.get(key, torch.zeros_like(weighted)) + weighted
                    gradient_counts[key] += batch_size
                parity_rows.append(
                    {
                        **prefix,
                        "probe_split": probe_split,
                        "batch_index": batch_index,
                        "layer": f"hidden_{layer_index + 1}",
                        "gate_mode": "relu_actual",
                        "production_delta_relative_error": parity,
                        "applied_gate_norm_ratio": 1.0,
                    }
                )
    return _finish(
        accumulators,
        prefix=prefix,
        device=device,
        gradient_sums=gradient_sums,
        gradient_counts=gradient_counts,
        parity_rows=parity_rows,
    )
