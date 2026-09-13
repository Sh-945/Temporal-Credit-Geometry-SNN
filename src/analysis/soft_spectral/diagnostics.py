"""Fixed-probe geometry and BPTT-alignment diagnostics for Experiment 03."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from analysis.feedback_expansion.capture import capture_layer_signals
from analysis.feedback_expansion.core import principal_subspace_metrics, temporal_coherence
from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics
from analysis.update_relevance.core import capture_bptt_updates, pca_basis


def _prepare(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    samples, labels = batch
    return (
        samples.transpose(0, 1).contiguous().to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        labels.to(device=device, non_blocking=True),
    )


def _capture_gate(model, layer_index: int, layer_input: Tensor) -> Tensor:
    captured: list[Tensor] = []

    def remember(_module, _inputs, output: Tensor) -> None:
        captured.append(output)

    linear = model.hidden_layers[layer_index].linear
    handle = linear.register_forward_hook(remember)
    try:
        spikes = model.replay_hidden(
            layer_index, layer_input.detach(), detach_temporal=True
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("gate capture expected exactly one hidden current")
    gate = torch.autograd.grad(
        spikes,
        captured[0],
        grad_outputs=torch.ones_like(spikes),
        retain_graph=False,
    )[0]
    return gate.reshape_as(spikes).detach()


def _weight_from_delta(delta: Tensor, layer_input: Tensor) -> Tensor:
    return torch.einsum(
        "tbo,tbi->oi", delta / delta.numel(), layer_input.detach()
    )


def _gradient_metrics(method_gradient: Tensor, bp_gradient: Tensor) -> dict[str, float]:
    method_norm = torch.linalg.vector_norm(method_gradient)
    bp_norm = torch.linalg.vector_norm(bp_gradient)
    cosine = torch.sum(method_gradient * bp_gradient) / (
        method_norm * bp_norm + 1e-30
    )
    threshold = max(1e-12, 1e-6 * float(bp_gradient.abs().max()))
    valid = (method_gradient.abs() > threshold) | (bp_gradient.abs() > threshold)
    agreement = (
        float(
            (
                torch.sign(method_gradient[valid])
                == torch.sign(bp_gradient[valid])
            ).float().mean()
        )
        if bool(valid.any())
        else float("nan")
    )
    return {
        "gradient_cosine": float(cosine),
        "method_gradient_norm": float(method_norm),
        "bp_gradient_norm": float(bp_norm),
        "relative_norm": float(method_norm / (bp_norm + 1e-30)),
        "sign_agreement": agreement,
        "sign_threshold": threshold,
        "sign_parameter_count": int(valid.sum()),
    }


def diagnose_model(
    *,
    model,
    feedback_bank,
    spectral_filter,
    method_key: str,
    seed: int,
    epoch: int,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    is_bptt: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Diagnose one immutable model/basis state on the fixed 512-sample probe."""

    model.eval()
    feedback_bank.eval()
    layer_count = len(model.hidden_specs)
    dimensions = [int(spec["dimension"]) for spec in model.hidden_specs]
    method_cov = {
        (layer, mode): OnlineCovariance(dimensions[layer])
        for layer in range(layer_count)
        for mode in ("aggregated", "timestep")
    }
    gate_cov = {layer: OnlineCovariance(dimensions[layer]) for layer in range(layer_count)}
    bp_cov = {
        (layer, mode): OnlineCovariance(dimensions[layer])
        for layer in range(layer_count)
        for mode in ("aggregated", "timestep")
    }
    gate_temporal_cosines: dict[int, list[Tensor]] = {
        layer: [] for layer in range(layer_count)
    }
    alignment_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(loader):
        samples, labels = _prepare(batch, device)
        with torch.no_grad():
            output, hidden_inputs, _readout = model.forward_with_cache(
                samples, detach_temporal=True
            )
            desired = torch.nn.functional.one_hot(
                labels, num_classes=model.num_classes
            ).to(output.dtype)
            error = output.mean(dim=0) - desired
        bp = capture_bptt_updates(model, samples, labels)

        if is_bptt:
            for layer_index, layer_input in enumerate(hidden_inputs):
                delta = bp["delta"][layer_index]
                method_cov[(layer_index, "timestep")].update(delta)
                method_cov[(layer_index, "aggregated")].update(delta.mean(dim=0))
                bp_cov[(layer_index, "timestep")].update(delta)
                bp_cov[(layer_index, "aggregated")].update(delta.mean(dim=0))
                gate = _capture_gate(model, layer_index, layer_input)
                gate_cov[layer_index].update(gate)
                _coherence, temporal_cosine = temporal_coherence(gate)
                gate_temporal_cosines[layer_index].append(temporal_cosine)
                values = _gradient_metrics(
                    bp["weight_gradient"][layer_index],
                    bp["weight_gradient"][layer_index],
                )
                alignment_rows.append(
                    {
                        "record_type": "weight_gradient",
                        "method": method_key,
                        "seed": seed,
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "layer": f"hidden_{layer_index + 1}",
                        **values,
                    }
                )
            continue

        for layer_index, layer_input in enumerate(hidden_inputs):
            capture = capture_layer_signals(
                model,
                feedback_bank,
                layer_index,
                layer_input,
                error,
                detach_temporal=True,
            )
            if spectral_filter is None:
                filtered = capture.delta
                values = {
                    "delta_norm": float(torch.linalg.vector_norm(capture.delta)),
                    "parallel_norm": float(torch.linalg.vector_norm(capture.delta)),
                    "tail_norm": 0.0,
                    "filtered_norm": float(torch.linalg.vector_norm(capture.delta)),
                    "parallel_tail_cosine": 0.0,
                    "delta_norm_ratio": 1.0,
                }
            else:
                filtered, values = spectral_filter.filter_tensor(
                    layer_index, capture.delta
                )
            method_cov[(layer_index, "timestep")].update(filtered)
            method_cov[(layer_index, "aggregated")].update(filtered.mean(dim=0))
            bp_delta = bp["delta"][layer_index]
            bp_cov[(layer_index, "timestep")].update(bp_delta)
            bp_cov[(layer_index, "aggregated")].update(bp_delta.mean(dim=0))
            gate_cov[layer_index].update(capture.gate)
            _coherence, temporal_cosine = temporal_coherence(capture.gate)
            gate_temporal_cosines[layer_index].append(temporal_cosine)
            method_weight = _weight_from_delta(filtered, layer_input)
            gradient_values = _gradient_metrics(
                method_weight, bp["weight_gradient"][layer_index]
            )
            alignment_rows.append(
                {
                    "record_type": "weight_gradient",
                    "method": method_key,
                    "seed": seed,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "layer": f"hidden_{layer_index + 1}",
                    **gradient_values,
                }
            )
            norm_rows.append(
                {
                    "record_type": "diagnostic_probe",
                    "method": method_key,
                    "seed": seed,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "layer": f"hidden_{layer_index + 1}",
                    **values,
                    "delta_gate_parity_relative": capture.parity_relative,
                }
            )

    geometry_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    method_bases: dict[tuple[int, str], Tensor] = {}
    bp_bases: dict[tuple[int, str], Tensor] = {}
    method_metrics: dict[tuple[int, str], dict[str, Any]] = {}
    bp_metrics: dict[tuple[int, str], dict[str, Any]] = {}
    for layer_index in range(layer_count):
        for mode in ("aggregated", "timestep"):
            method_basis, _values, metrics = pca_basis(
                method_cov[(layer_index, mode)], device
            )
            bp_basis, _bp_values, bp_values = pca_basis(
                bp_cov[(layer_index, mode)], device
            )
            method_bases[(layer_index, mode)] = method_basis.detach().cpu()
            bp_bases[(layer_index, mode)] = bp_basis.detach().cpu()
            method_metrics[(layer_index, mode)] = metrics
            bp_metrics[(layer_index, mode)] = bp_values
            geometry_rows.append(
                {
                    "method": method_key,
                    "seed": seed,
                    "epoch": epoch,
                    "layer": f"hidden_{layer_index + 1}",
                    "signal_type": (
                        "trained_bptt_credit" if is_bptt else "filtered_dfa_delta"
                    ),
                    "temporal_mode": mode,
                    **metrics,
                }
            )
            if not is_bptt:
                geometry_rows.append(
                    {
                        "method": method_key,
                        "seed": seed,
                        "epoch": epoch,
                        "layer": f"hidden_{layer_index + 1}",
                        "signal_type": "counterfactual_bptt_credit",
                        "temporal_mode": mode,
                        **bp_values,
                    }
                )
        gate_metrics, _s, _e, _c = spectrum_statistics(
            gate_cov[layer_index],
            algebraic_max_dim=dimensions[layer_index],
            device=device,
        )
        gate_rows.append(
            {
                "method": method_key,
                "seed": seed,
                "epoch": epoch,
                "layer": f"hidden_{layer_index + 1}",
                "temporal_gate_cosine": float(
                    torch.cat(gate_temporal_cosines[layer_index]).mean()
                ),
                "delta_timestep_r95": method_metrics[(layer_index, "timestep")]["r95"],
                "gate_r95": gate_metrics["r95"],
                "gate_entropy_rank": gate_metrics["entropy_rank"],
            }
        )

    for layer_index in range(layer_count):
        for mode in ("aggregated", "timestep"):
            method_r95 = int(method_metrics[(layer_index, mode)]["r95"])
            bp_r95 = int(bp_metrics[(layer_index, mode)]["r95"])
            candidates = [
                ("fixed_32", 32),
                ("fixed_64", 64),
                ("fixed_128", 128),
                ("min_r95", min(method_r95, bp_r95)),
            ]
            for source, rank in candidates:
                values = principal_subspace_metrics(
                    method_bases[(layer_index, mode)],
                    bp_bases[(layer_index, mode)],
                    rank,
                )
                alignment_rows.append(
                    {
                        "record_type": "credit_subspace",
                        "method": method_key,
                        "seed": seed,
                        "epoch": epoch,
                        "batch_index": None,
                        "layer": f"hidden_{layer_index + 1}",
                        "temporal_mode": mode,
                        "k_source": source,
                        "method_r95": method_r95,
                        "bp_r95": bp_r95,
                        **values,
                    }
                )
    return {
        "spectral_geometry": geometry_rows,
        "bp_alignment": alignment_rows,
        "gradient_norms": norm_rows,
        "gate_dynamics": gate_rows,
    }
