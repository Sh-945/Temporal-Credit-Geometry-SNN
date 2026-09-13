"""Exact replay capture for Experiment 01B.

The gate is obtained by autograd from the production replay itself.  It is
therefore the actual local derivative used by the configured neuron and
temporal rule, not a formula reimplemented by the diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LayerSignals:
    q: Tensor
    gate: Tensor
    delta: Tensor
    spikes: Tensor
    parity_max_abs: float
    parity_relative: float


def capture_layer_signals(
    model,
    feedback_bank,
    layer_index: int,
    layer_input: Tensor,
    output_error: Tensor,
    *,
    detach_temporal: bool,
) -> LayerSignals:
    """Capture exact ``q``, local derivative ``g``, and update signal ``delta``."""

    teaching = feedback_bank.project(layer_index, output_error).detach()
    linear = model.hidden_layers[layer_index].linear
    captured: list[Tensor] = []

    def remember_current(_module, _inputs, output: Tensor) -> None:
        captured.append(output)

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
    current = captured[0]
    gate = torch.autograd.grad(
        spikes,
        current,
        grad_outputs=torch.ones_like(spikes),
        retain_graph=True,
        create_graph=False,
    )[0].reshape_as(spikes)
    local_proxy = (spikes * teaching.unsqueeze(0)).mean()
    current_gradient = torch.autograd.grad(
        local_proxy, current, retain_graph=False, create_graph=False
    )[0].reshape_as(spikes)
    delta = current_gradient * current_gradient.numel()
    product = teaching.unsqueeze(0) * gate
    difference = delta - product
    denominator = float(torch.linalg.vector_norm(delta)) + 1e-30
    return LayerSignals(
        q=teaching,
        gate=gate.detach(),
        delta=delta.detach(),
        spikes=spikes.detach(),
        parity_max_abs=float(difference.abs().max()),
        parity_relative=float(torch.linalg.vector_norm(difference) / denominator),
    )


def shuffled_delta(signals: LayerSignals, shuffled_q: Tensor) -> Tensor:
    """Counterfactual label shuffle at an unchanged checkpoint and forward state."""

    if shuffled_q.shape != signals.q.shape:
        raise ValueError("shuffled q must align sample-for-sample with captured q")
    return signals.gate * shuffled_q.detach().unsqueeze(0)
