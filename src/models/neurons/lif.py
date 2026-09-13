"""LIF dynamics and the surrogate derivative used by all model families."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold: Tensor, beta: float) -> Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.beta = beta
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        # Fast-sigmoid surrogate: dH/dx ~= 1 / (1 + beta |x|)^2.
        surrogate = 1.0 / (1.0 + beta * x.abs()).square()
        return grad_output * surrogate, None


def surrogate_spike(x: Tensor, beta: float = 10.0) -> Tensor:
    return _SurrogateSpike.apply(x, beta)


def lif_sequence(
    current: Tensor,
    decay: float,
    threshold: float,
    surrogate_beta: float,
    detach_temporal: bool = False,
) -> Tensor:
    """Integrate a time-major current tensor and return every spike step.

    ``detach_temporal=True`` implements the no-trace/pointwise local rule: the
    current time step still uses the surrogate derivative, while recurrent
    state is detached before the next step.
    """

    if current.ndim < 3:
        raise ValueError(f"expected [T,B,...] current, got {tuple(current.shape)}")
    membrane = torch.zeros_like(current[0])
    previous_spike = torch.zeros_like(current[0])
    spikes = []
    for step_current in current:
        membrane = decay * membrane * (1.0 - previous_spike) + step_current
        spike = surrogate_spike(membrane - threshold, surrogate_beta)
        spikes.append(spike)
        if detach_temporal:
            membrane = membrane.detach()
            previous_spike = spike.detach()
        else:
            previous_spike = spike
    return torch.stack(spikes, dim=0)


class LIFCell(nn.Module):
    def __init__(
        self,
        decay: float = 0.5,
        threshold: float = 1.0,
        surrogate_beta: float = 10.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError("LIF decay must be in [0, 1]")
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.surrogate_beta = float(surrogate_beta)

    def forward(self, current: Tensor, detach_temporal: bool = False) -> Tensor:
        return lif_sequence(
            current,
            decay=self.decay,
            threshold=self.threshold,
            surrogate_beta=self.surrogate_beta,
            detach_temporal=detach_temporal,
        )
