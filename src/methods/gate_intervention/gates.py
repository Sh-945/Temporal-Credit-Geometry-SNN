"""Causal gate interventions used only by Paper 1 Experiment 03."""

from __future__ import annotations

from enum import Enum
from math import gcd

import torch
from torch import Tensor


class GateMode(str, Enum):
    OFF = "off"
    ACTUAL = "actual"
    TEMPORAL_MEAN = "temporal_mean_normmatched"
    TEMPORAL_MEAN_UNMATCHED = "temporal_mean_unmatched"
    TIMESTEP_SHUFFLED = "timestep_shuffled"


def _mix_seed(seed: int, epoch: int, sample_index: int) -> int:
    """Stable 63-bit integer mix; independent of Python's salted hash."""

    mask = (1 << 64) - 1
    value = int(seed) & mask
    for item in (epoch, sample_index):
        value ^= (int(item) + 0x9E3779B97F4A7C15 + ((value << 6) & mask) + (value >> 2)) & mask
        value = (value * 0xBF58476D1CE4E5B9) & mask
        value ^= value >> 27
    return value & ((1 << 63) - 1)


def deterministic_time_permutation(
    time_steps: int, *, seed: int, epoch: int, sample_index: int
) -> Tensor:
    return deterministic_time_permutations(
        time_steps,
        seed=seed,
        epoch=epoch,
        sample_indices=torch.tensor([sample_index]),
    )[0]


def deterministic_time_permutations(
    time_steps: int,
    *,
    seed: int,
    epoch: int,
    sample_indices: Tensor,
) -> Tensor:
    """Vector-friendly deterministic affine permutations, one per sample.

    For every row, ``pi(t)=(a*t+b) mod T`` with ``gcd(a,T)=1``.  This is a
    true permutation, is fully keyed by seed/epoch/source index, and avoids
    millions of per-sample RNG objects during formal G2 training.
    """

    time_steps = int(time_steps)
    coprime = [value for value in range(1, time_steps) if gcd(value, time_steps) == 1]
    if not coprime:
        return torch.zeros((sample_indices.numel(), time_steps), dtype=torch.long)
    rows = []
    steps = torch.arange(time_steps, dtype=torch.long)
    for sample_index in sample_indices.detach().cpu().tolist():
        mixed = _mix_seed(seed, epoch, int(sample_index))
        multiplier = coprime[mixed % len(coprime)]
        offset = (mixed // len(coprime)) % time_steps
        rows.append((multiplier * steps + offset) % time_steps)
    return torch.stack(rows, dim=0)


def apply_gate_intervention(
    gate: Tensor,
    mode: GateMode | str,
    *,
    seed: int,
    epoch: int,
    sample_indices: Tensor,
    time_permutations: Tensor | None = None,
    epsilon: float = 1e-12,
) -> Tensor:
    """Return a detached replacement gate without changing the forward pass."""

    selected = GateMode(mode)
    actual = gate.detach()
    if selected in {GateMode.OFF, GateMode.ACTUAL}:
        return actual
    if selected in {GateMode.TEMPORAL_MEAN, GateMode.TEMPORAL_MEAN_UNMATCHED}:
        mean_gate = actual.mean(dim=0, keepdim=True).expand_as(actual)
        if selected == GateMode.TEMPORAL_MEAN_UNMATCHED:
            return mean_gate
        scale = torch.linalg.vector_norm(actual) / (
            torch.linalg.vector_norm(mean_gate) + float(epsilon)
        )
        return mean_gate * scale
    if selected == GateMode.TIMESTEP_SHUFFLED:
        if sample_indices.numel() != actual.shape[1]:
            raise ValueError("sample_indices must contain one index per batch sample")
        permutations = time_permutations
        if permutations is None:
            permutations = deterministic_time_permutations(
                actual.shape[0],
                seed=seed,
                epoch=epoch,
                sample_indices=sample_indices,
            )
        permutations = permutations.to(actual.device)
        if tuple(permutations.shape) != (actual.shape[1], actual.shape[0]):
            raise ValueError("time_permutations must be [B,T]")
        gather_index = permutations.transpose(0, 1)
        while gather_index.ndim < actual.ndim:
            gather_index = gather_index.unsqueeze(-1)
        return torch.gather(actual, 0, gather_index.expand_as(actual))
    raise AssertionError(selected)
