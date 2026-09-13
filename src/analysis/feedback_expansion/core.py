"""Numerical primitives for Experiment 01B.

All feature-space bases are fitted from centered ``basis_fit`` sufficient
statistics.  Consequences of those bases are evaluated against independent
``basis_eval`` scatter matrices; no raw probe tensor is persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics


@dataclass
class PCASummary:
    basis: Tensor
    eigenvalues: Tensor
    singular_values: np.ndarray
    explained_energy: np.ndarray
    cumulative_energy: np.ndarray
    metrics: dict[str, Any]


def merge_covariances(items: list[OnlineCovariance]) -> OnlineCovariance:
    """Merge independent Welford accumulators without replaying observations."""

    if not items:
        raise ValueError("at least one accumulator is required")
    result = OnlineCovariance(items[0].ambient_dim)
    for item in items:
        if item.ambient_dim != result.ambient_dim:
            raise ValueError("ambient dimensions do not match")
        if item.count == 0:
            continue
        if result.count == 0:
            result.count = item.count
            result.mean.copy_(item.mean)
            result.m2.copy_(item.m2)
            continue
        total = result.count + item.count
        difference = item.mean - result.mean
        result.m2.add_(item.m2)
        result.m2.add_(
            torch.outer(difference, difference)
            * (result.count * item.count / float(total))
        )
        result.mean.add_(difference * (item.count / float(total)))
        result.count = total
    return result


def pca_from_covariance(
    accumulator: OnlineCovariance,
    device: torch.device,
    *,
    max_basis: int = 256,
    algebraic_max_dim: int | None = None,
) -> PCASummary:
    """Compute an exact feature-space eigendecomposition of centered scatter."""

    symmetric = 0.5 * (accumulator.m2 + accumulator.m2.T)
    try:
        values, vectors = torch.linalg.eigh(
            symmetric.to(device=device, dtype=torch.float32)
        )
    except RuntimeError:
        values, vectors = torch.linalg.eigh(symmetric)
        values = values.to(device=device, dtype=torch.float32)
        vectors = vectors.to(device=device, dtype=torch.float32)
    values = values.flip(0).clamp_min_(0)
    vectors = vectors.flip(1).contiguous()
    metrics, singular, energy, cumulative = spectrum_statistics(
        accumulator,
        algebraic_max_dim=(
            accumulator.ambient_dim
            if algebraic_max_dim is None
            else int(algebraic_max_dim)
        ),
        device=device,
    )
    keep = min(int(max_basis), vectors.shape[1])
    return PCASummary(
        basis=vectors[:, :keep].detach().cpu().to(torch.float32),
        eigenvalues=values.detach().cpu().to(torch.float64),
        singular_values=singular,
        explained_energy=energy,
        cumulative_energy=cumulative,
        metrics=metrics,
    )


def scatter_about(accumulator: OnlineCovariance, center: Tensor) -> Tensor:
    """Return scatter of held-out rows around a fit-set center."""

    center64 = center.detach().cpu().to(torch.float64)
    difference = accumulator.mean - center64
    return accumulator.m2 + accumulator.count * torch.outer(difference, difference)


def energy_in_basis(scatter: Tensor, basis: Tensor, rank: int) -> float:
    """Fraction of total centered energy captured by a feature-space basis."""

    k = min(int(rank), basis.shape[1])
    if k < 1:
        return float("nan")
    v = basis[:, :k].detach().cpu().to(torch.float64)
    numerator = torch.sum(v * (scatter @ v))
    denominator = torch.trace(scatter)
    if float(denominator) <= 0:
        return float("nan")
    return float((numerator / denominator).clamp(0.0, 1.0))


def direction_energy(scatter: Tensor, basis: Tensor) -> np.ndarray:
    """Normalized energy along every supplied direction."""

    v = basis.detach().cpu().to(torch.float64)
    total = float(torch.trace(scatter))
    if total <= 0:
        return np.zeros(v.shape[1], dtype=np.float64)
    values = torch.sum(v * (scatter @ v), dim=0).clamp_min_(0) / total
    return values.numpy()


def principal_subspace_metrics(
    basis_a: Tensor, basis_b: Tensor, rank: int
) -> dict[str, float]:
    """Overlap plus a compact principal-angle summary in degrees."""

    k = min(int(rank), basis_a.shape[1], basis_b.shape[1])
    if k < 1:
        raise ValueError("rank must be positive")
    a = basis_a[:, :k].to(torch.float64)
    b = basis_b[:, :k].to(torch.float64)
    cosines = torch.linalg.svdvals(a.T @ b).clamp_(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosines)).cpu().numpy()
    return {
        "k": k,
        "overlap": float(cosines.square().mean()),
        "mean_principal_angle": float(np.mean(angles)),
        "median_principal_angle": float(np.median(angles)),
        "p90_principal_angle": float(np.quantile(angles, 0.90)),
        "max_principal_angle": float(np.max(angles)),
    }


def broadening_fields(summary: PCASummary) -> dict[str, Any]:
    """Absolute and normalized leading-spectrum diagnostics."""

    eigen = summary.eigenvalues.numpy()
    total = float(eigen.sum())
    row: dict[str, Any] = {
        **summary.metrics,
        "total_centered_energy": total,
        "top1_absolute_energy": float(eigen[0]) if eigen.size else 0.0,
    }
    for k in (1, 5, 10, 20, 32, 64, 128):
        absolute = float(eigen[:k].sum())
        row[f"top{k}_absolute_energy"] = absolute
        row[f"top{k}_energy_fraction"] = absolute / total if total > 0 else 0.0
    return row


def effective_neuron_count(utilization: Tensor) -> tuple[float, float]:
    """Return Shannon entropy and its exponential for neuron utilization."""

    values = utilization.detach().cpu().to(torch.float64).clamp_min_(0)
    total = values.sum()
    if float(total) <= 0:
        return 0.0, 0.0
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * torch.log(positive)).sum())
    return entropy, float(np.exp(entropy))


def mean_pair_cosine(vectors: Tensor, labels: Tensor, *, seed: int, pairs: int = 4096) -> tuple[float, float, int, int]:
    """Deterministically sample same/different-class vector-pair cosines."""

    x = vectors.detach().cpu().to(torch.float32)
    y = labels.detach().cpu().to(torch.long)
    x = x / (torch.linalg.vector_norm(x, dim=1, keepdim=True) + 1e-12)
    generator = torch.Generator().manual_seed(int(seed))
    first = torch.randint(len(x), (pairs * 4,), generator=generator)
    second = torch.randint(len(x), (pairs * 4,), generator=generator)
    distinct = first != second
    first, second = first[distinct], second[distinct]
    cosine = (x[first] * x[second]).sum(dim=1)
    same = y[first] == y[second]
    same_values = cosine[same][:pairs]
    different_values = cosine[~same][:pairs]
    return (
        float(same_values.mean()) if same_values.numel() else float("nan"),
        float(different_values.mean()) if different_values.numel() else float("nan"),
        int(same_values.numel()),
        int(different_values.numel()),
    )


def temporal_coherence(signal: Tensor) -> tuple[Tensor, Tensor]:
    """Per-sample temporal coherence and mean off-diagonal cosine."""

    x = signal.detach().to(torch.float32)
    summed = x.sum(dim=0)
    denominator = x.shape[0] * x.square().sum(dim=(0, 2))
    coherence = summed.square().sum(dim=1) / (denominator + 1e-30)
    normalized = x / (torch.linalg.vector_norm(x, dim=2, keepdim=True) + 1e-12)
    normalized_sum = normalized.sum(dim=0)
    time_steps = x.shape[0]
    pair_cosine = (
        normalized_sum.square().sum(dim=1) - time_steps
    ) / max(1, time_steps * (time_steps - 1))
    return coherence.cpu(), pair_cosine.cpu()
