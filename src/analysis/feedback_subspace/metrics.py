"""Streaming centered-covariance and effective-rank calculations.

The diagnostic never needs to persist an ``N*T*D`` signal tensor.  Each
mini-batch is reduced to a mean vector and centered scatter matrix, then the
statistics are merged with the parallel/Welford covariance identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


FLOAT32_EPS = float(torch.finfo(torch.float32).eps)


@dataclass
class OnlineCovariance:
    """Centered scatter sufficient statistics for row observations."""

    ambient_dim: int
    unit_direction: bool = False
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        self.count = 0
        self.mean = torch.zeros(self.ambient_dim, dtype=torch.float64)
        self.m2 = torch.zeros(
            (self.ambient_dim, self.ambient_dim), dtype=torch.float64
        )

    @torch.no_grad()
    def update(self, observations: Tensor) -> None:
        x = observations.detach().reshape(-1, self.ambient_dim)
        if x.numel() == 0:
            return
        x = x.to(dtype=torch.float32)
        if self.unit_direction:
            x = x / (torch.linalg.vector_norm(x, dim=1, keepdim=True) + self.epsilon)
        chunk_count = int(x.shape[0])
        chunk_mean_device = x.mean(dim=0)
        centered = x - chunk_mean_device
        chunk_m2 = (centered.transpose(0, 1) @ centered).cpu().to(torch.float64)
        chunk_mean = chunk_mean_device.cpu().to(torch.float64)

        if self.count == 0:
            self.count = chunk_count
            self.mean.copy_(chunk_mean)
            self.m2.copy_(chunk_m2)
            return

        total = self.count + chunk_count
        difference = chunk_mean - self.mean
        self.m2.add_(chunk_m2)
        self.m2.add_(
            torch.outer(difference, difference)
            * (self.count * chunk_count / float(total))
        )
        self.mean.add_(difference * (chunk_count / float(total)))
        self.count = total

    def repeated(self, repeats: int) -> "OnlineCovariance":
        """Return statistics for repeating every row ``repeats`` times."""

        if repeats < 1:
            raise ValueError("repeats must be positive")
        result = OnlineCovariance(
            self.ambient_dim,
            unit_direction=self.unit_direction,
            epsilon=self.epsilon,
        )
        result.count = self.count * repeats
        result.mean.copy_(self.mean)
        result.m2.copy_(self.m2 * repeats)
        return result

    def uncentered_scatter(self) -> Tensor:
        return self.m2 + self.count * torch.outer(self.mean, self.mean)


class DualCovariance:
    """Raw-amplitude and row-normalized direction statistics."""

    def __init__(self, ambient_dim: int) -> None:
        self.raw = OnlineCovariance(ambient_dim, unit_direction=False)
        self.direction = OnlineCovariance(ambient_dim, unit_direction=True)

    @torch.no_grad()
    def update(self, observations: Tensor) -> None:
        self.raw.update(observations)
        self.direction.update(observations)

    def repeated(self, repeats: int) -> "DualCovariance":
        result = DualCovariance(self.raw.ambient_dim)
        result.raw = self.raw.repeated(repeats)
        result.direction = self.direction.repeated(repeats)
        return result


def _descending_eigenvalues(scatter: Tensor, device: torch.device) -> np.ndarray:
    symmetric = 0.5 * (scatter + scatter.transpose(0, 1))
    try:
        values = torch.linalg.eigvalsh(
            symmetric.to(device=device, dtype=torch.float32)
        )
    except RuntimeError:
        values = torch.linalg.eigvalsh(symmetric.to(dtype=torch.float64))
    values = values.clamp_min_(0).flip(0).cpu().to(torch.float64).numpy()
    return values


def _numerical_rank(singular_values: np.ndarray, rows: int, columns: int) -> int:
    if singular_values.size == 0 or singular_values[0] <= 0:
        return 0
    # Spectra are obtained from a float32 scatter matrix.  Round-off appears
    # in eigenvalue (sigma^2) space, so its resolvable singular-value floor is
    # sqrt(max(N,D)*eps)*sigma_max rather than the direct-SVD tolerance.
    tolerance = (
        max(rows, columns) * FLOAT32_EPS
    ) ** 0.5 * float(singular_values[0])
    return int(np.count_nonzero(singular_values > tolerance))


def uncentered_numerical_rank(
    accumulator: OnlineCovariance, device: torch.device
) -> int:
    eigenvalues = _descending_eigenvalues(accumulator.uncentered_scatter(), device)
    singular_values = np.sqrt(eigenvalues)
    return _numerical_rank(
        singular_values, accumulator.count, accumulator.ambient_dim
    )


def spectrum_statistics(
    accumulator: OnlineCovariance,
    *,
    algebraic_max_dim: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Return rank metrics and the complete centered singular spectrum."""

    eigenvalues = _descending_eigenvalues(accumulator.m2, device)
    singular_values = np.sqrt(eigenvalues)
    total = float(eigenvalues.sum())
    if total > 0.0:
        energy = eigenvalues / total
        cumulative = np.cumsum(energy)

        def energy_rank(threshold: float) -> int:
            return int(np.searchsorted(cumulative, threshold, side="left") + 1)

        positive = energy[energy > 0]
        entropy_rank = float(np.exp(-(positive * np.log(positive + 1e-30)).sum()))
        participation_ratio = float(total * total / np.square(eigenvalues).sum())
        stable_rank = float(total / eigenvalues[0])
        ranks = {
            "r50": energy_rank(0.50),
            "r80": energy_rank(0.80),
            "r90": energy_rank(0.90),
            "r95": energy_rank(0.95),
            "r99": energy_rank(0.99),
        }
    else:
        energy = np.zeros_like(eigenvalues)
        cumulative = np.zeros_like(eigenvalues)
        entropy_rank = participation_ratio = stable_rank = 0.0
        ranks = {name: 0 for name in ("r50", "r80", "r90", "r95", "r99")}

    ambient = accumulator.ambient_dim
    maximum = max(1, int(algebraic_max_dim))
    metrics: dict[str, Any] = {
        "observations": accumulator.count,
        "ambient_dim": ambient,
        "algebraic_max_dim": int(algebraic_max_dim),
        "numerical_rank": _numerical_rank(
            singular_values, accumulator.count, ambient
        ),
        **ranks,
        "entropy_rank": entropy_rank,
        "participation_ratio": participation_ratio,
        "stable_rank": stable_rank,
        "r95_over_ambient": ranks["r95"] / float(ambient),
        "r95_over_algebraic_max": ranks["r95"] / float(maximum),
        "entropy_over_ambient": entropy_rank / float(ambient),
        "participation_over_ambient": participation_ratio / float(ambient),
    }
    return metrics, singular_values, energy, cumulative
