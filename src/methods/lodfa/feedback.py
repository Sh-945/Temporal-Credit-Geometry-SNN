"""Actual low-rank feedback B = B_U diag(b_s) B_V^T."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def select_energy_rank(coefficients: Tensor, threshold: float) -> int:
    """Smallest rank whose squared-coefficient energy reaches ``threshold``."""

    if coefficients.ndim != 1 or coefficients.numel() == 0:
        raise ValueError("energy coefficients must be a non-empty vector")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("energy threshold must be in (0, 1]")
    energy = coefficients.detach().abs().square()
    total = energy.sum()
    if total <= torch.finfo(energy.dtype).eps:
        return 1
    descending = energy.sort(descending=True).values
    cumulative = descending.cumsum(0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(threshold, device=energy.device)).item() + 1)


def _orthogonal_columns(rows: int, columns: int) -> Tensor:
    matrix = torch.randn(rows, columns)
    if rows >= columns:
        return torch.linalg.qr(matrix, mode="reduced").Q
    # This branch is normally prevented by max_rank validation.
    return torch.linalg.qr(matrix.transpose(0, 1), mode="reduced").Q.transpose(0, 1)


class LowRankFeedback(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        error_dim: int,
        rank: int,
        trainable: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        maximum = min(hidden_dim, error_dim)
        if not 1 <= rank <= maximum:
            raise ValueError(f"rank {rank} must be in [1, {maximum}]")
        bu = _orthogonal_columns(hidden_dim, rank)
        bv = _orthogonal_columns(error_dim, rank)
        bs = torch.full((rank,), float(scale) / rank**0.5)
        self.B_U = nn.Parameter(bu, requires_grad=trainable)
        self.B_S = nn.Parameter(bs, requires_grad=trainable)
        self.B_V = nn.Parameter(bv, requires_grad=trainable)
        self.register_buffer("_active_rank", torch.tensor(rank, dtype=torch.long))
        self.maximum_rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.error_dim = int(error_dim)
        self.trainable_feedback = bool(trainable)

    @property
    def active_rank(self) -> int:
        return int(self._active_rank.item())

    def set_active_rank(self, rank: int) -> None:
        if not 1 <= rank <= self.maximum_rank:
            raise ValueError(f"active rank {rank} must be in [1, {self.maximum_rank}]")
        self._active_rank.fill_(int(rank))

    @torch.no_grad()
    def update_rank_from_energy(self, threshold: float, non_increasing: bool = True) -> int:
        current = self.active_rank
        order = self.B_S[:current].abs().argsort(descending=True)
        self.B_U[:, :current].copy_(self.B_U[:, :current][:, order])
        self.B_S[:current].copy_(self.B_S[:current][order])
        self.B_V[:, :current].copy_(self.B_V[:, :current][:, order])
        chosen = select_energy_rank(self.B_S[:current], threshold)
        if non_increasing:
            chosen = min(chosen, current)
        self.set_active_rank(chosen)
        return chosen

    def forward(self, error: Tensor) -> Tensor:
        if error.ndim != 2 or error.shape[1] != self.error_dim:
            raise ValueError(
                f"error must be [B,{self.error_dim}], got {tuple(error.shape)}"
            )
        rank = self.active_rank
        code = error @ self.B_V[:, :rank]
        code = code * self.B_S[:rank]
        return code @ self.B_U[:, :rank].transpose(0, 1)

    def materialize(self) -> Tensor:
        rank = self.active_rank
        return (
            self.B_U[:, :rank]
            @ torch.diag(self.B_S[:rank])
            @ self.B_V[:, :rank].transpose(0, 1)
        )
