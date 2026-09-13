"""Dense fixed random direct-feedback matrix used by DFA/sDFA."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DenseFeedback(nn.Module):
    def __init__(self, hidden_dim: int, error_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        matrix = torch.randn(hidden_dim, error_dim)
        matrix = scale * matrix / max(1, error_dim) ** 0.5
        self.register_buffer("B", matrix)
        self.hidden_dim = int(hidden_dim)
        self.error_dim = int(error_dim)

    @property
    def active_rank(self) -> int:
        return min(self.hidden_dim, self.error_dim)

    def forward(self, error: Tensor) -> Tensor:
        if error.ndim != 2 or error.shape[1] != self.error_dim:
            raise ValueError(
                f"error must be [B,{self.error_dim}], got {tuple(error.shape)}"
            )
        return error @ self.B.transpose(0, 1)

    def materialize(self) -> Tensor:
        return self.B
