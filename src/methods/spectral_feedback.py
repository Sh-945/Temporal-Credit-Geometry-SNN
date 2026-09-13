"""Causal post-gate spectral regulation for Experiment 03.

The filter is attached to the replayed hidden *current* gradient.  Autograd has
already multiplied the DFA teaching vector by the production surrogate gate at
that point, while the linear layer has not yet formed its weight gradient.
Thus this module changes only the requested post-gate teaching geometry.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
import warnings

import torch
from torch import Tensor, nn


@dataclass
class ProjectionDiagnostics:
    layer_index: int
    delta_norm: float
    parallel_norm: float
    tail_norm: float
    filtered_norm: float
    parallel_tail_cosine: float
    delta_norm_ratio: float
    dense_weight_gradient_norm: float
    filtered_weight_gradient_norm: float
    weight_gradient_norm_ratio: float


class SpectralProjector:
    """Orthonormal feature-space projector with two-band tail attenuation."""

    def __init__(self, basis: Tensor, alpha: float) -> None:
        if basis.ndim != 2:
            raise ValueError("basis must be [D,k]")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.basis = basis.detach()
        self.alpha = float(alpha)

    def components(self, delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if delta.shape[-1] != self.basis.shape[0]:
            raise ValueError(
                f"delta feature dimension {delta.shape[-1]} does not match basis {self.basis.shape[0]}"
            )
        basis = self.basis.to(device=delta.device, dtype=delta.dtype)
        parallel = (delta @ basis) @ basis.transpose(0, 1)
        tail = delta - parallel
        filtered = parallel + self.alpha * tail
        return parallel, tail, filtered

    def __call__(self, delta: Tensor) -> Tensor:
        return self.components(delta)[2]


class SoftSpectralFilter:
    """Layer-wise bases plus autograd hooks for the real FC update path."""

    def __init__(
        self,
        hidden_dimensions: list[int],
        *,
        alpha: float,
        norm_matched: bool = False,
    ) -> None:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.hidden_dimensions = [int(value) for value in hidden_dimensions]
        self.alpha = float(alpha)
        self.norm_matched = bool(norm_matched)
        self.bases: list[Tensor | None] = [None for _ in hidden_dimensions]
        self.basis_source_epochs: list[int | None] = [None for _ in hidden_dimensions]
        self._batch_diagnostics: list[ProjectionDiagnostics] = []

    @property
    def bypasses_projection(self) -> bool:
        # This branch is deliberately exact: alpha=1 never installs a hook.
        return self.alpha == 1.0 and not self.norm_matched

    def set_basis(self, layer_index: int, basis: Tensor, source_epoch: int) -> None:
        dimension = self.hidden_dimensions[layer_index]
        if basis.ndim != 2 or basis.shape[0] != dimension:
            raise ValueError(
                f"layer {layer_index} basis must be [{dimension},k], got {tuple(basis.shape)}"
            )
        gram = basis.detach().to(torch.float64).T @ basis.detach().to(torch.float64)
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
            raise ValueError("basis columns are not orthonormal")
        self.bases[layer_index] = basis.detach().cpu().to(torch.float32).contiguous()
        self.basis_source_epochs[layer_index] = int(source_epoch)

    def clear_bases(self) -> None:
        self.bases = [None for _ in self.hidden_dimensions]
        self.basis_source_epochs = [None for _ in self.hidden_dimensions]

    def filter_tensor(self, layer_index: int, delta: Tensor) -> tuple[Tensor, dict[str, float]]:
        basis = self.bases[layer_index]
        if basis is None or self.bypasses_projection:
            norm = float(torch.linalg.vector_norm(delta))
            return delta, {
                "delta_norm": norm,
                "parallel_norm": norm,
                "tail_norm": 0.0,
                "filtered_norm": norm,
                "parallel_tail_cosine": 0.0,
                "delta_norm_ratio": 1.0,
            }
        projector = SpectralProjector(basis, self.alpha)
        parallel, tail, filtered = projector.components(delta)
        delta_norm = torch.linalg.vector_norm(delta)
        parallel_norm = torch.linalg.vector_norm(parallel)
        tail_norm = torch.linalg.vector_norm(tail)
        filtered_norm = torch.linalg.vector_norm(filtered)
        if self.norm_matched:
            filtered = filtered * (delta_norm / (filtered_norm + 1e-30))
            filtered_norm = torch.linalg.vector_norm(filtered)
        cosine = torch.sum(parallel * tail) / (parallel_norm * tail_norm + 1e-30)
        return filtered, {
            "delta_norm": float(delta_norm),
            "parallel_norm": float(parallel_norm),
            "tail_norm": float(tail_norm),
            "filtered_norm": float(filtered_norm),
            "parallel_tail_cosine": float(cosine),
            "delta_norm_ratio": float(filtered_norm / (delta_norm + 1e-30)),
        }

    @contextmanager
    def intercept_linear(self, layer_index: int, linear: nn.Linear) -> Iterator[None]:
        """Filter current gradients after the surrogate gate and before dW."""

        if self.bypasses_projection or self.bases[layer_index] is None:
            yield
            return

        def attach(_module, inputs, output: Tensor) -> None:
            linear_input = inputs[0].detach()

            def regulate(dense_gradient: Tensor) -> Tensor:
                filtered, values = self.filter_tensor(layer_index, dense_gradient)
                dense_weight = dense_gradient.reshape(-1, dense_gradient.shape[-1]).T @ linear_input.reshape(-1, linear_input.shape[-1])
                filtered_weight = filtered.reshape(-1, filtered.shape[-1]).T @ linear_input.reshape(-1, linear_input.shape[-1])
                dense_weight_norm = torch.linalg.vector_norm(dense_weight)
                filtered_weight_norm = torch.linalg.vector_norm(filtered_weight)
                self._batch_diagnostics.append(
                    ProjectionDiagnostics(
                        layer_index=layer_index,
                        delta_norm=values["delta_norm"],
                        parallel_norm=values["parallel_norm"],
                        tail_norm=values["tail_norm"],
                        filtered_norm=values["filtered_norm"],
                        parallel_tail_cosine=values["parallel_tail_cosine"],
                        delta_norm_ratio=values["delta_norm_ratio"],
                        dense_weight_gradient_norm=float(dense_weight_norm),
                        filtered_weight_gradient_norm=float(filtered_weight_norm),
                        weight_gradient_norm_ratio=float(
                            filtered_weight_norm / (dense_weight_norm + 1e-30)
                        ),
                    )
                )
                return filtered

            output.register_hook(regulate)

        handle = linear.register_forward_hook(attach)
        try:
            yield
        finally:
            handle.remove()

    def drain_batch_diagnostics(self) -> list[dict[str, Any]]:
        rows = [vars(item) for item in self._batch_diagnostics]
        self._batch_diagnostics.clear()
        return rows

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "norm_matched": self.norm_matched,
            "hidden_dimensions": self.hidden_dimensions,
            "bases": self.bases,
            "basis_source_epochs": self.basis_source_epochs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if [int(value) for value in state["hidden_dimensions"]] != self.hidden_dimensions:
            raise ValueError("spectral-filter hidden dimensions do not match")
        if float(state["alpha"]) != self.alpha or bool(state["norm_matched"]) != self.norm_matched:
            raise ValueError("spectral-filter configuration does not match state")
        self.bases = [
            None if basis is None else basis.detach().cpu().to(torch.float32)
            for basis in state["bases"]
        ]
        self.basis_source_epochs = list(state["basis_source_epochs"])


class DynamicBasisTracker:
    """Causal epoch-end PCA/random basis refresh state."""

    def __init__(self, hidden_dimensions: list[int], energy_threshold: float = 0.95) -> None:
        if not 0.0 < float(energy_threshold) <= 1.0:
            raise ValueError("energy threshold must be in (0,1]")
        self.hidden_dimensions = [int(value) for value in hidden_dimensions]
        self.energy_threshold = float(energy_threshold)
        self.previous_bases: list[Tensor | None] = [None for _ in hidden_dimensions]

    @staticmethod
    def energy_rank(eigenvalues: Tensor, threshold: float) -> int:
        values = eigenvalues.detach().to(torch.float64).clamp_min_(0)
        total = values.sum()
        if float(total) <= 0:
            return 1
        cumulative = torch.cumsum(values / total, dim=0)
        target = torch.tensor(
            threshold, dtype=cumulative.dtype, device=cumulative.device
        )
        return int(torch.searchsorted(cumulative, target).item() + 1)

    def learned_basis(
        self,
        scatter: Tensor,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, int]:
        symmetric = 0.5 * (scatter + scatter.T)
        gpu_matrix = symmetric.to(device=device, dtype=torch.float32)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(gpu_matrix)
        except RuntimeError as error:
            if "linalg.eigh" not in str(error):
                raise
            warnings.warn(
                "float32 device eigh did not converge; retrying the same "
                "symmetric scatter matrix with CPU float64 eigh",
                RuntimeWarning,
                stacklevel=2,
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(
                symmetric.detach().cpu().to(torch.float64)
            )
        eigenvalues = eigenvalues.flip(0).clamp_min_(0)
        eigenvectors = eigenvectors.flip(1).contiguous()
        rank = self.energy_rank(eigenvalues, self.energy_threshold)
        return (
            eigenvectors[:, :rank].detach().cpu().to(torch.float32),
            eigenvalues.detach().cpu().to(torch.float64),
            rank,
        )

    def random_basis(self, layer_index: int, rank: int, random_seed: int) -> Tensor:
        dimension = self.hidden_dimensions[layer_index]
        if not 1 <= int(rank) <= dimension:
            raise ValueError("random rank outside layer dimension")
        generator = torch.Generator().manual_seed(int(random_seed))
        matrix = torch.randn(dimension, int(rank), generator=generator)
        q, r = torch.linalg.qr(matrix, mode="reduced")
        signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
        return (q * signs.unsqueeze(0)).to(torch.float32)
