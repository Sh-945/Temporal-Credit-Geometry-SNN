"""Pre-registered geometry metrics for Paper 1 Experiment 03."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from analysis.feedback_expansion.core import pca_from_covariance, temporal_coherence
from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics


def feature_signal(signal: Tensor) -> Tensor:
    """Map a matched current tensor to its declared feedback feature space."""

    if signal.ndim == 3:
        return signal
    if signal.ndim == 5:
        # Conv DFA feedback is channel-level and is broadcast over space.
        return signal.mean(dim=(-2, -1))
    raise ValueError(f"unsupported current tensor shape: {tuple(signal.shape)}")


@dataclass
class GeometryAccumulator:
    ambient_dim: int
    raw: dict[str, OnlineCovariance] = field(init=False)
    within: dict[str, dict[int, OnlineCovariance]] = field(init=False)
    coherence_sum: float = 0.0
    cosine_sum: float = 0.0
    temporal_samples: int = 0

    def __post_init__(self) -> None:
        self.raw = {
            "timestep": OnlineCovariance(self.ambient_dim),
            "aggregated": OnlineCovariance(self.ambient_dim),
        }
        self.within = {
            "timestep": defaultdict(lambda: OnlineCovariance(self.ambient_dim)),
            "aggregated": defaultdict(lambda: OnlineCovariance(self.ambient_dim)),
        }

    @torch.no_grad()
    def update(self, signal: Tensor, labels: Tensor) -> None:
        value = feature_signal(signal).detach()
        if value.shape[-1] != self.ambient_dim:
            raise ValueError("signal feature dimension does not match accumulator")
        aggregated = value.mean(dim=0)
        self.raw["timestep"].update(value)
        self.raw["aggregated"].update(aggregated)
        labels_cpu = labels.detach().cpu()
        for label in labels_cpu.unique().tolist():
            selected = labels_cpu == int(label)
            device_selected = selected.to(value.device)
            self.within["timestep"][int(label)].update(value[:, device_selected])
            self.within["aggregated"][int(label)].update(aggregated[device_selected])
        coherence, cosine = temporal_coherence(value)
        self.coherence_sum += float(coherence.sum())
        self.cosine_sum += float(cosine.sum())
        self.temporal_samples += int(coherence.numel())

    def temporal_metrics(self) -> dict[str, float]:
        count = max(1, self.temporal_samples)
        return {
            "temporal_coherence": self.coherence_sum / count,
            "mean_temporal_cosine": self.cosine_sum / count,
        }

    def rows(
        self,
        *,
        device: torch.device,
        prefix: dict[str, Any],
        probe_split: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Tensor]]:
        rows: list[dict[str, Any]] = []
        bases: dict[str, Tensor] = {}
        temporal = self.temporal_metrics()
        for mode in ("timestep", "aggregated"):
            metrics, _singular, _energy, _cumulative = spectrum_statistics(
                self.raw[mode], algebraic_max_dim=self.ambient_dim, device=device
            )
            summary = pca_from_covariance(
                self.raw[mode], device, max_basis=self.ambient_dim
            )
            bases[mode] = summary.basis
            rows.append(
                {
                    **prefix,
                    "probe_split": probe_split,
                    "temporal_mode": mode,
                    "residualization": "raw",
                    **metrics,
                    **temporal,
                }
            )
            within_metrics = spectrum_from_class_covariances(
                self.within[mode], self.ambient_dim, device
            )
            rows.append(
                {
                    **prefix,
                    "probe_split": probe_split,
                    "temporal_mode": mode,
                    "residualization": "within_class",
                    **within_metrics,
                    **temporal,
                }
            )
        return rows, bases


def spectrum_from_class_covariances(
    covariances: dict[int, OnlineCovariance],
    ambient_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    combined = OnlineCovariance(ambient_dim)
    combined.count = sum(item.count for item in covariances.values())
    if combined.count:
        combined.m2.copy_(sum((item.m2 for item in covariances.values()), torch.zeros_like(combined.m2)))
    metrics, _singular, _energy, _cumulative = spectrum_statistics(
        combined, algebraic_max_dim=ambient_dim, device=device
    )
    return metrics


def matrix_spectrum_metrics(matrix: Tensor) -> dict[str, float | int]:
    value = matrix.detach().to(torch.float64)
    if value.ndim > 2:
        value = value.flatten(start_dim=1)
    singular = torch.linalg.svdvals(value).cpu().numpy()
    energy = np.square(singular)
    total = float(energy.sum())
    if total <= 0:
        return {
            **{name: 0 for name in ("r50", "r80", "r90", "r95", "r99")},
            "entropy_rank": 0.0,
            "participation_ratio": 0.0,
            "stable_rank": 0.0,
        }
    probability = energy / total
    cumulative = np.cumsum(probability)
    result: dict[str, float | int] = {}
    for name, threshold in (("r50", .50), ("r80", .80), ("r90", .90), ("r95", .95), ("r99", .99)):
        result[name] = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    positive = probability[probability > 0]
    result["entropy_rank"] = float(np.exp(-(positive * np.log(positive)).sum()))
    result["participation_ratio"] = float(total * total / np.square(energy).sum())
    result["stable_rank"] = float(total / energy[0])
    return result


def tensor_cosine(first: Tensor, second: Tensor) -> float:
    a, b = first.detach().reshape(-1), second.detach().reshape(-1)
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-30))
