"""Per-hidden-layer feedback construction and structural penalties."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from methods.dfa import DenseFeedback
from methods.lodfa import LowRankFeedback
from methods.regularization import feedback_regularization


def _layer_ranks(
    specs: list[dict[str, Any]], error_dim: int, feedback: dict[str, Any]
) -> list[int]:
    explicit = feedback.get("layer_ranks")
    if explicit is not None:
        if len(explicit) != len(specs):
            raise ValueError("method.feedback.layer_ranks must match hidden layer count")
        values = [int(value) for value in explicit]
    else:
        requested = feedback.get("rank")
        fraction = feedback.get("rank_fraction")
        values = []
        for spec in specs:
            maximum = min(int(spec["dimension"]), error_dim)
            if requested is not None:
                values.append(min(int(requested), maximum))
            elif fraction is not None:
                values.append(max(1, min(maximum, round(maximum * float(fraction)))))
            else:
                values.append(maximum)
    for rank, spec in zip(values, specs):
        maximum = min(int(spec["dimension"]), error_dim)
        if not 1 <= rank <= maximum:
            raise ValueError(f"feedback rank {rank} outside [1,{maximum}]")
    return values


class FeedbackBank(nn.Module):
    def __init__(self, model: nn.Module, config: dict[str, Any]) -> None:
        super().__init__()
        method = config["method"]
        self.method_name = str(method["name"]).lower()
        self.error_dim = int(model.num_classes)
        self.feedback_config = method.get("feedback", {})
        self.layers = nn.ModuleList()

        if self.method_name == "bptt":
            return
        if self.method_name in {"dfa", "sdfa", "pipesdfa"}:
            for spec in model.hidden_specs:
                self.layers.append(
                    DenseFeedback(
                        int(spec["dimension"]),
                        self.error_dim,
                        scale=float(self.feedback_config.get("scale", 1.0)),
                    )
                )
            return
        if self.method_name not in {"lrsdfa", "lodfa"}:
            raise ValueError(f"unsupported training method: {self.method_name}")

        ranks = _layer_ranks(model.hidden_specs, self.error_dim, self.feedback_config)
        trainable = (
            bool(self.feedback_config.get("trainable", True))
            if self.method_name == "lodfa"
            else False
        )
        for spec, rank in zip(model.hidden_specs, ranks):
            self.layers.append(
                LowRankFeedback(
                    int(spec["dimension"]),
                    self.error_dim,
                    rank,
                    trainable=trainable,
                    scale=float(self.feedback_config.get("scale", 1.0)),
                )
            )

    @property
    def active_ranks(self) -> list[int]:
        return [layer.active_rank for layer in self.layers]

    def project(self, layer_index: int, error: Tensor) -> Tensor:
        return self.layers[layer_index](error)

    def regularization(
        self, model: nn.Module
    ) -> tuple[Tensor, dict[str, float]]:
        device = next(model.parameters()).device
        zero = torch.zeros((), device=device)
        if self.method_name != "lodfa":
            return zero, {"alignment": 0.0, "orthogonal": 0.0, "hoyer": 0.0}
        weights = self.feedback_config.get("loss_weights", {})
        totals = {"alignment": zero, "orthogonal": zero, "hoyer": zero}
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, LowRankFeedback):
                continue
            losses = feedback_regularization(layer, model.hidden_weight_matrix(index))
            for name, value in losses.items():
                totals[name] = totals[name] + value
        weighted = (
            float(weights.get("alignment", 0.0)) * totals["alignment"]
            + float(weights.get("orthogonal", 0.0)) * totals["orthogonal"]
            + float(weights.get("hoyer", 0.0)) * totals["hoyer"]
        )
        detached = {name: float(value.detach()) for name, value in totals.items()}
        return weighted, detached

    @torch.no_grad()
    def update_dynamic_ranks(self, epoch: int, total_epochs: int) -> list[int]:
        dynamic = self.feedback_config.get("dynamic_rank", {})
        if self.method_name != "lodfa" or not bool(dynamic.get("enabled", False)):
            return self.active_ranks
        warmup = int(dynamic.get("warmup_epochs", 0))
        interval = max(1, int(dynamic.get("update_interval", 1)))
        if epoch < warmup or (epoch - warmup) % interval != 0:
            return self.active_ranks
        threshold = float(dynamic.get("energy_threshold", 0.95))
        floor = max(1, int(dynamic.get("minimum_rank", 1)))
        for layer in self.layers:
            if isinstance(layer, LowRankFeedback):
                selected = layer.update_rank_from_energy(
                    threshold, non_increasing=True
                )
                layer.set_active_rank(max(floor, selected))
        return self.active_ranks
