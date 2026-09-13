"""Time-shared feed-forward ANN control for Paper 1 Experiment 03."""

from __future__ import annotations

from math import prod
from typing import Any

import torch
from torch import Tensor, nn


class TemporalANN(nn.Module):
    """Apply one shared MLP independently to every event-frame timestep.

    There is deliberately no membrane, leak, reset, threshold, or recurrent
    neuronal state.  The classifier emits per-timestep logits and the training
    objective consumes their temporal mean.
    """

    architecture = "temporal_ann"

    def __init__(
        self,
        input_shape: list[int],
        hidden_features: list[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        if not hidden_features:
            raise ValueError("TemporalANN needs at least one hidden layer")
        dimensions = [prod(input_shape), *map(int, hidden_features)]
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(hidden_features))
        )
        self.readout = nn.Linear(dimensions[-1], int(num_classes))
        self.num_classes = int(num_classes)

    @property
    def hidden_specs(self) -> list[dict[str, Any]]:
        return [
            {"kind": "fc", "dimension": layer.out_features}
            for layer in self.hidden_layers
        ]

    def forward_with_cache(
        self, sequence: Tensor
    ) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor]:
        if sequence.ndim < 3:
            raise ValueError(
                f"input must be time-major [T,B,...], got {tuple(sequence.shape)}"
            )
        time, batch = sequence.shape[:2]
        x = sequence.flatten(start_dim=2)
        hidden_inputs: list[Tensor] = []
        preactivations: list[Tensor] = []
        for layer in self.hidden_layers:
            hidden_inputs.append(x)
            current = layer(x.reshape(time * batch, -1)).reshape(
                time, batch, layer.out_features
            )
            preactivations.append(current)
            x = torch.relu(current)
        logits = self.readout(x.reshape(time * batch, -1)).reshape(
            time, batch, self.num_classes
        )
        return logits, hidden_inputs, preactivations, x

    def forward(self, sequence: Tensor) -> Tensor:
        return self.forward_with_cache(sequence)[0]

    def replay_hidden(self, index: int, sequence: Tensor) -> tuple[Tensor, Tensor]:
        if sequence.ndim != 3:
            raise ValueError("hidden input must be [T,B,D]")
        time, batch = sequence.shape[:2]
        layer = self.hidden_layers[index]
        current = layer(sequence.reshape(time * batch, -1)).reshape(
            time, batch, layer.out_features
        )
        return torch.relu(current), current

    def replay_readout(self, sequence: Tensor) -> Tensor:
        time, batch = sequence.shape[:2]
        return self.readout(sequence.reshape(time * batch, -1)).reshape(
            time, batch, self.num_classes
        )

    def hidden_weight_matrix(self, index: int) -> Tensor:
        return self.hidden_layers[index].weight

    def load_matched_snn_state(self, state: dict[str, Tensor]) -> None:
        """Copy shape-identical FC weights from a paired FC-SNN initialization."""

        translated: dict[str, Tensor] = {}
        for index in range(len(self.hidden_layers)):
            translated[f"hidden_layers.{index}.weight"] = state[
                f"hidden_layers.{index}.linear.weight"
            ]
            translated[f"hidden_layers.{index}.bias"] = state[
                f"hidden_layers.{index}.linear.bias"
            ]
        translated["readout.weight"] = state["readout.linear.weight"]
        translated["readout.bias"] = state["readout.linear.bias"]
        self.load_state_dict(translated, strict=True)
