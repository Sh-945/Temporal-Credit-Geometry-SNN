"""Fully connected SNN with replayable hidden layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from models.neurons import LIFCell


class SpikingLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, neuron: dict[str, Any]):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.neuron = LIFCell(**neuron)

    def forward(self, sequence: Tensor, detach_temporal: bool = False) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"linear sequence must be [T,B,D], got {tuple(sequence.shape)}")
        time, batch = sequence.shape[:2]
        current = self.linear(sequence.reshape(time * batch, -1)).reshape(
            time, batch, self.linear.out_features
        )
        return self.neuron(current, detach_temporal=detach_temporal)


class FCSpikingNetwork(nn.Module):
    architecture = "fc"

    def __init__(
        self,
        input_features: int,
        hidden_features: list[int],
        num_classes: int,
        neuron: dict[str, Any],
    ) -> None:
        super().__init__()
        if not hidden_features:
            raise ValueError("FC model needs at least one hidden layer")
        dimensions = [input_features, *hidden_features]
        self.hidden_layers = nn.ModuleList(
            SpikingLinear(dimensions[i], dimensions[i + 1], neuron)
            for i in range(len(hidden_features))
        )
        self.readout = SpikingLinear(hidden_features[-1], num_classes, neuron)
        self.num_classes = int(num_classes)

    @property
    def hidden_specs(self) -> list[dict[str, Any]]:
        return [
            {"kind": "fc", "dimension": layer.linear.out_features}
            for layer in self.hidden_layers
        ]

    def _flatten(self, sequence: Tensor) -> Tensor:
        if sequence.ndim < 3:
            raise ValueError(f"input must be time-major [T,B,...], got {tuple(sequence.shape)}")
        return sequence.flatten(start_dim=2)

    def forward_with_cache(
        self, sequence: Tensor, detach_temporal: bool = False
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        hidden_inputs: list[Tensor] = []
        x = self._flatten(sequence)
        for layer in self.hidden_layers:
            hidden_inputs.append(x)
            x = layer(x, detach_temporal=detach_temporal)
        readout_input = x
        output_spikes = self.readout(x, detach_temporal=detach_temporal)
        return output_spikes, hidden_inputs, readout_input

    def forward(self, sequence: Tensor, detach_temporal: bool = False) -> Tensor:
        return self.forward_with_cache(sequence, detach_temporal)[0]

    def replay_hidden(
        self, index: int, sequence: Tensor, detach_temporal: bool
    ) -> Tensor:
        return self.hidden_layers[index](sequence, detach_temporal=detach_temporal)

    def replay_readout(self, sequence: Tensor) -> Tensor:
        return self.readout(sequence, detach_temporal=False)

    def hidden_weight_matrix(self, index: int) -> Tensor:
        return self.hidden_layers[index].linear.weight
