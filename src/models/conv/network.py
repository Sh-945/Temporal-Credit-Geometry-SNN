"""Convolutional SNN with channel-level local teaching signals."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.fc.network import SpikingLinear
from models.neurons import LIFCell


class SpikingConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neuron: dict[str, Any],
        kernel_size: int = 3,
        pool: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.neuron = LIFCell(**neuron)
        self.pool = int(pool)

    def forward(
        self, sequence: Tensor, detach_temporal: bool = False
    ) -> tuple[Tensor, Tensor]:
        if sequence.ndim != 5:
            raise ValueError(
                f"convolution sequence must be [T,B,C,H,W], got {tuple(sequence.shape)}"
            )
        time, batch = sequence.shape[:2]
        current = self.conv(sequence.flatten(0, 1)).reshape(
            time, batch, self.conv.out_channels, sequence.shape[-2], sequence.shape[-1]
        )
        spikes = self.neuron(current, detach_temporal=detach_temporal)
        if self.pool > 1:
            pooled = F.avg_pool2d(spikes.flatten(0, 1), self.pool).reshape(
                time, batch, self.conv.out_channels, spikes.shape[-2] // self.pool,
                spikes.shape[-1] // self.pool
            )
        else:
            pooled = spikes
        return spikes, pooled


class ConvSpikingNetwork(nn.Module):
    architecture = "conv"

    def __init__(
        self,
        input_shape: list[int],
        channels: list[int],
        hidden_features: int,
        num_classes: int,
        neuron: dict[str, Any],
        pool: int = 2,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or not channels:
            raise ValueError("conv input_shape must be [C,H,W] and channels cannot be empty")
        channel_sizes = [input_shape[0], *channels]
        self.conv_layers = nn.ModuleList(
            SpikingConvBlock(
                channel_sizes[i], channel_sizes[i + 1], neuron=neuron, pool=pool
            )
            for i in range(len(channels))
        )
        reduction = pool ** len(channels)
        final_h = input_shape[1] // reduction
        final_w = input_shape[2] // reduction
        if final_h < 1 or final_w < 1:
            raise ValueError("pooling reduces the configured input below one pixel")
        flattened = channels[-1] * final_h * final_w
        self.fc_hidden = SpikingLinear(flattened, hidden_features, neuron)
        self.readout = SpikingLinear(hidden_features, num_classes, neuron)
        self.num_classes = int(num_classes)

    @property
    def hidden_specs(self) -> list[dict[str, Any]]:
        specs = [
            {"kind": "conv", "dimension": layer.conv.out_channels}
            for layer in self.conv_layers
        ]
        specs.append({"kind": "fc", "dimension": self.fc_hidden.linear.out_features})
        return specs

    def forward_with_cache(
        self, sequence: Tensor, detach_temporal: bool = False
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        hidden_inputs: list[Tensor] = []
        x = sequence
        for layer in self.conv_layers:
            hidden_inputs.append(x)
            _, x = layer(x, detach_temporal=detach_temporal)
        x = x.flatten(start_dim=2)
        hidden_inputs.append(x)
        x = self.fc_hidden(x, detach_temporal=detach_temporal)
        readout_input = x
        output_spikes = self.readout(x, detach_temporal=detach_temporal)
        return output_spikes, hidden_inputs, readout_input

    def forward(self, sequence: Tensor, detach_temporal: bool = False) -> Tensor:
        return self.forward_with_cache(sequence, detach_temporal)[0]

    def replay_hidden(
        self, index: int, sequence: Tensor, detach_temporal: bool
    ) -> Tensor:
        if index < len(self.conv_layers):
            spikes, _ = self.conv_layers[index](
                sequence, detach_temporal=detach_temporal
            )
            return spikes
        if index == len(self.conv_layers):
            return self.fc_hidden(sequence, detach_temporal=detach_temporal)
        raise IndexError(index)

    def replay_readout(self, sequence: Tensor) -> Tensor:
        return self.readout(sequence, detach_temporal=False)

    def hidden_weight_matrix(self, index: int) -> Tensor:
        if index < len(self.conv_layers):
            weight = self.conv_layers[index].conv.weight
            return weight.flatten(start_dim=1)
        if index == len(self.conv_layers):
            return self.fc_hidden.linear.weight
        raise IndexError(index)
