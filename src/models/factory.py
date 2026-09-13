"""Model construction from validated configuration."""

from __future__ import annotations

from math import prod
from typing import Any

from models.conv import ConvSpikingNetwork
from models.fc import FCSpikingNetwork


def build_model(config: dict[str, Any]):
    model_config = config["model"]
    neuron = {
        key: value for key, value in config["neuron"].items() if key != "type"
    }
    neuron_type = str(config["neuron"].get("type", "lif")).lower()
    if neuron_type == "if":
        neuron["decay"] = 1.0
    elif neuron_type != "lif":
        raise ValueError(f"unsupported neuron type: {neuron_type}")
    input_shape = list(model_config["input_shape"])
    architecture = model_config["architecture"]
    if architecture == "fc":
        return FCSpikingNetwork(
            input_features=prod(input_shape),
            hidden_features=list(model_config["hidden_features"]),
            num_classes=int(model_config["num_classes"]),
            neuron=neuron,
        )
    if architecture in {"conv", "convsmall", "vgg11"}:
        return ConvSpikingNetwork(
            input_shape=input_shape,
            channels=list(model_config["channels"]),
            hidden_features=int(model_config["hidden_features"]),
            num_classes=int(model_config["num_classes"]),
            neuron=neuron,
            pool=int(model_config.get("pool", 2)),
        )
    raise ValueError(f"unsupported model architecture: {architecture}")
