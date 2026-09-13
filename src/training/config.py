"""YAML loading, path resolution and command-line overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DATASETS = {"synthetic", "nmnist", "dvs_gesture", "ncaltech101", "shd"}
SUPPORTED_METHODS = {"bptt", "dfa", "sdfa", "lrsdfa", "lodfa", "pipesdfa"}


def _parse_override(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        dotted, raw_value = item.split("=", 1)
        parts = dotted.split(".")
        target: dict[str, Any] = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(f"unknown override path: {dotted}")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"unknown override key: {dotted}")
        target[parts[-1]] = _parse_override(raw_value)


def apply_smoke_settings(config: dict[str, Any]) -> None:
    architecture = config["model"]["architecture"]
    classes = int(config["model"]["num_classes"])
    config["data"].update(
        {
            "dataset": "synthetic",
            "root": ".",
            "time_steps": 4,
            "train_samples": 8,
            "test_samples": 4,
            "batch_size": 4,
            "num_workers": 0,
        }
    )
    if architecture == "fc":
        config["model"]["input_shape"] = [2, 8, 8]
        config["model"]["hidden_features"] = [16, 12]
    else:
        config["model"]["input_shape"] = [2, 8, 8]
        config["model"]["channels"] = [4, 6]
        config["model"]["hidden_features"] = 12
    feedback = config["method"].get("feedback", {})
    if "rank" in feedback and feedback["rank"] is not None:
        feedback["rank"] = min(int(feedback["rank"]), classes)
    if "layer_ranks" in feedback and feedback["layer_ranks"] is not None:
        feedback["layer_ranks"] = [
            min(int(rank), classes) for rank in feedback["layer_ranks"][:2]
        ]
        # Conv smoke models have three hidden layers.
        if architecture != "fc":
            while len(feedback["layer_ranks"]) < 3:
                feedback["layer_ranks"].append(min(4, classes))
    config["training"]["epochs"] = 1
    config["training"]["output_dir"] = (
        f"outputs/smoke/{config['experiment']['name']}"
    )


def _validate(config: dict[str, Any]) -> None:
    required = {"experiment", "data", "model", "neuron", "method", "training"}
    missing = required.difference(config)
    if missing:
        raise KeyError(f"missing top-level config sections: {sorted(missing)}")
    dataset = str(config["data"]["dataset"]).lower()
    method = str(config["method"]["name"]).lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method}")
    if int(config["data"]["time_steps"]) < 1:
        raise ValueError("data.time_steps must be positive")
    if int(config["model"]["num_classes"]) < 2:
        raise ValueError("model.num_classes must be at least two")
    if config["method"].get("local_loss", "proxy") not in {"proxy"}:
        raise ValueError("the final implementation supports only the patent proxy local loss")
    if config["method"].get("temporal_mode", "pointwise") not in {
        "pointwise",
        "local_bptt",
    }:
        raise ValueError("method.temporal_mode must be pointwise or local_bptt")


def load_config(
    path: str | Path, overrides: list[str] | None = None, smoke: bool = False
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError("configuration root must be a mapping")
    config = copy.deepcopy(loaded)
    config["_config_path"] = str(config_path)
    if overrides:
        apply_overrides(config, overrides)
    if smoke:
        apply_smoke_settings(config)
    _validate(config)
    return config


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPOSITORY_ROOT / path).resolve()
