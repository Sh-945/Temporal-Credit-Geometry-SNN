#!/usr/bin/env python3
"""Numerically audit the production DFA local-gradient contraction.

This is a deterministic source-level check; it does not train a model and it
does not claim to reproduce a paper checkpoint.  It exercises the same
``Trainer._local_batch`` and ``_capture_snn_dfa_layer`` code used by the
experiment.  For every hidden layer it compares autograd with

    local_weight / (T * B * D_out) * sum_{t,b} delta[t,b]^T x[t,b].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from analysis.paper1_geometry.diagnostics import (  # noqa: E402
    _capture_snn_dfa_layer,
    _weight_from_delta,
)
from methods import FeedbackBank  # noqa: E402
from methods.gate_intervention import GateMode  # noqa: E402
from models import build_model  # noqa: E402
from training.config import load_config  # noqa: E402
from training.engine import Trainer, _one_hot  # noqa: E402
from training.seed import seed_everything  # noqa: E402


def _relative_scale_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Maximum absolute discrepancy relative to the largest expected entry."""

    denominator = max(float(expected.detach().abs().max()), 1e-30)
    return float((actual.detach() - expected.detach()).abs().max()) / denominator


def _relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(expected.detach())) + 1e-30
    return float(torch.linalg.vector_norm(actual.detach() - expected.detach())) / denominator


def run(seed: int = 20260912, tolerance: float = 1e-6) -> dict[str, Any]:
    """Run the fixed synthetic parity check and return a JSON-safe record."""

    config_path = REPO_ROOT / "configs" / "nmnist" / "paper1_experiment03.yaml"
    config = load_config(config_path)
    # Small deterministic tensors keep the check fast while retaining the
    # production three-layer FC, pointwise-DFA and reduction paths.
    config["experiment"]["seed"] = int(seed)
    config["data"].update(
        {
            "dataset": "synthetic",
            "root": ".",
            "time_steps": 4,
            "batch_size": 3,
            "train_samples": 6,
            "test_samples": 3,
            "num_workers": 0,
            "pin_memory": False,
        }
    )
    config["model"]["input_shape"] = [2, 4, 4]
    config["model"]["hidden_features"] = [9, 8, 7]
    config["training"]["gradient_clip"] = 0.0

    seed_everything(seed)
    device = torch.device("cpu")
    model = build_model(config)
    feedback_bank = FeedbackBank(model, config)
    trainer = Trainer(model, feedback_bank, config, device)
    model.eval()
    feedback_bank.eval()

    generator = torch.Generator().manual_seed(seed + 1)
    samples = torch.rand(
        config["data"]["time_steps"],
        config["data"]["batch_size"],
        *config["model"]["input_shape"],
        generator=generator,
    )
    labels = torch.tensor([1, 2, 3], dtype=torch.long)
    sample_indices = torch.tensor([101, 203, 307], dtype=torch.long)

    with torch.no_grad():
        output, hidden_inputs, _readout_input = model.forward_with_cache(
            samples, detach_temporal=True
        )
        desired = _one_hot(labels, model.num_classes).to(output.dtype)
        output_error = output.mean(dim=0) - desired

    local_weight = float(config["method"].get("local_weight", 1.0))
    captures: list[dict[str, Any]] = []
    expected_gradients: list[torch.Tensor] = []
    for layer_index, layer_input in enumerate(hidden_inputs):
        capture = _capture_snn_dfa_layer(
            model,
            feedback_bank,
            layer_index,
            layer_input,
            output_error,
            gate_mode=GateMode.ACTUAL,
            seed=seed,
            epoch=0,
            sample_indices=sample_indices,
        )
        explicit = local_weight * _weight_from_delta(capture["delta"], layer_input)
        capture_gradient = local_weight * capture["weight_gradient"]
        expected_gradients.append(explicit.detach())
        captures.append(
            {
                "layer": f"H{layer_index + 1}",
                "delta_shape": list(capture["delta"].shape),
                "input_shape": list(layer_input.shape),
                "gradient_shape": list(explicit.shape),
                "global_reduction_scalar": local_weight / capture["delta"].numel(),
                "explicit_vs_capture_max_relative_error": _relative_scale_error(
                    explicit, capture_gradient
                ),
                "explicit_vs_capture_relative_l2_error": _relative_l2_error(
                    explicit, capture_gradient
                ),
                "production_delta_relative_l2_error": float(
                    capture["parity_relative"]
                ),
            }
        )

    trainer.optimizer.zero_grad(set_to_none=True)
    values = trainer._local_batch(samples, labels)
    total = values["total"]
    assert isinstance(total, torch.Tensor)
    total.backward()
    for layer_index, (row, expected) in enumerate(zip(captures, expected_gradients)):
        actual = model.hidden_layers[layer_index].linear.weight.grad
        if actual is None:
            raise RuntimeError(f"autograd did not produce a gradient for H{layer_index + 1}")
        row["explicit_vs_trainer_max_relative_error"] = _relative_scale_error(
            explicit := expected, actual
        )
        row["explicit_vs_trainer_relative_l2_error"] = _relative_l2_error(
            explicit, actual
        )

    compared = [
        float(row[key])
        for row in captures
        for key in (
            "explicit_vs_capture_max_relative_error",
            "explicit_vs_trainer_max_relative_error",
        )
    ]
    maximum = max(compared, default=float("inf"))
    return {
        "status": "passed" if maximum <= tolerance else "failed",
        "check_scope": "deterministic synthetic source-level parity; no training",
        "seed": int(seed),
        "device": "cpu",
        "temporal_mode": config["method"]["temporal_mode"],
        "local_loss_reduction": "sum_over_layers(mean_over_T_B_D(spikes * detached_q))",
        "explicit_contraction": (
            "local_weight/(T*B*D_out) * sum_{t,b} delta[t,b]^T x[t,b]"
        ),
        "tolerance": float(tolerance),
        "max_relative_error": maximum,
        "layers": captures,
    }


def _write_json(path: Path, record: dict[str, Any], force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite generated output: {path}")
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    record = run(seed=args.seed, tolerance=args.tolerance)
    if args.output:
        _write_json(args.output.resolve(), record, args.force)
    print(json.dumps(record, indent=2, allow_nan=False))
    raise SystemExit(0 if record["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
