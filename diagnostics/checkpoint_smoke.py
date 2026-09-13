#!/usr/bin/env python3
"""Replay one frozen N-MNIST DFA checkpoint/layer without training.

The command verifies the checkpoint SHA-256 and the complete archived probe
ordering before computing delta on the basis_eval half. Missing external
artifacts produce a machine-readable skipped record and exit status 3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics  # noqa: E402
from analysis.paper1_geometry.data import build_paper1_split, make_loaders  # noqa: E402
from analysis.paper1_geometry.diagnostics import (  # noqa: E402
    _capture_snn_dfa_layer,
    _prepare,
    _weight_from_delta,
)
from analysis.paper1_geometry.metrics import feature_signal  # noqa: E402
from methods import FeedbackBank  # noqa: E402
from methods.gate_intervention import GateMode  # noqa: E402
from models import build_model  # noqa: E402
from training.data import build_datasets  # noqa: E402


EXIT_FAILED = 1
EXIT_SKIPPED = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a slash-separated repository-relative path in saved evidence."""

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # External data/checkpoint paths belong to the reviewer environment;
        # avoid persisting a workstation-specific absolute path.
        return f"external:{path.name}"


def read_probe(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {"probe_position", "dataset_index", "probe_split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"probe CSV must contain {sorted(required)}: {path}")
    rows.sort(key=lambda row: int(row["probe_position"]))
    return rows


def _component_metrics(
    accumulator: OnlineCovariance, device: torch.device
) -> dict[str, Any]:
    metrics, _singular, _energy, _cumulative = spectrum_statistics(
        accumulator,
        algebraic_max_dim=accumulator.ambient_dim,
        device=device,
    )
    return {
        key: metrics[key]
        for key in (
            "observations",
            "ambient_dim",
            "r50",
            "r80",
            "r90",
            "r95",
            "r99",
            "stable_rank",
            "entropy_rank",
            "participation_ratio",
            "r95_over_ambient",
        )
    }


def decompose_signal(
    signal: torch.Tensor, *, chunk_size: int, device: torch.device
) -> dict[str, Any]:
    """Use the manuscript total/within-time/between-sample decomposition."""

    if signal.ndim != 3:
        raise ValueError(f"expected feature signal [T,N,D], got {tuple(signal.shape)}")
    time_steps, samples, dimension = map(int, signal.shape)
    total = OnlineCovariance(dimension)
    within = OnlineCovariance(dimension)
    sample_means = OnlineCovariance(dimension)
    for start in range(0, samples, chunk_size):
        value = signal[:, start : start + chunk_size].to(
            device=device, dtype=torch.float32
        )
        total.update(value)
        means = value.mean(dim=0)
        within.update(value - means.unsqueeze(0))
        sample_means.update(means)
    between = sample_means.repeated(time_steps)
    residual = total.m2 - within.m2 - between.m2
    total_norm = float(torch.linalg.vector_norm(total.m2))
    total_trace = float(torch.trace(total.m2))
    components: dict[str, Any] = {}
    for name, accumulator in (
        ("total", total),
        ("within_time", within),
        ("between_sample", between),
    ):
        trace = float(torch.trace(accumulator.m2))
        components[name] = {
            "scatter_trace": trace,
            "trace_fraction_of_total": trace / total_trace if total_trace > 0 else 0.0,
            **_component_metrics(accumulator, device),
        }
    return {
        "shape_T_N_D": [time_steps, samples, dimension],
        "centering": "single global mean over all T*N observations",
        "scatter_identity": "S_total = S_within_time + S_between_sample",
        "identity_relative_frobenius": float(torch.linalg.vector_norm(residual))
        / (total_norm + 1e-30),
        "trace_fraction_sum": (
            components["within_time"]["trace_fraction_of_total"]
            + components["between_sample"]["trace_fraction_of_total"]
        ),
        "components": components,
    }


def _resolve_layer(value: str, number_of_layers: int) -> int:
    token = value.strip().lower().replace("hidden_", "").replace("h", "")
    try:
        index = int(token) - 1
    except ValueError as error:
        raise ValueError(f"layer must be H1, H2, ...; got {value!r}") from error
    if not 0 <= index < number_of_layers:
        raise ValueError(f"layer {value!r} is not present in the checkpoint")
    return index


def _write(path: Path | None, record: dict[str, Any], *, force: bool) -> None:
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite generated output: {path}")
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")


def skipped(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason_code": reason,
        "operation": "checkpoint replay only; no optimizer step; no training",
        **details,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    probe = args.probe.resolve()
    data_root = args.data_root.resolve()
    if not checkpoint.is_file():
        return skipped("canonical_checkpoint_missing", checkpoint=portable_path(checkpoint))
    if not probe.is_file():
        return skipped("fixed_probe_csv_missing", probe=portable_path(probe))
    if not data_root.is_dir():
        return skipped("dataset_root_missing", data_root="external:data_root")
    if not args.expected_sha256:
        return skipped(
            "expected_checkpoint_sha256_missing",
            explanation="A checkpoint is not accepted as canonical without its frozen hash.",
        )

    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256.lower() != args.expected_sha256.lower():
        return {
            "status": "failed",
            "reason_code": "checkpoint_sha256_mismatch",
            "expected_sha256": args.expected_sha256.lower(),
            "actual_sha256": actual_sha256,
            "operation": "checkpoint replay only; no optimizer step; no training",
        }

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"epoch", "model_state", "feedback_state", "config"}
    missing = sorted(required.difference(state))
    if missing:
        raise KeyError(f"checkpoint is missing keys: {missing}")
    if int(state["epoch"]) != 99:
        return {
            "status": "failed",
            "reason_code": "not_completed_epoch_100_checkpoint",
            "stored_zero_based_epoch": int(state["epoch"]),
            "expected_zero_based_epoch": 99,
        }

    config = state["config"]
    if str(config["data"]["dataset"]).lower() != "nmnist":
        raise ValueError("this reviewer smoke command currently supports N-MNIST only")
    if str(config["model"]["architecture"]).lower() != "fc":
        raise ValueError("this reviewer smoke command expects the canonical FC model")
    if str(config["method"]["name"]).lower() not in {"dfa", "sdfa"}:
        raise ValueError("delta replay requires a DFA/sDFA checkpoint")
    config["data"]["root"] = str(data_root)
    config["data"]["num_workers"] = 0
    config["data"]["pin_memory"] = False

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return skipped("cuda_unavailable", requested_device=args.device)
    model = build_model(config).to(device)
    feedback_bank = FeedbackBank(model, config).to(device)
    model.load_state_dict(state["model_state"])
    feedback_bank.load_state_dict(state["feedback_state"])
    model.eval()
    feedback_bank.eval()
    layer_index = _resolve_layer(args.layer, len(model.hidden_specs))

    try:
        train_dataset, test_dataset = build_datasets(config)
    except FileNotFoundError:
        steps = int(config["data"]["time_steps"])
        return skipped(
            "preprocessed_dataset_missing",
            expected_relative_layout=(
                f"nmnist/frames_number_{steps}_split_by_number/{{train,test}}"
            ),
        )
    split_config = config.get("experiment03", {})
    split = build_paper1_split(
        len(train_dataset),
        split_seed=int(split_config.get("split_seed", 20260830)),
        probe_seed=int(split_config.get("probe_seed", 20260831)),
        validation_fraction=float(split_config.get("validation_fraction", 0.10)),
        probe_size=int(split_config.get("probe_size", 1024)),
    )
    probe_rows = read_probe(probe)
    archived_indices = [int(row["dataset_index"]) for row in probe_rows]
    archived_splits = [row["probe_split"] for row in probe_rows]
    generated_splits = [
        "basis_fit" if position < len(split.basis_fit_indices) else "basis_eval"
        for position in range(len(split.probe_indices))
    ]
    if split.probe_indices.tolist() != archived_indices:
        return {
            "status": "failed",
            "reason_code": "probe_index_order_mismatch",
            "probe_sha256": sha256_file(probe),
        }
    if archived_splits != generated_splits:
        return {
            "status": "failed",
            "reason_code": "probe_fit_eval_assignment_mismatch",
            "probe_sha256": sha256_file(probe),
        }

    loaders = make_loaders(
        train_dataset,
        test_dataset,
        split,
        config,
        seed=int(args.seed),
        workers=0,
    )
    chunks: list[torch.Tensor] = []
    parity_errors: list[float] = []
    contraction_errors: list[float] = []
    sample_count = 0
    for batch in loaders["basis_eval"]:
        samples, labels, indices = _prepare(batch, device)
        with torch.no_grad():
            output, hidden_inputs, _readout_input = model.forward_with_cache(
                samples, detach_temporal=True
            )
            desired = F.one_hot(labels, num_classes=model.num_classes).to(output.dtype)
            output_error = output.mean(dim=0) - desired
        capture = _capture_snn_dfa_layer(
            model,
            feedback_bank,
            layer_index,
            hidden_inputs[layer_index],
            output_error,
            gate_mode=GateMode.ACTUAL,
            seed=int(args.seed),
            epoch=100,
            sample_indices=indices,
        )
        explicit = _weight_from_delta(capture["delta"], hidden_inputs[layer_index])
        denominator = float(capture["weight_gradient"].abs().max()) + 1e-30
        contraction_errors.append(
            float((explicit - capture["weight_gradient"]).abs().max()) / denominator
        )
        parity_errors.append(float(capture["parity_relative"]))
        chunks.append(feature_signal(capture["delta"]).detach().cpu())
        sample_count += int(labels.numel())

    signal = torch.cat(chunks, dim=1)
    decomposition = decompose_signal(
        signal,
        chunk_size=int(config["data"]["batch_size"]),
        device=device,
    )
    measured_r95 = int(decomposition["components"]["total"]["r95"])
    measured_within = float(
        decomposition["components"]["within_time"]["trace_fraction_of_total"]
    )

    checks: dict[str, Any] = {
        "checkpoint_sha256": "passed",
        "checkpoint_completed_epoch_100": "passed",
        "probe_index_order": "passed",
        "probe_fit_eval_assignment": "passed",
        "sample_count_512": "passed" if sample_count == 512 else "failed",
        "explicit_gradient_contraction": (
            "passed"
            if max(contraction_errors, default=float("inf")) <= args.tolerance
            else "failed"
        ),
        "covariance_identity": (
            "passed"
            if decomposition["identity_relative_frobenius"] <= args.tolerance
            else "failed"
        ),
    }
    if args.expected_r95 is not None:
        checks["frozen_r95"] = (
            "passed" if measured_r95 == args.expected_r95 else "failed"
        )
    if args.expected_within_fraction is not None:
        checks["frozen_within_fraction"] = (
            "passed"
            if abs(measured_within - args.expected_within_fraction)
            <= args.continuous_tolerance
            else "failed"
        )
    status = "passed" if set(checks.values()) == {"passed"} else "failed"
    return {
        "status": status,
        "operation": "checkpoint replay only; no optimizer step; no training",
        "dataset": "N-MNIST",
        "seed": int(args.seed),
        "layer": f"H{layer_index + 1}",
        "checkpoint": portable_path(checkpoint),
        "checkpoint_sha256": actual_sha256,
        "probe": portable_path(probe),
        "probe_sha256": sha256_file(probe),
        "probe_split": "basis_eval",
        "probe_samples": sample_count,
        "time_steps": int(signal.shape[0]),
        "checks": checks,
        "max_production_delta_relative_l2_error": max(parity_errors, default=None),
        "max_explicit_gradient_relative_error": max(contraction_errors, default=None),
        "expected_r95": args.expected_r95,
        "measured_r95": measured_r95,
        "expected_within_time_trace_fraction": args.expected_within_fraction,
        "measured_within_time_trace_fraction": measured_within,
        "decomposition": decomposition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--layer", default="H1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-r95", type=int)
    parser.add_argument("--expected-within-fraction", type=float)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--continuous-tolerance", type=float, default=5e-6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        record = run(args)
    except Exception as error:
        record = {
            "status": "failed",
            "reason_code": "unhandled_replay_error",
            "error_type": type(error).__name__,
            "detail": str(error),
            "traceback": traceback.format_exc(),
            "operation": "checkpoint replay only; no optimizer step; no training",
        }
    _write(args.output.resolve() if args.output else None, record, force=args.force)
    if record["status"] == "passed":
        raise SystemExit(0)
    if record["status"] == "skipped":
        raise SystemExit(EXIT_SKIPPED)
    raise SystemExit(EXIT_FAILED)


if __name__ == "__main__":
    main()
