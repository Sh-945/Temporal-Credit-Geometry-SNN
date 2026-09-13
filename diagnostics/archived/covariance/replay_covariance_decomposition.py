"""Offline covariance decomposition from frozen Paper-v9 checkpoints.

This script only replays completed checkpoints on the archived diagnostic probe.
It writes new per-seed artifacts below ``08_covariance_decomposition`` and never
updates the frozen Paper-v9 result tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


HERE = Path(__file__).resolve().parent
V9 = HERE.parent
ROOT = V9.parents[1]
FORMAL = ROOT / "results/paper1_experiment03_20260901_server"
LEGACY = ROOT / "results/experiment03_soft_spectral/experiment03_20260831_server"
STATELESS = (
    V9 / "06_mechanism_exploration/01_stateless_surrogate_control"
)
sys.path[:0] = [str(ROOT), str(V9), str(STATELESS)]

from analysis.feedback_subspace.metrics import (  # noqa: E402
    OnlineCovariance,
    spectrum_statistics,
)
from analysis.paper1_geometry.data import (  # noqa: E402
    build_paper1_split,
    make_loaders,
)
from analysis.paper1_geometry.diagnostics import (  # noqa: E402
    _capture_ann_layer,
    _capture_snn_dfa_layer,
    _prepare,
)
from analysis.paper1_geometry.metrics import feature_signal  # noqa: E402
from methods import FeedbackBank  # noqa: E402
from models import build_model  # noqa: E402
from models.temporal_ann_control import TemporalANN  # noqa: E402
from stateless_surrogate import make_stateless  # noqa: E402
from training.data import build_datasets  # noqa: E402


N_MNIST_SEEDS = (20260830, 20260831, 20260901, 20260908, 20260909)
THREE_SEEDS = (20260830, 20260831, 20260901)
EXPERIMENT_SEEDS = {
    "nmnist": N_MNIST_SEEDS,
    "ann": N_MNIST_SEEDS,
    "stateless": THREE_SEEDS,
    "dvs": THREE_SEEDS,
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: str
    method: str
    probe_size: int
    coordinate_space: str


SPECS = {
    "nmnist": ExperimentSpec(
        "nmnist", "N-MNIST", "SNN-DFA", 1024, "neuron"
    ),
    "ann": ExperimentSpec(
        "ann", "N-MNIST", "temporal ANN-DFA", 1024, "neuron"
    ),
    "stateless": ExperimentSpec(
        "stateless",
        "N-MNIST",
        "memoryless surrogate-spiking DFA",
        1024,
        "neuron",
    ),
    "dvs": ExperimentSpec(
        "dvs",
        "DVS-Gesture",
        "SNN-DFA",
        896,
        "spatially averaged channel",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_payload(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("refusing to serialize an empty result table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    destination = io.StringIO(newline="")
    writer = csv.DictWriter(destination, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return destination.getvalue()


def write_or_validate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = csv_payload(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf8", newline="") as source:
            existing = source.read()
        if existing != payload:
            raise RuntimeError(f"existing partial result conflicts with replay: {path}")
        return
    with path.open("x", encoding="utf8", newline="") as destination:
        destination.write(payload)


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")


def write_blocked(path: Path, value: Any) -> None:
    if not path.exists():
        write_json_exclusive(path, value)


def checkpoint_path(experiment: str, seed: int) -> Path:
    if experiment == "nmnist":
        if seed in THREE_SEEDS:
            return LEGACY / f"checkpoints/full_dense/seed_{seed}/epoch_100.pt"
        return V9 / f"03_five_seed/checkpoints/dfa_trained/seed_{seed}/epoch_100.pt"
    if experiment == "ann":
        if seed in THREE_SEEDS:
            return (
                FORMAL
                / f"checkpoints/stageB_ann/temporal_ann_dfa/seed_{seed}/epoch_100.pt"
            )
        return (
            V9
            / f"03_five_seed/checkpoints/temporal_ann_dfa/seed_{seed}/epoch_100.pt"
        )
    if experiment == "stateless":
        return STATELESS / f"checkpoints/seed_{seed}/epoch_100.pt"
    if experiment == "dvs":
        return (
            FORMAL
            / f"checkpoints/stageD_cross_dataset/dvs_dfa/seed_{seed}/epoch_100.pt"
        )
    raise KeyError(experiment)


def probe_path(experiment: str) -> Path:
    if experiment == "dvs":
        return FORMAL / "stageD_cross_dataset/diagnostic_probe.csv"
    return FORMAL / "diagnostic_probe.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf8") as source:
        return list(csv.DictReader(source))


def frozen_reference(
    experiment: str, seed: int
) -> tuple[Path, dict[str, dict[str, Any]]]:
    references: dict[str, dict[str, Any]] = {}
    if experiment in {"nmnist", "ann"}:
        prefix = "dfa_trained" if experiment == "nmnist" else "temporal_ann_dfa"
        path = V9 / f"01_spectral_metrics/{prefix}_{seed}.csv"
        rows = [
            row
            for row in read_csv(path)
            if row["probe_split"] == "basis_eval"
            and row["temporal_mode"] == "timestep"
        ]
        for row in rows:
            references[row["layer"]] = {
                "r95": int(row["r95"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
    elif experiment == "stateless":
        path = STATELESS / "stateless_surrogate_per_seed.csv"
        rows = [row for row in read_csv(path) if int(row["seed"]) == seed]
        for row in rows:
            references[row["layer"]] = {
                "r95": int(row["r95"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
    elif experiment == "dvs":
        path = V9 / "04_gate_statistics/dvs_per_seed.csv"
        rows = [
            row
            for row in read_csv(path)
            if int(row["seed"]) == seed
            and row["gate_space"] == "canonical_spatial_mean_channels"
        ]
        for row in rows:
            references[row["layer"]] = {
                "r95": int(row["credit_r95"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
    else:
        raise KeyError(experiment)
    if len(references) != 3:
        raise RuntimeError(
            f"expected three frozen layer references for {experiment}/{seed}, "
            f"found {len(references)} in {path}"
        )
    return path, references


def alpha1_reference(seed: int) -> tuple[Path, dict[str, int]]:
    path = (
        V9
        / "06_mechanism_exploration/00_offline_gate_manipulations/"
        "temporal_homogenization_per_seed.csv"
    )
    rows = [
        row
        for row in read_csv(path)
        if int(row["seed"]) == seed
        and math.isclose(float(row["alpha"]), 1.0)
        and row["scale_mode"] == "global_frobenius_norm_matched"
    ]
    result = {row["layer"]: int(row["r95"]) for row in rows}
    if len(result) != 3:
        raise RuntimeError(f"missing alpha=1 references for seed {seed}: {path}")
    return path, result


def spectral_result(
    accumulator: OnlineCovariance, device: torch.device
) -> tuple[dict[str, Any], np.ndarray]:
    metrics, singular, _energy, _cumulative = spectrum_statistics(
        accumulator,
        algebraic_max_dim=accumulator.ambient_dim,
        device=device,
    )
    return {
        "observations": metrics["observations"],
        "r50": metrics["r50"],
        "r80": metrics["r80"],
        "r90": metrics["r90"],
        "r95": metrics["r95"],
        "r99": metrics["r99"],
        "stable_rank": metrics["stable_rank"],
        "entropy_effective_rank": metrics["entropy_rank"],
        "participation_ratio": metrics["participation_ratio"],
        "r95_over_D": metrics["r95_over_ambient"],
    }, singular


def spectral_row(
    accumulator: OnlineCovariance, device: torch.device
) -> dict[str, Any]:
    return spectral_result(accumulator, device)[0]


def write_or_validate_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        archived = np.load(path, allow_pickle=False)
        if archived.shape != value.shape or not np.array_equal(archived, value):
            raise RuntimeError(f"existing spectrum conflicts with replay: {path}")
        return
    with path.open("xb") as destination:
        np.save(destination, value, allow_pickle=False)


def persist_spectra(
    *,
    experiment: str,
    seed: int,
    layer: str,
    condition: str,
    signal: str,
    spectra: dict[str, np.ndarray],
) -> dict[str, str]:
    """Persist complete centered singular spectra with one file per component."""

    safe_condition = condition.replace(" ", "_")
    result: dict[str, str] = {}
    for component, singular in spectra.items():
        path = (
            HERE
            / "spectra"
            / f"{experiment}_seed_{seed}_{layer}_{safe_condition}_{signal}_{component}.npy"
        )
        write_or_validate_npy(path, singular)
        result[component] = str(path.relative_to(ROOT))
    return result


def decompose_signal(
    signal: Tensor,
    *,
    chunk_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, np.ndarray]]:
    """Return total, within-time, and between-sample centered scatters.

    For ``signal`` x with shape [T,N,D], this computes

      S_total   = sum_nt (x_nt - mu) (x_nt - mu)^T
      S_within  = sum_nt (x_nt - mean_t(x_n)) (...)^T
      S_between = T sum_n (mean_t(x_n) - mu) (...)^T.

    Thus S_total = S_within + S_between. Dividing every matrix by N*T
    gives the population-covariance decomposition used in the manuscript.
    """

    if signal.ndim != 3:
        raise ValueError(f"expected [T,N,D], got {tuple(signal.shape)}")
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
    denominator = float(torch.linalg.vector_norm(total.m2))
    identity_relative = float(torch.linalg.vector_norm(residual)) / (
        denominator + 1e-30
    )
    total_trace = float(torch.trace(total.m2))
    parts = {
        "total": total,
        "within_time": within,
        "between_sample": between,
    }
    rows: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    for name, accumulator in parts.items():
        trace = float(torch.trace(accumulator.m2))
        summary, singular = spectral_result(accumulator, device)
        trace_fraction = trace / total_trace if total_trace > 0.0 else 0.0
        numerical_zero = (
            name != "total"
            and total_trace > 0.0
            and trace_fraction <= 1e-12
        )
        if numerical_zero:
            # A time-constant tensor has exactly zero within-time covariance.
            # Float32 mean/subtraction can leave ~1e-14 relative trace, whose
            # normalized spectrum has no physical meaning.  Preserve the raw
            # trace for auditability but report all scale-invariant ranks as 0.
            singular = np.zeros_like(singular)
            summary.update(
                {
                    "r50": 0,
                    "r80": 0,
                    "r90": 0,
                    "r95": 0,
                    "r99": 0,
                    "stable_rank": 0.0,
                    "entropy_effective_rank": 0.0,
                    "participation_ratio": 0.0,
                    "r95_over_D": 0.0,
                }
            )
        spectra[name] = singular
        rows.append(
            {
                "component": name,
                "scatter_trace": trace,
                "trace_fraction_of_total": trace_fraction,
                "numerically_zero_component": numerical_zero,
                **summary,
            }
        )
    checks = {
        "identity_relative_frobenius": identity_relative,
        "total_trace": total_trace,
        "within_trace_fraction": rows[1]["trace_fraction_of_total"],
        "between_trace_fraction": rows[2]["trace_fraction_of_total"],
    }
    return rows, checks, spectra


def q_carrier_row(
    q: Tensor,
    *,
    chunk_size: int,
    device: torch.device,
    feedback_matrix: Tensor,
) -> tuple[dict[str, Any], np.ndarray]:
    if q.ndim != 2:
        raise ValueError(f"expected q [N,D], got {tuple(q.shape)}")
    accumulator = OnlineCovariance(int(q.shape[1]))
    for start in range(0, q.shape[0], chunk_size):
        accumulator.update(q[start : start + chunk_size].to(device))
    absolute = q.detach().abs().to(torch.float64)
    matrix = feedback_matrix.detach().cpu().to(torch.float64)
    summary, singular = spectral_result(accumulator, device)
    return {
        "component": "carrier_total",
        "scatter_trace": float(torch.trace(accumulator.m2)),
        "trace_fraction_of_total": 1.0,
        **summary,
        "q_exact_zero_fraction": float((absolute == 0).to(torch.float64).mean()),
        "q_near_zero_fraction_1e-6": float(
            (absolute < 1e-6).to(torch.float64).mean()
        ),
        "q_active_coordinate_fraction": float(
            (absolute.max(dim=0).values > 1e-12).to(torch.float64).mean()
        ),
        "q_rms": float(q.detach().to(torch.float64).square().mean().sqrt()),
        "feedback_rows": int(matrix.shape[0]),
        "feedback_columns": int(matrix.shape[1]),
        "feedback_matrix_rank": int(torch.linalg.matrix_rank(matrix)),
        "feedback_active_row_fraction": float(
            (matrix.abs().max(dim=1).values > 1e-12).to(torch.float64).mean()
        ),
    }, singular


def load_model_and_bank(
    experiment: str, state: dict[str, Any], device: torch.device
):
    config = state["config"]
    if experiment == "ann":
        model = TemporalANN(
            config["model"]["input_shape"],
            config["model"]["hidden_features"],
            config["model"]["num_classes"],
        )
        replaced: list[str] = []
    else:
        model = build_model(config)
        replaced = make_stateless(model) if experiment == "stateless" else []
    bank = FeedbackBank(model, config)
    model.load_state_dict(state["model_state"], strict=True)
    bank.load_state_dict(state["feedback_state"], strict=True)
    model.to(device).eval()
    bank.to(device).eval()
    return model, bank, replaced


def archived_probe_rows(experiment: str) -> list[dict[str, str]]:
    return read_csv(probe_path(experiment))


def capture_signals(
    *,
    experiment: str,
    seed: int,
    state: dict[str, Any],
    device: torch.device,
) -> tuple[
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[tuple[int, ...]],
    list[str],
]:
    config = state["config"]
    model, bank, replaced = load_model_and_bank(experiment, state, device)
    train, test = build_datasets(config)
    split = build_paper1_split(
        len(train),
        split_seed=20260830,
        probe_seed=20260831,
        validation_fraction=0.1,
        probe_size=SPECS[experiment].probe_size,
    )
    archived = archived_probe_rows(experiment)
    expected_indices = [int(row["dataset_index"]) for row in archived]
    expected_splits = [row["probe_split"] for row in archived]
    actual_splits = [
        "basis_fit" if position < SPECS[experiment].probe_size // 2 else "basis_eval"
        for position in range(SPECS[experiment].probe_size)
    ]
    if split.probe_indices.tolist() != expected_indices:
        raise RuntimeError("diagnostic probe index mismatch")
    if actual_splits != expected_splits:
        raise RuntimeError("diagnostic probe fit/eval ordering mismatch")
    loader = make_loaders(train, test, split, config, seed=seed)["basis_eval"]
    layer_count = len(model.hidden_specs)
    deltas: list[list[Tensor]] = [[] for _ in range(layer_count)]
    gates: list[list[Tensor]] = [[] for _ in range(layer_count)]
    carriers: list[list[Tensor]] = [[] for _ in range(layer_count)]
    raw_shapes: list[tuple[int, ...] | None] = [None] * layer_count
    for batch in loader:
        samples, labels, indices = _prepare(batch, device)
        with torch.no_grad():
            outputs = (
                model.forward_with_cache(samples)
                if experiment == "ann"
                else model.forward_with_cache(samples, detach_temporal=True)
            )
            desired = F.one_hot(labels, num_classes=model.num_classes).to(
                outputs[0].dtype
            )
            error = outputs[0].mean(dim=0) - desired
        for layer_index, layer_input in enumerate(outputs[1]):
            if experiment == "ann":
                q, gate, delta, parity = _capture_ann_layer(
                    model, bank, layer_index, layer_input, error
                )
                if parity > 1e-5:
                    raise RuntimeError(
                        f"ANN local-error parity failed at layer {layer_index}: {parity}"
                    )
            else:
                capture = _capture_snn_dfa_layer(
                    model,
                    bank,
                    layer_index,
                    layer_input,
                    error,
                    gate_mode="actual",
                    seed=seed,
                    epoch=100,
                    sample_indices=indices,
                )
                q = capture["q"]
                gate = capture["gate"]
                delta = capture["delta"]
                if capture["parity_relative"] > 1e-5:
                    raise RuntimeError(
                        "SNN production local-error parity failed at layer "
                        f"{layer_index}: {capture['parity_relative']}"
                    )
            raw_shapes[layer_index] = tuple(int(value) for value in gate.shape)
            if experiment == "dvs":
                gate = feature_signal(gate)
                delta = feature_signal(delta)
            deltas[layer_index].append(delta.detach().cpu())
            gates[layer_index].append(gate.detach().cpu())
            carriers[layer_index].append(q.detach().cpu())
    joined_delta = [torch.cat(value, dim=1) for value in deltas]
    joined_gate = [torch.cat(value, dim=1) for value in gates]
    joined_q = [torch.cat(value, dim=0) for value in carriers]
    expected_n = SPECS[experiment].probe_size // 2
    for layer_index, (delta, gate, q) in enumerate(
        zip(joined_delta, joined_gate, joined_q)
    ):
        dimension = int(model.hidden_specs[layer_index]["dimension"])
        expected = (30, expected_n, dimension)
        if tuple(delta.shape) != expected or tuple(gate.shape) != expected:
            raise RuntimeError(
                f"projected shape mismatch at H{layer_index + 1}: "
                f"delta={tuple(delta.shape)}, gate={tuple(gate.shape)}, expected={expected}"
            )
        if tuple(q.shape) != (expected_n, dimension):
            raise RuntimeError(f"carrier shape mismatch at H{layer_index + 1}")
    feedback = [layer.B.detach().cpu() for layer in bank.layers]
    del model, bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        joined_delta,
        joined_gate,
        joined_q,
        feedback,
        [shape for shape in raw_shapes if shape is not None],
        replaced,
    )


def base_row(
    *,
    spec: ExperimentSpec,
    seed: int,
    layer: str,
    condition: str,
    signal: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    frozen_source: Path,
    probe: Path,
    n: int,
    t: int,
    d: int,
    raw_shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "experiment": spec.name,
        "dataset": spec.dataset,
        "method": spec.method,
        "seed": seed,
        "layer": layer,
        "condition": condition,
        "signal": signal,
        "probe_split": "basis_eval",
        "N": n,
        "T": t,
        "D": d,
        "coordinate_space": spec.coordinate_space,
        "raw_capture_shape_last_batch": json.dumps(raw_shape),
        "normalization": "centered scatter; population covariance = scatter/(N*T)",
        "checkpoint_rule": "completed_epoch_100; stored checkpoint epoch=99",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_parity_source": str(frozen_source.relative_to(ROOT)),
        "probe_file": str(probe.relative_to(ROOT)),
    }


def run_one(experiment: str, seed: int, device: torch.device) -> dict[str, Any]:
    spec = SPECS[experiment]
    output = HERE / "per_seed" / f"{experiment}_seed_{seed}.csv"
    complete = HERE / "per_seed" / f"{experiment}_seed_{seed}.complete.json"
    blocked = HERE / "per_seed" / f"{experiment}_seed_{seed}.BLOCKED.json"
    checkpoint = checkpoint_path(experiment, seed)
    probe = probe_path(experiment)
    if complete.exists():
        record = json.loads(complete.read_text(encoding="utf8"))
        if not output.exists():
            raise RuntimeError(f"completion marker exists without CSV: {complete}")
        if record["checkpoint"] != str(checkpoint.relative_to(ROOT)):
            raise RuntimeError(f"completion marker checkpoint conflict: {complete}")
        return {"experiment": experiment, "seed": seed, "status": "already_complete"}
    start = time.time()
    frozen_source, expected = frozen_reference(experiment, seed)
    checkpoint_sha = sha256_file(checkpoint)
    expected_hashes = {row["checkpoint_sha256"] for row in expected.values()}
    if expected_hashes != {checkpoint_sha}:
        reason = {
            "status": "blocked",
            "reason": "checkpoint SHA-256 does not match frozen reference",
            "experiment": experiment,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "actual_sha256": checkpoint_sha,
            "expected_sha256": sorted(expected_hashes),
            "frozen_source": str(frozen_source),
        }
        write_blocked(blocked, reason)
        raise RuntimeError(reason["reason"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != 99:
        raise RuntimeError(f"expected stored epoch 99, found {state['epoch']}")
    delta, gate, q, feedback, raw_shapes, replaced = capture_signals(
        experiment=experiment,
        seed=seed,
        state=state,
        device=device,
    )
    chunk_size = int(state["config"]["data"]["batch_size"])
    n = int(delta[0].shape[1])
    t = int(delta[0].shape[0])
    rows: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    identity_checks: list[dict[str, Any]] = []
    alpha_checks: list[dict[str, Any]] = []
    alpha_source: Path | None = None
    alpha_expected: dict[str, int] = {}
    if experiment == "nmnist":
        alpha_source, alpha_expected = alpha1_reference(seed)
    for layer_index, (delta_value, gate_value, q_value, matrix) in enumerate(
        zip(delta, gate, q, feedback)
    ):
        layer = f"H{layer_index + 1}"
        dimension = int(delta_value.shape[-1])
        base_arguments = {
            "spec": spec,
            "seed": seed,
            "layer": layer,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "frozen_source": frozen_source,
            "probe": probe,
            "n": n,
            "t": t,
            "d": dimension,
            "raw_shape": raw_shapes[layer_index],
        }
        for signal_name, value in (("delta", delta_value), ("gate", gate_value)):
            component_rows, checks, component_spectra = decompose_signal(
                value, chunk_size=chunk_size, device=device
            )
            spectrum_paths = persist_spectra(
                experiment=experiment,
                seed=seed,
                layer=layer,
                condition="actual",
                signal=signal_name,
                spectra=component_spectra,
            )
            for component in component_rows:
                component["spectrum_file"] = spectrum_paths[
                    component["component"]
                ]
            identity_checks.append(
                {
                    "layer": layer,
                    "condition": "actual",
                    "signal": signal_name,
                    **checks,
                }
            )
            prefix = base_row(
                **base_arguments, condition="actual", signal=signal_name
            )
            for component in component_rows:
                rows.append(
                    {
                        **prefix,
                        "identity_relative_frobenius": checks[
                            "identity_relative_frobenius"
                        ],
                        **component,
                    }
                )
            if signal_name == "delta":
                replay_r95 = int(component_rows[0]["r95"])
                expected_r95 = int(expected[layer]["r95"])
                parity.append(
                    {
                        "layer": layer,
                        "expected_frozen_r95": expected_r95,
                        "replay_total_r95": replay_r95,
                        "passed": replay_r95 == expected_r95,
                    }
                )
                for row in rows[-3:]:
                    row["expected_frozen_r95"] = expected_r95
                    row["replay_total_r95"] = replay_r95
                    row["frozen_r95_parity_passed"] = replay_r95 == expected_r95
        carrier, carrier_singular = q_carrier_row(
            q_value,
            chunk_size=chunk_size,
            device=device,
            feedback_matrix=matrix,
        )
        carrier_prefix = base_row(
            **base_arguments, condition="actual", signal="q_carrier"
        )
        carrier["spectrum_file"] = persist_spectra(
            experiment=experiment,
            seed=seed,
            layer=layer,
            condition="actual",
            signal="q_carrier",
            spectra={"carrier_total": carrier_singular},
        )["carrier_total"]
        rows.append(
            {
                **carrier_prefix,
                "identity_relative_frobenius": "",
                **carrier,
            }
        )
        if experiment == "dvs":
            spatial_factor = (
                int(raw_shapes[layer_index][-1]) * int(raw_shapes[layer_index][-2])
                if len(raw_shapes[layer_index]) == 5
                else 1
            )
            rows[-1]["effective_carrier_scale_in_projected_delta"] = 1.0 / float(
                spatial_factor
            )
        if experiment == "nmnist":
            mean_gate = gate_value.mean(dim=0, keepdim=True).expand_as(gate_value)
            scale = float(
                torch.linalg.vector_norm(gate_value)
                / (torch.linalg.vector_norm(mean_gate) + 1e-12)
            )
            homogeneous_gate = mean_gate * scale
            homogeneous_delta = homogeneous_gate * q_value.unsqueeze(0)
            norm_error = float(
                (
                    torch.linalg.vector_norm(homogeneous_gate)
                    - torch.linalg.vector_norm(gate_value)
                ).abs()
                / (torch.linalg.vector_norm(gate_value) + 1e-30)
            )
            # The archived intervention performs both reductions over a
            # 12.3M-entry float32 tensor.  Its replay is identified primarily
            # by exact r95 parity; a 2e-3 relative guard accommodates the
            # observed reduction-order roundoff while still detecting a wrong
            # scaling rule.
            for signal_name, value in (
                ("delta", homogeneous_delta),
                ("gate", homogeneous_gate),
            ):
                component_rows, checks, component_spectra = decompose_signal(
                    value, chunk_size=chunk_size, device=device
                )
                spectrum_paths = persist_spectra(
                    experiment=experiment,
                    seed=seed,
                    layer=layer,
                    condition="alpha_1_homogenized_frobenius_matched",
                    signal=signal_name,
                    spectra=component_spectra,
                )
                for component in component_rows:
                    component["spectrum_file"] = spectrum_paths[
                        component["component"]
                    ]
                prefix = base_row(
                    **base_arguments,
                    condition="alpha_1_homogenized_frobenius_matched",
                    signal=signal_name,
                )
                for component in component_rows:
                    rows.append(
                        {
                            **prefix,
                            "identity_relative_frobenius": checks[
                                "identity_relative_frobenius"
                            ],
                            "homogenization_scale": scale,
                            "gate_frobenius_norm_relative_error": norm_error,
                            **component,
                        }
                    )
                identity_checks.append(
                    {
                        "layer": layer,
                        "condition": "alpha_1_homogenized_frobenius_matched",
                        "signal": signal_name,
                        **checks,
                    }
                )
                if signal_name == "delta":
                    replay_r95 = int(component_rows[0]["r95"])
                    reference_r95 = int(alpha_expected[layer])
                    alpha_checks.append(
                        {
                            "layer": layer,
                            "expected_existing_alpha1_r95": reference_r95,
                            "replay_alpha1_total_r95": replay_r95,
                            "r95_passed": replay_r95 == reference_r95,
                            "within_trace_fraction": checks[
                                "within_trace_fraction"
                            ],
                            "within_zero_passed": checks["within_trace_fraction"]
                            <= 1e-10,
                            "gate_norm_match_relative_error": norm_error,
                            "gate_norm_match_tolerance": 2e-3,
                            "gate_norm_match_passed": norm_error <= 2e-3,
                        }
                    )
                    for row in rows[-3:]:
                        row["alpha1_reference_r95"] = reference_r95
                        row["alpha1_r95_parity_passed"] = (
                            replay_r95 == reference_r95
                        )
    all_parity = all(check["passed"] for check in parity)
    identity_ok = all(
        check["identity_relative_frobenius"] <= 5e-6
        for check in identity_checks
    )
    alpha_ok = all(
        check["r95_passed"]
        and check["within_zero_passed"]
        and check["gate_norm_match_passed"]
        for check in alpha_checks
    )
    if not all_parity or not identity_ok or not alpha_ok:
        reason = {
            "status": "blocked",
            "reason": "a preregistered parity or decomposition identity check failed",
            "experiment": experiment,
            "seed": seed,
            "frozen_r95_parity": parity,
            "identity_checks": identity_checks,
            "alpha1_checks": alpha_checks,
        }
        write_blocked(blocked, reason)
        raise RuntimeError(reason["reason"])
    write_or_validate_csv(output, rows)
    helper_paths = [
        ROOT / "analysis/paper1_geometry/diagnostics.py",
        ROOT / "analysis/paper1_geometry/metrics.py",
        ROOT / "analysis/feedback_subspace/metrics.py",
        ROOT / "analysis/paper1_geometry/data.py",
    ]
    if experiment == "stateless":
        helper_paths.append(STATELESS / "stateless_surrogate.py")
    provenance = {
        "status": "complete",
        "operation": "checkpoint replay only; no optimizer step; no training",
        "experiment": experiment,
        "dataset": spec.dataset,
        "method": spec.method,
        "seed": seed,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_stored_epoch": int(state["epoch"]),
        "probe": str(probe.relative_to(ROOT)),
        "probe_sha256": sha256_file(probe),
        "probe_split": "basis_eval",
        "N": n,
        "T": t,
        "dimensions": [int(value.shape[-1]) for value in delta],
        "raw_capture_shapes_last_batch": [list(value) for value in raw_shapes],
        "coordinate_space": spec.coordinate_space,
        "frozen_reference": str(frozen_source.relative_to(ROOT)),
        "frozen_reference_sha256": sha256_file(frozen_source),
        "frozen_r95_parity": parity,
        "alpha1_reference": (
            str(alpha_source.relative_to(ROOT)) if alpha_source is not None else None
        ),
        "alpha1_checks": alpha_checks,
        "decomposition_identity_checks": identity_checks,
        "decomposition": {
            "total_scatter": "sum_nt (x_nt-grand_mean)(x_nt-grand_mean)^T",
            "within_time_scatter": "sum_nt (x_nt-sample_time_mean_n)(x_nt-sample_time_mean_n)^T",
            "between_sample_scatter": "T*sum_n (sample_time_mean_n-grand_mean)(sample_time_mean_n-grand_mean)^T",
            "identity": "total = within_time + between_sample",
            "covariance_normalization": "divide each scatter by N*T",
        },
        "dvs_projection": (
            "mean over spatial H,W before covariance; delta retains the production 1/(H*W) teaching-signal scale"
            if experiment == "dvs"
            else None
        ),
        "replaced_cells": replaced,
        "source_helpers": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path in helper_paths
        ],
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.time() - start,
        "output_csv": str(output.relative_to(ROOT)),
        "rows": len(rows),
        "frozen_v9_modified": False,
    }
    write_json_exclusive(complete, provenance)
    return {"experiment": experiment, "seed": seed, "status": "complete"}


def selected_jobs(experiment: str, seeds: list[int] | None) -> list[tuple[str, int]]:
    experiments: Iterable[str] = (
        SPECS.keys() if experiment == "all" else (experiment,)
    )
    jobs: list[tuple[str, int]] = []
    for name in experiments:
        supported = EXPERIMENT_SEEDS[name]
        if seeds is None:
            jobs.extend((name, seed) for seed in supported)
            continue
        selected = [seed for seed in seeds if seed in supported]
        if experiment != "all" and len(selected) != len(seeds):
            unsupported = sorted(set(seeds).difference(supported))
            raise ValueError(f"unsupported seeds for {name}: {unsupported}")
        jobs.extend((name, seed) for seed in selected)
    if not jobs:
        raise ValueError("no supported experiment/seed jobs selected")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen checkpoints and decompose local-error covariance."
    )
    parser.add_argument(
        "--experiment",
        choices=("nmnist", "ann", "stateless", "dvs", "all"),
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Optional seed; repeat to select multiple seeds. Omit for all supported seeds.",
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(4)
    results = []
    for experiment, seed in selected_jobs(arguments.experiment, arguments.seed):
        result = run_one(experiment, seed, device)
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({"status": "complete", "jobs": results}), flush=True)


if __name__ == "__main__":
    main()
