"""Cross-checkpoint geometry for Experiment 01B."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from analysis.feedback_expansion.core import (
    direction_energy,
    energy_in_basis,
    principal_subspace_metrics,
    scatter_about,
)
from analysis.feedback_expansion.diagnostic import CheckpointDiagnostics, SavedBasis


FIXED_RANKS = (8, 16, 32, 64, 128, 256)


def _rank_candidates(final: SavedBasis) -> list[tuple[str, int]]:
    candidates = [(f"fixed_{k}", k) for k in FIXED_RANKS]
    candidates.extend(
        [
            ("final_entropy_rank", int(round(final.metrics["entropy_rank"]))),
            ("final_r90", int(final.metrics["r90"])),
            ("final_r95", int(final.metrics["r95"])),
        ]
    )
    result: list[tuple[str, int]] = []
    seen: set[int] = set()
    for source, rank in candidates:
        rank = max(1, min(rank, final.basis.shape[0], final.basis.shape[1]))
        if rank not in seen:
            result.append((source, rank))
            seen.add(rank)
    return result


def analyze_seed_trajectory(
    *,
    seed: int,
    checkpoints: dict[int, CheckpointDiagnostics],
    scheduled_epochs: list[int],
) -> dict[str, list[dict[str, Any]]]:
    """Analyze delta bases using only predeclared common scheduled epochs."""

    epochs = [epoch for epoch in scheduled_epochs if epoch in checkpoints]
    if len(epochs) < 2:
        raise ValueError(f"seed {seed} has fewer than two scheduled checkpoints")
    final_epoch = max(epochs)
    overlap_rows: list[dict[str, Any]] = []
    novel_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []

    layers = sorted(checkpoints[final_epoch].delta_eval)
    for layer in layers:
        # Both representations are retained for rotation/overlap; expansion
        # consequence metrics use the training-rule aggregated signal.
        for temporal_mode in ("aggregated", "timestep"):
            final = checkpoints[final_epoch].bases[
                (layer, "post_gate_delta", temporal_mode)
            ]
            ranks = _rank_candidates(final)
            for epoch_a in epochs:
                a = checkpoints[epoch_a].bases[
                    (layer, "post_gate_delta", temporal_mode)
                ]
                for epoch_b in epochs:
                    b = checkpoints[epoch_b].bases[
                        (layer, "post_gate_delta", temporal_mode)
                    ]
                    for source, rank in ranks:
                        metrics = principal_subspace_metrics(a.basis, b.basis, rank)
                        overlap_rows.append(
                            {
                                "seed": seed,
                                "layer": layer,
                                "signal_type": "post_gate_delta",
                                "temporal_mode": temporal_mode,
                                "epoch_a": epoch_a,
                                "epoch_b": epoch_b,
                                "k_source": source,
                                **metrics,
                            }
                        )

        final = checkpoints[final_epoch].bases[
            (layer, "post_gate_delta", "aggregated")
        ]
        ranks = _rank_candidates(final)
        final_scatter = scatter_about(
            checkpoints[final_epoch].delta_eval[layer], final.mean
        )
        for epoch_index, epoch in enumerate(epochs):
            current = checkpoints[epoch].bases[
                (layer, "post_gate_delta", "aggregated")
            ]
            current_scatter = scatter_about(
                checkpoints[epoch].delta_eval[layer], current.mean
            )
            for source, rank in ranks:
                current_in_final = energy_in_basis(current_scatter, final.basis, rank)
                final_in_current = energy_in_basis(final_scatter, current.basis, rank)
                capture_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "epoch": epoch,
                        "k_source": source,
                        "k": rank,
                        "current_energy_in_final": current_in_final,
                        "final_energy_in_current": final_in_current,
                    }
                )
            if epoch_index > 0:
                previous_epoch = epochs[epoch_index - 1]
                previous = checkpoints[previous_epoch].bases[
                    (layer, "post_gate_delta", "aggregated")
                ]
                candidates = [
                    ("fixed_32", 32),
                    ("fixed_64", 64),
                    ("fixed_128", 128),
                    ("previous_r95", int(previous.metrics["r95"])),
                    ("final_r95", int(final.metrics["r95"])),
                ]
                seen: set[int] = set()
                for source, requested in candidates:
                    rank = max(1, min(requested, previous.basis.shape[1]))
                    if (source, rank) in seen:
                        continue
                    seen.add((source, rank))
                    old = energy_in_basis(current_scatter, previous.basis, rank)
                    novel_rows.append(
                        {
                            "seed": seed,
                            "layer": layer,
                            "epoch": epoch,
                            "previous_epoch": previous_epoch,
                            "k_source": source,
                            "k": rank,
                            "old_energy_ratio": old,
                            "novel_energy_ratio": 1.0 - old,
                        }
                    )

        final_directions = final.basis[:, :128]
        epoch_energy = {}
        for epoch in epochs:
            current = checkpoints[epoch].bases[
                (layer, "post_gate_delta", "aggregated")
            ]
            scatter = scatter_about(checkpoints[epoch].delta_eval[layer], current.mean)
            epoch_energy[epoch] = direction_energy(scatter, final_directions)
            for direction, energy in enumerate(epoch_energy[epoch], start=1):
                energy_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "epoch": epoch,
                        "direction": direction,
                        "normalized_energy": float(energy),
                    }
                )
        final_energy = epoch_energy[final_epoch]
        for direction in range(final_directions.shape[1]):
            values = [epoch_energy[epoch][direction] for epoch in epochs]
            denominator = float(final_energy[direction])

            def birth(threshold: float) -> int | None:
                if denominator <= 0:
                    return None
                for candidate_epoch, value in zip(epochs, values):
                    if float(value) >= threshold * denominator:
                        return candidate_epoch
                return None

            activation_rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "direction": direction + 1,
                    "final_rank_order": direction + 1,
                    "birth20": birth(0.20),
                    "birth50": birth(0.50),
                    "birth80": birth(0.80),
                    "final_energy": denominator,
                    "energy_by_epoch": json.dumps(
                        {str(epoch): float(value) for epoch, value in zip(epochs, values)}
                    ),
                }
            )
    return {
        "overlap": overlap_rows,
        "novel": novel_rows,
        "capture": capture_rows,
        "activation": activation_rows,
        "direction_energy": energy_rows,
    }


def enrich_shuffled_control(
    *,
    seed: int,
    checkpoints: dict[int, CheckpointDiagnostics],
    final_epoch: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_result = checkpoints[final_epoch]
    for epoch, result in sorted(checkpoints.items()):
        if not result.control_rows:
            continue
        for row in result.control_rows:
            layer = str(row["layer"])
            final_basis = final_result.bases[
                (layer, "post_gate_delta", "aggregated")
            ]
            condition = str(row["label_condition"])
            covariance = (
                result.delta_eval[layer]
                if condition == "correct"
                else result.shuffled_eval[layer]
            )
            current = result.bases[(layer, "post_gate_delta", "aggregated")]
            center = (
                current.mean
                if condition == "correct"
                else result.shuffled_mean[layer]
            )
            scatter = scatter_about(covariance, center)
            for source, rank in (
                ("fixed_32", 32),
                ("fixed_64", 64),
                ("fixed_128", 128),
                ("final_r95", int(final_basis.metrics["r95"])),
            ):
                capture = energy_in_basis(scatter, final_basis.basis, rank)
                rows.append(
                    {
                        **row,
                        "k_source": source,
                        "k": min(rank, final_basis.basis.shape[1]),
                        "energy_in_final_correct_subspace": capture,
                        "novel_energy_to_final_correct": 1.0 - capture,
                    }
                )
    return rows
