"""Checkpoint-level causal update-preservation diagnostics."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from analysis.feedback_subspace.metrics import OnlineCovariance
from analysis.update_relevance.core import (
    capture_bptt_updates,
    capture_dfa_layer_update,
    evaluate_sdfa_losses,
    pca_basis,
    principal_subspace_overlap,
    project_update,
    random_orthogonal_basis,
    virtual_sgd_step,
)


def _prepare_batch(
    batch: tuple[Tensor, Tensor], device: torch.device
) -> tuple[Tensor, Tensor]:
    samples, target = batch
    return (
        samples.transpose(0, 1)
        .contiguous()
        .to(device=device, dtype=torch.float32, non_blocking=True),
        target.to(device=device, non_blocking=True),
    )


def _dfa_inputs(model, samples: Tensor, target: Tensor, pointwise: bool):
    with torch.no_grad():
        output, hidden_inputs, _readout_input = model.forward_with_cache(
            samples, detach_temporal=pointwise
        )
        desired = F.one_hot(target, num_classes=model.num_classes).to(
            device=samples.device, dtype=output.dtype
        )
        output_error = output.mean(dim=0) - desired
    return hidden_inputs, output_error.detach()


def _basis_checksum(basis: Tensor) -> str:
    values = basis.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def fit_dfa_bases(
    model,
    feedback_bank,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    seed: int,
    epoch: int,
    pointwise: bool,
) -> tuple[dict[tuple[int, str], Tensor], list[dict[str, Any]]]:
    """Fit aggregated and timestep DFA PCA bases on basis_fit only."""

    covariances: dict[tuple[int, str], OnlineCovariance] = {}
    for layer_index, spec in enumerate(model.hidden_specs):
        dimension = int(spec["dimension"])
        covariances[(layer_index, "aggregated")] = OnlineCovariance(dimension)
        covariances[(layer_index, "timestep")] = OnlineCovariance(dimension)

    for batch in loader:
        samples, target = _prepare_batch(batch, device)
        hidden_inputs, output_error = _dfa_inputs(model, samples, target, pointwise)
        for layer_index, layer_input in enumerate(hidden_inputs):
            capture = capture_dfa_layer_update(
                model,
                feedback_bank,
                layer_index,
                layer_input,
                output_error,
                detach_temporal=pointwise,
            )
            delta = capture["delta"]
            covariances[(layer_index, "timestep")].update(delta)
            covariances[(layer_index, "aggregated")].update(delta.mean(dim=0))

    bases: dict[tuple[int, str], Tensor] = {}
    metadata: list[dict[str, Any]] = []
    for (layer_index, temporal_mode), covariance in sorted(covariances.items()):
        basis, eigenvalues, metrics = pca_basis(covariance, device)
        bases[(layer_index, temporal_mode)] = basis
        metadata.append(
            {
                "seed": seed,
                "epoch": epoch,
                "layer": f"hidden_{layer_index + 1}",
                "basis_type": f"{temporal_mode}_delta_pca",
                "fit_split": "basis_fit",
                "basis_checksum": _basis_checksum(basis),
                "leading_eigenvalue": float(eigenvalues[0]),
                **metrics,
            }
        )
    return bases, metadata


def fit_bptt_bases(
    model,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    seed: int,
    epoch: int,
) -> tuple[dict[tuple[int, str], Tensor], list[dict[str, Any]]]:
    """Fit counterfactual-BPTT PCA bases on the same non-leaking split."""

    covariances: dict[tuple[int, str], OnlineCovariance] = {}
    for layer_index, spec in enumerate(model.hidden_specs):
        dimension = int(spec["dimension"])
        covariances[(layer_index, "aggregated")] = OnlineCovariance(dimension)
        covariances[(layer_index, "timestep")] = OnlineCovariance(dimension)
    for batch in loader:
        samples, target = _prepare_batch(batch, device)
        capture = capture_bptt_updates(model, samples, target)
        for layer_index, delta in enumerate(capture["delta"]):
            covariances[(layer_index, "timestep")].update(delta)
            covariances[(layer_index, "aggregated")].update(delta.mean(dim=0))

    bases: dict[tuple[int, str], Tensor] = {}
    metadata: list[dict[str, Any]] = []
    for (layer_index, temporal_mode), covariance in sorted(covariances.items()):
        basis, eigenvalues, metrics = pca_basis(covariance, device)
        bases[(layer_index, temporal_mode)] = basis
        metadata.append(
            {
                "seed": seed,
                "epoch": epoch,
                "layer": f"hidden_{layer_index + 1}",
                "basis_type": f"bptt_{temporal_mode}_delta_pca",
                "fit_split": "basis_fit",
                "basis_checksum": _basis_checksum(basis),
                "leading_eigenvalue": float(eigenvalues[0]),
                **metrics,
            }
        )
    return bases, metadata


def _metric_row(
    *,
    seed: int,
    epoch: int,
    batch_index: int,
    layer_index: int,
    hidden_dim: int,
    basis_type: str,
    k: int,
    k_source: str,
    projection_type: str,
    random_repeat: int | None,
    random_seed: int | None,
    norm_matching: bool,
    values: dict[str, Tensor | float],
    capture: dict[str, Tensor | float],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "epoch": epoch,
        "batch_index": batch_index,
        "layer": f"hidden_{layer_index + 1}",
        "basis_type": basis_type,
        "k": k,
        "k_source": k_source,
        "k_ratio": k / float(hidden_dim),
        "projection_type": projection_type,
        "random_repeat": random_repeat,
        "random_seed": random_seed,
        "norm_matching": bool(norm_matching),
        "gradient_cosine": values["gradient_cosine"],
        "relative_error": values["relative_error"],
        "energy_ratio": values["energy_ratio"],
        "sign_agreement": values["sign_agreement"],
        "sign_threshold": values["sign_threshold"],
        "sign_parameter_count": values["sign_parameter_count"],
        "update_norm_ratio": values["update_norm_ratio"],
        "delta_energy_ratio": values["delta_energy_ratio"],
        "norm_match_scale": values["norm_match_scale"],
        "full_weight_gradient_norm": float(
            torch.linalg.vector_norm(capture["weight_gradient"])
        ),
        "parity_weight_max_abs": capture["parity_weight_max_abs"],
        "parity_weight_relative": capture["parity_weight_relative"],
        "parity_bias_max_abs": capture["parity_bias_max_abs"],
        "parity_bias_relative": capture["parity_bias_relative"],
    }


def _loss_row(
    common: dict[str, Any],
    before: dict[str, float],
    after: dict[str, float],
    learning_rate: float,
) -> dict[str, Any]:
    return {
        **common,
        "learning_rate": learning_rate,
        "optimizer_interpretation": "layer-local SGD-equivalent; checkpoint unchanged",
        "loss_evaluation_precision": "float64 checkpoint copy",
        "task_loss_before": before["task_loss"],
        "task_loss_after": after["task_loss"],
        "task_loss_change": after["task_loss"] - before["task_loss"],
        "local_proxy_loss_before": before["local_proxy_loss"],
        "local_proxy_loss_after": after["local_proxy_loss"],
        "local_proxy_loss_change": (
            after["local_proxy_loss"] - before["local_proxy_loss"]
        ),
    }


def evaluate_gradient_preservation(
    model,
    feedback_bank,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    seed: int,
    epoch: int,
    bases: dict[tuple[int, str], Tensor],
    ranks_by_layer: dict[int, list[tuple[str, int]]],
    random_repeats: int,
    pointwise: bool,
    learning_rate: float,
    run_loss_descent: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate top/random projectors against the exact layer weight gradient."""

    metric_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    random_cache: dict[tuple[int, int, int], tuple[Tensor, int]] = {}
    for batch_index, batch in enumerate(loader):
        samples, target = _prepare_batch(batch, device)
        hidden_inputs, output_error = _dfa_inputs(model, samples, target, pointwise)
        loss_model = None
        loss_feedback_bank = None
        loss_samples = None
        before = None
        if run_loss_descent:
            # The configured SGD-equivalent step is often below float32 weight
            # resolution (lr=1e-4 times local gradients around 1e-5).  A
            # float64 state copy preserves the exact same parameters and
            # learning rate while making the virtual finite step measurable.
            loss_model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
            loss_feedback_bank = copy.deepcopy(feedback_bank).to(
                device=device, dtype=torch.float64
            )
            loss_model.eval()
            loss_feedback_bank.eval()
            loss_samples = samples.to(dtype=torch.float64)
            before = evaluate_sdfa_losses(
                loss_model,
                loss_feedback_bank,
                loss_samples,
                target,
                detach_temporal=pointwise,
            )
        for layer_index, layer_input in enumerate(hidden_inputs):
            hidden_dim = int(model.hidden_specs[layer_index]["dimension"])
            capture = capture_dfa_layer_update(
                model,
                feedback_bank,
                layer_index,
                layer_input,
                output_error,
                detach_temporal=pointwise,
            )
            if (
                float(capture["parity_weight_relative"]) > 2e-5
                or float(capture["parity_bias_relative"]) > 2e-5
            ):
                raise AssertionError(
                    f"gradient parity failed seed={seed} epoch={epoch} "
                    f"layer={layer_index + 1}: {capture['parity_weight_relative']}"
                )

            identity_values: dict[str, Tensor | float] = {
                "gradient_cosine": 1.0,
                "relative_error": 0.0,
                "energy_ratio": 1.0,
                "sign_agreement": 1.0,
                "sign_threshold": 0.0,
                "sign_parameter_count": int(capture["weight_gradient"].numel()),
                "update_norm_ratio": 1.0,
                "delta_energy_ratio": 1.0,
                "norm_match_scale": 1.0,
            }
            metric_rows.append(
                _metric_row(
                    seed=seed,
                    epoch=epoch,
                    batch_index=batch_index,
                    layer_index=layer_index,
                    hidden_dim=hidden_dim,
                    basis_type="identity",
                    k=hidden_dim,
                    k_source="full_D",
                    projection_type="full",
                    random_repeat=None,
                    random_seed=None,
                    norm_matching=False,
                    values=identity_values,
                    capture=capture,
                )
            )
            if run_loss_descent:
                assert before is not None
                with virtual_sgd_step(
                    loss_model,
                    layer_index,
                    capture["weight_gradient"].to(dtype=torch.float64),
                    capture["bias_gradient"].to(dtype=torch.float64),
                    learning_rate,
                ):
                    after = evaluate_sdfa_losses(
                        loss_model,
                        loss_feedback_bank,
                        loss_samples,
                        target,
                        detach_temporal=pointwise,
                    )
                loss_rows.append(
                    _loss_row(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "layer": f"hidden_{layer_index + 1}",
                            "basis_type": "identity",
                            "k": hidden_dim,
                            "k_source": "full_D",
                            "k_ratio": 1.0,
                            "projection_type": "full",
                            "random_repeat": None,
                            "random_seed": None,
                            "norm_matching": False,
                        },
                        before,
                        after,
                        learning_rate,
                    )
                )

            for temporal_mode in ("aggregated", "timestep"):
                full_basis = bases[(layer_index, temporal_mode)]
                for k_source, requested_k in ranks_by_layer[layer_index]:
                    k = min(int(requested_k), hidden_dim)
                    top_basis = full_basis[:, :k]
                    for norm_matching in (False, True):
                        projected = project_update(
                            capture["delta"],
                            capture["weight_gradient"],
                            capture["bias_gradient"],
                            top_basis,
                            norm_matching=norm_matching,
                        )
                        metric_rows.append(
                            _metric_row(
                                seed=seed,
                                epoch=epoch,
                                batch_index=batch_index,
                                layer_index=layer_index,
                                hidden_dim=hidden_dim,
                                basis_type=f"top_{temporal_mode}",
                                k=k,
                                k_source=k_source,
                                projection_type="top",
                                random_repeat=None,
                                random_seed=None,
                                norm_matching=norm_matching,
                                values=projected,
                                capture=capture,
                            )
                        )
                        if run_loss_descent and k < hidden_dim:
                            assert before is not None
                            with virtual_sgd_step(
                                loss_model,
                                layer_index,
                                projected["weight_gradient"].to(dtype=torch.float64),
                                projected["bias_gradient"].to(dtype=torch.float64),
                                learning_rate,
                            ):
                                after = evaluate_sdfa_losses(
                                    loss_model,
                                    loss_feedback_bank,
                                    loss_samples,
                                    target,
                                    detach_temporal=pointwise,
                                )
                            loss_rows.append(
                                _loss_row(
                                    {
                                        "seed": seed,
                                        "epoch": epoch,
                                        "batch_index": batch_index,
                                        "layer": f"hidden_{layer_index + 1}",
                                        "basis_type": f"top_{temporal_mode}",
                                        "k": k,
                                        "k_source": k_source,
                                        "k_ratio": k / float(hidden_dim),
                                        "projection_type": "top",
                                        "random_repeat": None,
                                        "random_seed": None,
                                        "norm_matching": norm_matching,
                                    },
                                    before,
                                    after,
                                    learning_rate,
                                )
                            )

            # A random projector is independent of whether the learned basis
            # was fitted from aggregated or timestep observations.
            for k_source, requested_k in ranks_by_layer[layer_index]:
                k = min(int(requested_k), hidden_dim)
                if k == hidden_dim:
                    continue
                for repeat in range(random_repeats):
                    cache_key = (layer_index, k, repeat)
                    if cache_key not in random_cache:
                        random_seed = (
                            seed * 1_000_003
                            + epoch * 10_009
                            + (layer_index + 1) * 1_009
                            + k * 97
                            + repeat
                        ) % (2**63 - 1)
                        random_cache[cache_key] = (
                            random_orthogonal_basis(
                                hidden_dim, k, random_seed, device
                            ),
                            random_seed,
                        )
                    random_basis, random_seed = random_cache[cache_key]
                    for norm_matching in (False, True):
                        projected = project_update(
                            capture["delta"],
                            capture["weight_gradient"],
                            capture["bias_gradient"],
                            random_basis,
                            norm_matching=norm_matching,
                        )
                        metric_rows.append(
                            _metric_row(
                                seed=seed,
                                epoch=epoch,
                                batch_index=batch_index,
                                layer_index=layer_index,
                                hidden_dim=hidden_dim,
                                basis_type="random_orthogonal",
                                k=k,
                                k_source=k_source,
                                projection_type="random",
                                random_repeat=repeat,
                                random_seed=random_seed,
                                norm_matching=norm_matching,
                                values=projected,
                                capture=capture,
                            )
                        )
                        if run_loss_descent:
                            assert before is not None
                            with virtual_sgd_step(
                                loss_model,
                                layer_index,
                                projected["weight_gradient"].to(dtype=torch.float64),
                                projected["bias_gradient"].to(dtype=torch.float64),
                                learning_rate,
                            ):
                                after = evaluate_sdfa_losses(
                                    loss_model,
                                    loss_feedback_bank,
                                    loss_samples,
                                    target,
                                    detach_temporal=pointwise,
                                )
                            loss_rows.append(
                                _loss_row(
                                    {
                                        "seed": seed,
                                        "epoch": epoch,
                                        "batch_index": batch_index,
                                        "layer": f"hidden_{layer_index + 1}",
                                        "basis_type": "random_orthogonal",
                                        "k": k,
                                        "k_source": k_source,
                                        "k_ratio": k / float(hidden_dim),
                                        "projection_type": "random",
                                        "random_repeat": repeat,
                                        "random_seed": random_seed,
                                        "norm_matching": norm_matching,
                                    },
                                    before,
                                    after,
                                    learning_rate,
                                )
                            )
        del loss_model, loss_feedback_bank, loss_samples
    return metric_rows, loss_rows


def evaluate_bptt_counterfactual(
    model,
    feedback_bank,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    seed: int,
    epoch: int,
    dfa_fit_bases: dict[tuple[int, str], Tensor],
    dfa_fit_metadata: list[dict[str, Any]],
    bptt_fit_bases: dict[tuple[int, str], Tensor],
    bptt_fit_metadata: list[dict[str, Any]],
    pointwise: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare DFA and true-task BPTT credit on held-out evaluation samples."""

    covariances: dict[tuple[str, int, str], OnlineCovariance] = {}
    for method in ("DFA", "BPTT"):
        for layer_index, spec in enumerate(model.hidden_specs):
            dimension = int(spec["dimension"])
            for temporal_mode in ("aggregated", "timestep"):
                covariances[(method, layer_index, temporal_mode)] = OnlineCovariance(
                    dimension
                )
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        samples, target = _prepare_batch(batch, device)
        hidden_inputs, output_error = _dfa_inputs(model, samples, target, pointwise)
        dfa_captures = []
        for layer_index, layer_input in enumerate(hidden_inputs):
            dfa_capture = capture_dfa_layer_update(
                model,
                feedback_bank,
                layer_index,
                layer_input,
                output_error,
                detach_temporal=pointwise,
            )
            dfa_captures.append(dfa_capture)
            delta = dfa_capture["delta"]
            covariances[("DFA", layer_index, "timestep")].update(delta)
            covariances[("DFA", layer_index, "aggregated")].update(
                delta.mean(dim=0)
            )
        bptt_capture = capture_bptt_updates(model, samples, target)
        for layer_index, delta in enumerate(bptt_capture["delta"]):
            covariances[("BPTT", layer_index, "timestep")].update(delta)
            covariances[("BPTT", layer_index, "aggregated")].update(
                delta.mean(dim=0)
            )
            dfa_gradient = dfa_captures[layer_index]["weight_gradient"]
            bptt_gradient = bptt_capture["weight_gradient"][layer_index]
            cosine = torch.sum(dfa_gradient * bptt_gradient) / (
                torch.linalg.vector_norm(dfa_gradient)
                * torch.linalg.vector_norm(bptt_gradient)
                + 1e-30
            )
            rows.append(
                {
                    "record_type": "weight_gradient_alignment",
                    "seed": seed,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "layer": f"hidden_{layer_index + 1}",
                    "method": "DFA_vs_BPTT",
                    "temporal_mode": "weight_gradient",
                    "split": "evaluation",
                    "gradient_cosine": float(cosine),
                }
            )

    evaluation_metrics: dict[tuple[str, int, str], dict[str, Any]] = {}
    for (method, layer_index, temporal_mode), covariance in sorted(
        covariances.items()
    ):
        _basis, _eigenvalues, metrics = pca_basis(covariance, device)
        evaluation_metrics[(method, layer_index, temporal_mode)] = metrics
        rows.append(
            {
                "record_type": "effective_rank",
                "seed": seed,
                "epoch": epoch,
                "batch_index": None,
                "layer": f"hidden_{layer_index + 1}",
                "method": method,
                "temporal_mode": temporal_mode,
                "split": "evaluation",
                "gradient_cosine": None,
                **metrics,
            }
        )

    dfa_meta = {
        (int(row["layer"].split("_")[-1]) - 1, row["basis_type"].split("_")[0]): row
        for row in dfa_fit_metadata
    }
    bptt_meta = {
        (
            int(row["layer"].split("_")[-1]) - 1,
            row["basis_type"].replace("bptt_", "").split("_")[0],
        ): row
        for row in bptt_fit_metadata
    }
    angle_rows: list[dict[str, Any]] = []
    for layer_index, _spec in enumerate(model.hidden_specs):
        for temporal_mode in ("aggregated", "timestep"):
            dfa_r95 = int(dfa_meta[(layer_index, temporal_mode)]["r95"])
            bptt_r95 = int(bptt_meta[(layer_index, temporal_mode)]["r95"])
            candidates = [
                ("min_r95", min(dfa_r95, bptt_r95)),
                ("fixed_32", 32),
                ("fixed_64", 64),
                ("fixed_128", 128),
            ]
            seen = set()
            for k_source, requested in candidates:
                k = min(
                    requested,
                    dfa_fit_bases[(layer_index, temporal_mode)].shape[1],
                    bptt_fit_bases[(layer_index, temporal_mode)].shape[1],
                )
                if k in seen or k < 1:
                    continue
                seen.add(k)
                overlap, angles, cosines = principal_subspace_overlap(
                    dfa_fit_bases[(layer_index, temporal_mode)],
                    bptt_fit_bases[(layer_index, temporal_mode)],
                    k,
                )
                for angle_index, (angle, cosine) in enumerate(
                    zip(angles, cosines), start=1
                ):
                    angle_rows.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "layer": f"hidden_{layer_index + 1}",
                            "temporal_mode": temporal_mode,
                            "k_source": k_source,
                            "k": k,
                            "overlap_k": overlap,
                            "principal_angle_index": angle_index,
                            "principal_angle_degrees": float(angle),
                            "principal_angle_cosine": float(cosine),
                            "dfa_fit_r95": dfa_r95,
                            "bptt_fit_r95": bptt_r95,
                        }
                    )
    return rows, angle_rows
