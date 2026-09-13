"""Numerical primitives for Experiment 02.

The diagnostic deliberately consumes the same replayed local proxy used by
``Trainer._local_batch``.  Projection is applied to the exact current
gradient, and the corresponding FC weight gradient follows from the chain
rule.  No projector is installed in the normal training path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics


def _task_loss(output_spikes: Tensor, target: Tensor, classes: int) -> Tensor:
    mean_output = output_spikes.mean(dim=0)
    desired = F.one_hot(target, num_classes=classes).to(
        device=mean_output.device, dtype=mean_output.dtype
    )
    return 0.5 * (mean_output - desired).square().sum(dim=1).mean()


def capture_dfa_layer_update(
    model,
    feedback_bank,
    layer_index: int,
    layer_input: Tensor,
    output_error: Tensor,
    *,
    detach_temporal: bool,
) -> dict[str, Tensor | float]:
    """Return exact DFA current, weight, and bias gradients for one FC layer."""

    if model.hidden_specs[layer_index]["kind"] != "fc":
        raise ValueError("Experiment 02 primary path currently requires FC layers")
    teaching = feedback_bank.project(layer_index, output_error).detach()
    linear = model.hidden_layers[layer_index].linear
    captured: list[Tensor] = []

    def remember_current(_module, _inputs, output: Tensor) -> None:
        captured.append(output)

    handle = linear.register_forward_hook(remember_current)
    try:
        spikes = model.replay_hidden(
            layer_index, layer_input.detach(), detach_temporal=detach_temporal
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one current for hidden_{layer_index + 1}, got {len(captured)}"
        )
    local_proxy = (spikes * teaching.unsqueeze(0)).mean()
    current_gradient, weight_gradient, bias_gradient = torch.autograd.grad(
        local_proxy,
        (captured[0], linear.weight, linear.bias),
        retain_graph=False,
        create_graph=False,
    )
    current_gradient = current_gradient.reshape_as(spikes)
    chain_weight = torch.einsum(
        "tbo,tbi->oi", current_gradient, layer_input.detach()
    )
    chain_bias = current_gradient.sum(dim=(0, 1))
    weight_difference = chain_weight - weight_gradient
    bias_difference = chain_bias - bias_gradient
    weight_denominator = float(torch.linalg.vector_norm(weight_gradient)) + 1e-30
    bias_denominator = float(torch.linalg.vector_norm(bias_gradient)) + 1e-30
    mean_denominator = int(current_gradient.numel())
    return {
        "delta": (current_gradient * mean_denominator).detach(),
        "current_gradient": current_gradient.detach(),
        "weight_gradient": weight_gradient.detach(),
        "bias_gradient": bias_gradient.detach(),
        "teaching": teaching.detach(),
        "local_proxy": float(local_proxy.detach()),
        "parity_weight_max_abs": float(weight_difference.abs().max()),
        "parity_weight_relative": float(
            torch.linalg.vector_norm(weight_difference) / weight_denominator
        ),
        "parity_bias_max_abs": float(bias_difference.abs().max()),
        "parity_bias_relative": float(
            torch.linalg.vector_norm(bias_difference) / bias_denominator
        ),
    }


def capture_bptt_updates(
    model,
    samples: Tensor,
    target: Tensor,
) -> dict[str, Any]:
    """Backpropagate the true task loss through an unchanged sDFA checkpoint."""

    captured: list[Tensor] = []
    handles = []
    if getattr(model, "architecture", None) == "conv":
        current_modules = [layer.conv for layer in model.conv_layers] + [
            model.fc_hidden.linear
        ]
        weights = [layer.conv.weight for layer in model.conv_layers] + [
            model.fc_hidden.linear.weight
        ]
    else:
        current_modules = [layer.linear for layer in model.hidden_layers]
        weights = [layer.linear.weight for layer in model.hidden_layers]
    for module in current_modules:
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, destination=captured: destination.append(output)
            )
        )
    try:
        output, _hidden_inputs, _readout_input = model.forward_with_cache(
            samples, detach_temporal=False
        )
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(model.hidden_specs):
        raise RuntimeError(
            f"expected {len(model.hidden_specs)} BPTT currents, got {len(captured)}"
        )
    task_loss = _task_loss(output, target, model.num_classes)
    gradients = torch.autograd.grad(
        task_loss,
        tuple(captured) + tuple(weights),
        retain_graph=False,
        create_graph=False,
    )
    time_steps, batch_size = samples.shape[:2]
    current_gradients = []
    for value in gradients[: len(captured)]:
        if value.ndim == 2:
            current_gradients.append(
                value.detach().reshape(time_steps, batch_size, value.shape[-1])
            )
        elif value.ndim == 4:
            current_gradients.append(
                value.detach().reshape(time_steps, batch_size, *value.shape[1:])
            )
        else:
            raise ValueError(f"unsupported hidden current gradient shape: {tuple(value.shape)}")
    weight_gradients = [value.detach() for value in gradients[len(captured) :]]
    # Remove one global task-loss mean scalar for rank analysis only.  Weight
    # gradient alignment always uses the unscaled production gradients above.
    spectrum_scale = time_steps * batch_size
    deltas = [value * spectrum_scale for value in current_gradients]
    return {
        "task_loss": float(task_loss.detach()),
        "delta": deltas,
        "current_gradient": current_gradients,
        "weight_gradient": weight_gradients,
    }


def pca_basis(
    accumulator: OnlineCovariance,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return descending centered-PCA directions and Experiment 01 metrics."""

    symmetric = 0.5 * (accumulator.m2 + accumulator.m2.transpose(0, 1))
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            symmetric.to(device=device, dtype=torch.float32)
        )
    except RuntimeError:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        eigenvalues = eigenvalues.to(device=device, dtype=torch.float32)
        eigenvectors = eigenvectors.to(device=device, dtype=torch.float32)
    order = torch.arange(eigenvalues.numel() - 1, -1, -1, device=device)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min_(0)
    eigenvectors = eigenvectors.index_select(1, order).contiguous()
    statistics, _singular, _energy, _cumulative = spectrum_statistics(
        accumulator,
        algebraic_max_dim=accumulator.ambient_dim,
        device=device,
    )
    return eigenvectors, eigenvalues, statistics


def random_orthogonal_basis(
    ambient_dim: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    """Generate a deterministic Haar-like orthonormal frame via QR."""

    if not 1 <= rank <= ambient_dim:
        raise ValueError(f"rank {rank} outside [1,{ambient_dim}]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(ambient_dim, rank, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix.to(device), mode="reduced")
    signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    return q * signs.unsqueeze(0)


def _sign_agreement(projected: Tensor, full: Tensor) -> tuple[float, float, int]:
    threshold = max(1e-12, 1e-6 * float(full.abs().max()))
    valid = (full.abs() > threshold) | (projected.abs() > threshold)
    count = int(valid.sum())
    if count == 0:
        return float("nan"), threshold, 0
    agreement = (torch.sign(projected[valid]) == torch.sign(full[valid])).float().mean()
    return float(agreement), threshold, count


def project_update(
    delta: Tensor,
    weight_gradient: Tensor,
    bias_gradient: Tensor,
    basis: Tensor,
    *,
    norm_matching: bool,
) -> dict[str, Tensor | float]:
    """Project exact DFA current gradients and return weight-update metrics."""

    flat_delta = delta.reshape(-1, delta.shape[-1])
    coordinates = flat_delta @ basis
    full_delta_energy = flat_delta.square().sum()
    projected_delta_energy = coordinates.square().sum()
    scale = torch.ones((), device=delta.device, dtype=delta.dtype)
    if norm_matching:
        scale = torch.sqrt(
            full_delta_energy / (projected_delta_energy + torch.finfo(delta.dtype).eps)
        )
    projected_weight = basis @ (basis.transpose(0, 1) @ weight_gradient)
    projected_bias = basis @ (basis.transpose(0, 1) @ bias_gradient)
    projected_weight = projected_weight * scale
    projected_bias = projected_bias * scale

    full_norm = torch.linalg.vector_norm(weight_gradient)
    projected_norm = torch.linalg.vector_norm(projected_weight)
    inner = torch.sum(projected_weight * weight_gradient)
    cosine = inner / (projected_norm * full_norm + 1e-30)
    relative_error = torch.linalg.vector_norm(projected_weight - weight_gradient) / (
        full_norm + 1e-30
    )
    energy_ratio = projected_norm.square() / (full_norm.square() + 1e-30)
    sign_agreement, sign_threshold, sign_count = _sign_agreement(
        projected_weight, weight_gradient
    )
    return {
        "weight_gradient": projected_weight.detach(),
        "bias_gradient": projected_bias.detach(),
        "gradient_cosine": float(cosine),
        "relative_error": float(relative_error),
        "energy_ratio": float(energy_ratio),
        "sign_agreement": sign_agreement,
        "sign_threshold": sign_threshold,
        "sign_parameter_count": sign_count,
        "update_norm_ratio": float(projected_norm / (full_norm + 1e-30)),
        "delta_energy_ratio": float(
            projected_delta_energy / (full_delta_energy + 1e-30)
        ),
        "norm_match_scale": float(scale),
    }


def principal_subspace_overlap(
    basis_a: Tensor,
    basis_b: Tensor,
    rank: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return normalized overlap, principal angles in degrees, and cosines."""

    rank = min(int(rank), basis_a.shape[1], basis_b.shape[1])
    if rank < 1:
        raise ValueError("principal-subspace rank must be positive")
    cross = basis_a[:, :rank].transpose(0, 1) @ basis_b[:, :rank]
    cosines = torch.linalg.svdvals(cross).clamp_(0.0, 1.0)
    overlap = float(cosines.square().sum() / rank)
    angles = torch.rad2deg(torch.acos(cosines))
    return overlap, angles.cpu().numpy(), cosines.cpu().numpy()


@torch.no_grad()
def evaluate_sdfa_losses(
    model,
    feedback_bank,
    samples: Tensor,
    target: Tensor,
    *,
    detach_temporal: bool,
) -> dict[str, float]:
    """Evaluate the task and exact production proxy values without gradients."""

    output, hidden_inputs, _readout_input = model.forward_with_cache(
        samples, detach_temporal=detach_temporal
    )
    task_loss = _task_loss(output, target, model.num_classes)
    desired = F.one_hot(target, num_classes=model.num_classes).to(
        device=output.device, dtype=output.dtype
    )
    output_error = output.mean(dim=0) - desired
    local_losses = []
    for index, layer_input in enumerate(hidden_inputs):
        spikes = model.replay_hidden(
            index, layer_input.detach(), detach_temporal=detach_temporal
        )
        teaching = feedback_bank.project(index, output_error).detach()
        local_losses.append((spikes * teaching.unsqueeze(0)).mean())
    return {
        "task_loss": float(task_loss),
        "local_proxy_loss": float(torch.stack(local_losses).sum()),
    }


@contextmanager
def virtual_sgd_step(
    model,
    layer_index: int,
    weight_gradient: Tensor,
    bias_gradient: Tensor,
    learning_rate: float,
) -> Iterator[None]:
    """Apply and then exactly restore a layer-local SGD-equivalent update."""

    linear = model.hidden_layers[layer_index].linear
    original_weight = linear.weight.detach().clone()
    original_bias = linear.bias.detach().clone()
    with torch.no_grad():
        linear.weight.add_(weight_gradient, alpha=-float(learning_rate))
        linear.bias.add_(bias_gradient, alpha=-float(learning_rate))
    try:
        yield
    finally:
        with torch.no_grad():
            linear.weight.copy_(original_weight)
            linear.bias.copy_(original_bias)
