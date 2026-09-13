"""Projection alignment, factor orthogonality and Hoyer sparsity."""

from __future__ import annotations

import torch
from torch import Tensor

from methods.lodfa import LowRankFeedback


def _projector(basis: Tensor) -> Tensor:
    q = torch.linalg.qr(basis, mode="reduced").Q
    return q @ q.transpose(0, 1)


def feedback_regularization(
    feedback: LowRankFeedback,
    forward_weight: Tensor,
    epsilon: float = 1e-8,
) -> dict[str, Tensor]:
    rank = feedback.active_rank
    # W is [hidden/output channels, flattened input].  Its leading left singular
    # vectors span the forward output subspace used by the patent/paper projector.
    with torch.no_grad():
        detached_weight = forward_weight.detach()
        if detached_weight.numel() <= 1_000_000:
            q_weight = torch.linalg.svd(
                detached_weight, full_matrices=False
            ).U[:, :rank]
        else:
            # Exact full SVD is prohibitive for a wide conv-to-FC matrix.
            # A fixed-seed randomized SVD preserves the intended leading
            # subspace objective while keeping the regularizer practical.
            devices = (
                [detached_weight.device]
                if detached_weight.is_cuda
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(0)
                q_weight = torch.svd_lowrank(
                    detached_weight, q=rank, niter=1
                )[0][:, :rank]
    p_weight = q_weight @ q_weight.transpose(0, 1)
    p_feedback = _projector(feedback.B_U[:, :rank])
    alignment = (p_weight - p_feedback).square().sum()

    eye_u = torch.eye(rank, device=forward_weight.device, dtype=forward_weight.dtype)
    eye_v = torch.eye(rank, device=forward_weight.device, dtype=forward_weight.dtype)
    gram_u = feedback.B_U[:, :rank].transpose(0, 1) @ feedback.B_U[:, :rank]
    gram_v = feedback.B_V[:, :rank].transpose(0, 1) @ feedback.B_V[:, :rank]
    orthogonal = (gram_u - eye_u).square().sum() + (gram_v - eye_v).square().sum()

    singular = feedback.B_S[:rank]
    hoyer = singular.abs().sum() / (singular.square().sum().sqrt() + epsilon)
    return {"alignment": alignment, "orthogonal": orthogonal, "hoyer": hoyer}
