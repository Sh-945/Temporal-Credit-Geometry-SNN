"""Causal diagnostics linking teaching-signal subspaces to weight updates."""

from .core import (
    capture_bptt_updates,
    capture_dfa_layer_update,
    evaluate_sdfa_losses,
    pca_basis,
    principal_subspace_overlap,
    project_update,
    random_orthogonal_basis,
    virtual_sgd_step,
)

__all__ = [
    "capture_bptt_updates",
    "capture_dfa_layer_update",
    "evaluate_sdfa_losses",
    "pca_basis",
    "principal_subspace_overlap",
    "project_update",
    "random_orthogonal_basis",
    "virtual_sgd_step",
]
