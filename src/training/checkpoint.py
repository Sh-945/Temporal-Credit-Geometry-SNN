"""Complete, resumable checkpoint state."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 1


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model,
    feedback_bank,
    optimizer,
    scheduler,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": FORMAT_VERSION,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "feedback_state": feedback_bank.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "metrics": metrics,
        "active_ranks": feedback_bank.active_ranks,
        "rng_state": _rng_state(),
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    *,
    model,
    feedback_bank,
    optimizer=None,
    scheduler=None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format: {state.get('format_version')}")
    model.load_state_dict(state["model_state"])
    feedback_bank.load_state_dict(state["feedback_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state["scheduler_state"] is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    if restore_rng:
        rng = state["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return state
