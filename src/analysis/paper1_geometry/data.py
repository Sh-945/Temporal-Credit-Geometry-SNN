"""Deterministic split and indexed loaders for Paper 1 Experiment 03."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class IndexedSubset(Dataset):
    """A subset that also returns the immutable source-dataset index."""

    def __init__(self, dataset: Dataset, indices: list[int] | np.ndarray) -> None:
        self.dataset = dataset
        self.indices = [int(value) for value in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        source_index = self.indices[position]
        sample, label = self.dataset[source_index]
        return sample, label, torch.tensor(source_index, dtype=torch.long)


@dataclass
class Paper1Split:
    training_indices: np.ndarray
    validation_indices: np.ndarray
    probe_indices: np.ndarray
    basis_fit_indices: np.ndarray
    basis_eval_indices: np.ndarray

    def rows(self) -> list[dict[str, int | str]]:
        rows: list[dict[str, int | str]] = []
        rows.extend(
            {"dataset_index": int(index), "split": "training", "probe_position": -1}
            for index in self.training_indices
        )
        rows.extend(
            {"dataset_index": int(index), "split": "validation", "probe_position": -1}
            for index in self.validation_indices
        )
        for position, index in enumerate(self.probe_indices):
            rows.append(
                {
                    "dataset_index": int(index),
                    "split": "basis_fit" if position < len(self.basis_fit_indices) else "basis_eval",
                    "probe_position": position,
                }
            )
        return rows


def build_paper1_split(
    dataset_length: int,
    *,
    split_seed: int = 20260830,
    probe_seed: int = 20260831,
    validation_fraction: float = 0.10,
    probe_size: int = 1024,
) -> Paper1Split:
    """Match the formal 90/10 split and retain random probe order.

    Earlier utilities sorted the sampled probe indices for I/O locality.  That
    is unsafe for a class-folder dataset when the first and second 512 samples
    are designated fit/eval subsets, so Paper 1 preserves the seeded random
    order explicitly.
    """

    if probe_size < 2 or probe_size % 2:
        raise ValueError("probe_size must be an even integer")
    split_rng = np.random.default_rng(int(split_seed))
    permutation = split_rng.permutation(int(dataset_length))
    validation_count = max(1, int(round(dataset_length * validation_fraction)))
    validation = np.sort(permutation[:validation_count])
    training = np.sort(permutation[validation_count:])
    if len(training) < probe_size:
        raise ValueError(
            f"training split has {len(training)} samples, fewer than probe_size={probe_size}"
        )
    probe_rng = np.random.default_rng(int(probe_seed))
    probe = probe_rng.permutation(training)[:probe_size]
    half = probe_size // 2
    fit, evaluation = probe[:half].copy(), probe[half:].copy()
    if len(set(map(int, training)).intersection(map(int, validation))) != 0:
        raise AssertionError("training and validation overlap")
    if not set(map(int, probe)).issubset(set(map(int, training))):
        raise AssertionError("probe is not training-only")
    if set(map(int, fit)).intersection(map(int, evaluation)):
        raise AssertionError("basis_fit and basis_eval overlap")
    return Paper1Split(training, validation, probe, fit, evaluation)


def make_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    split: Paper1Split,
    config: dict[str, Any],
    *,
    seed: int,
    workers: int = 0,
) -> dict[str, DataLoader]:
    batch_size = int(config["data"]["batch_size"])
    common = {
        "batch_size": batch_size,
        "num_workers": int(workers),
        "pin_memory": bool(config["data"].get("pin_memory", False)),
    }
    generator = torch.Generator().manual_seed(int(seed))
    return {
        "train": DataLoader(
            IndexedSubset(train_dataset, split.training_indices),
            shuffle=True,
            generator=generator,
            **common,
        ),
        "validation": DataLoader(
            IndexedSubset(train_dataset, split.validation_indices),
            shuffle=False,
            **common,
        ),
        "test": DataLoader(
            IndexedSubset(test_dataset, np.arange(len(test_dataset))),
            shuffle=False,
            **common,
        ),
        "basis_fit": DataLoader(
            IndexedSubset(train_dataset, split.basis_fit_indices),
            shuffle=False,
            **common,
        ),
        "basis_eval": DataLoader(
            IndexedSubset(train_dataset, split.basis_eval_indices),
            shuffle=False,
            **common,
        ),
    }
