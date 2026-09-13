"""Dataset adapters for local frame files and SHD event HDF5."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from training.config import resolve_repo_path


class SyntheticSpikeDataset(Dataset):
    def __init__(
        self, samples: int, time_steps: int, input_shape: list[int], classes: int, seed: int
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = (
            torch.rand(samples, time_steps, *input_shape, generator=generator) < 0.1
        ).float()
        self.targets = torch.randint(classes, (samples,), generator=generator)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]


class NPZFrameDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        labels: list[int],
        time_steps: int,
        resize: list[int] | None = None,
    ) -> None:
        if not files:
            raise FileNotFoundError("no NPZ frame files matched the requested split")
        self.files = files
        self.labels = labels
        self.time_steps = int(time_steps)
        self.resize = resize

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        with np.load(self.files[index]) as archive:
            if "frames" not in archive:
                raise KeyError(f"{self.files[index]} has no 'frames' array")
            frames = torch.from_numpy(archive["frames"]).float()
        if frames.shape[0] != self.time_steps:
            indices = torch.linspace(0, frames.shape[0] - 1, self.time_steps).round().long()
            frames = frames.index_select(0, indices)
        if self.resize is not None and list(frames.shape[-2:]) != list(self.resize):
            frames = F.interpolate(
                frames, size=self.resize, mode="bilinear", align_corners=False
            )
        return frames, torch.tensor(self.labels[index], dtype=torch.long)


class SHDDataset(Dataset):
    def __init__(self, path: Path, time_steps: int, input_units: int = 700) -> None:
        try:
            import h5py
        except ImportError as error:
            raise ImportError("h5py is required for the SHD dataset") from error
        self.path = path
        self.time_steps = int(time_steps)
        self.input_units = int(input_units)
        with h5py.File(path, "r") as handle:
            self.length = len(handle["labels"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        import h5py

        with h5py.File(self.path, "r") as handle:
            times = np.asarray(handle["spikes/times"][index], dtype=np.float32)
            units = np.asarray(handle["spikes/units"][index], dtype=np.int64)
            label = int(handle["labels"][index])
        frames = torch.zeros(self.time_steps, self.input_units)
        if len(times):
            maximum = max(float(times.max()), 1e-8)
            bins = np.minimum((times / maximum * self.time_steps).astype(np.int64), self.time_steps - 1)
            valid = (units >= 0) & (units < self.input_units)
            frames[bins[valid], units[valid]] = 1.0
        return frames, torch.tensor(label, dtype=torch.long)


def _class_folders(split_root: Path) -> tuple[list[Path], list[int]]:
    folders = sorted(
        (path for path in split_root.iterdir() if path.is_dir()),
        key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name),
    )
    files: list[Path] = []
    labels: list[int] = []
    for fallback_label, folder in enumerate(folders):
        label = int(folder.name) if folder.name.isdigit() else fallback_label
        matches = sorted(folder.glob("*.npz"))
        files.extend(matches)
        labels.extend([label] * len(matches))
    return files, labels


def _stable_partition(path: Path, test_fraction: float = 0.2) -> str:
    digest = hashlib.sha1(path.as_posix().encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return "test" if bucket < test_fraction else "train"


def _frame_dataset(config: dict[str, Any], split: str) -> NPZFrameDataset:
    data = config["data"]
    root = resolve_repo_path(data["root"])
    time_steps = int(data["time_steps"])
    dataset = data["dataset"]
    if dataset == "nmnist":
        base = root / "nmnist" / f"frames_number_{time_steps}_split_by_number"
    elif dataset == "dvs_gesture":
        base = root / "dvs" / f"frames_number_{time_steps}_split_by_number"
    elif dataset == "ncaltech101":
        base = root / "caltech101" / f"frames_number_{time_steps}_split_by_number"
    else:
        raise ValueError(dataset)
    split_root = base / split
    if split_root.is_dir():
        files, labels = _class_folders(split_root)
    elif dataset == "ncaltech101" and base.is_dir():
        all_files, all_labels = _class_folders(base)
        chosen = [
            index
            for index, path in enumerate(all_files)
            if _stable_partition(path, float(data.get("test_fraction", 0.2))) == split
        ]
        files = [all_files[index] for index in chosen]
        labels = [all_labels[index] for index in chosen]
    else:
        raise FileNotFoundError(
            f"expected frame dataset directory does not exist: {split_root}"
        )
    limit = data.get(f"{split}_limit")
    if limit is not None:
        files, labels = files[: int(limit)], labels[: int(limit)]
    return NPZFrameDataset(files, labels, time_steps, data.get("resize"))


def build_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    data = config["data"]
    dataset = data["dataset"]
    if dataset == "synthetic":
        shape = list(config["model"]["input_shape"])
        classes = int(config["model"]["num_classes"])
        seed = int(config["experiment"]["seed"])
        return (
            SyntheticSpikeDataset(
                int(data.get("train_samples", 32)),
                int(data["time_steps"]),
                shape,
                classes,
                seed,
            ),
            SyntheticSpikeDataset(
                int(data.get("test_samples", 16)),
                int(data["time_steps"]),
                shape,
                classes,
                seed + 1,
            ),
        )
    if dataset in {"nmnist", "dvs_gesture", "ncaltech101"}:
        return _frame_dataset(config, "train"), _frame_dataset(config, "test")
    if dataset == "shd":
        root = resolve_repo_path(data["root"]) / "hdspikes"
        steps = int(data["time_steps"])
        return SHDDataset(root / "shd_train.h5", steps), SHDDataset(
            root / "shd_test.h5", steps
        )
    raise ValueError(dataset)


def build_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = build_datasets(config)
    data = config["data"]
    common = {
        "batch_size": int(data["batch_size"]),
        "num_workers": int(data.get("num_workers", 0)),
        "pin_memory": bool(data.get("pin_memory", False)),
    }
    if common["num_workers"] > 0:
        common["persistent_workers"] = bool(data.get("persistent_workers", False))
        if data.get("prefetch_factor") is not None:
            common["prefetch_factor"] = int(data["prefetch_factor"])
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )
