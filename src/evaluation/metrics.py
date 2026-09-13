from __future__ import annotations

import torch
from torch import Tensor


def classification_accuracy(mean_output: Tensor, target: Tensor) -> Tensor:
    return (mean_output.argmax(dim=1) == target).float().mean()
