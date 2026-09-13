"""Materialize one immutable frame dataset once for concurrent local workers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.soft_spectral.training import materialize_dataset
from training.config import load_config
from training.data import build_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel NPZ readers used only while building the immutable cache.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    output = Path(args.output).resolve()
    train, test = build_datasets(load_config(config_path))
    cached = {
        "train": materialize_dataset(train, num_workers=args.num_workers),
        "test": materialize_dataset(test, num_workers=args.num_workers),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "train_samples": len(cached["train"]),
                "test_samples": len(cached["test"]),
                "num_workers": args.num_workers,
                "bytes": output.stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
