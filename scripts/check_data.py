"""Validate expected files and one decoded sample without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.config import load_config
from training.data import build_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    train, test = build_datasets(config)
    sample, label = train[0]
    test_sample, test_label = test[0]
    print(
        json.dumps(
            {
                "dataset": config["data"]["dataset"],
                "train_samples": len(train),
                "test_samples": len(test),
                "train_shape": list(sample.shape),
                "train_label": int(label),
                "test_shape": list(test_sample.shape),
                "test_label": int(test_label),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
