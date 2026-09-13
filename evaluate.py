"""Single evaluation entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from methods import FeedbackBank
from models import build_model
from training.checkpoint import load_checkpoint
from training.config import load_config
from training.data import build_loaders
from training.engine import Trainer
from training.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config, args.set, smoke=args.smoke)
    seed_everything(int(config["experiment"]["seed"]))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    _, test_loader = build_loaders(config)
    model = build_model(config)
    bank = FeedbackBank(model, config)
    trainer = Trainer(model, bank, config, device)
    state = load_checkpoint(
        args.checkpoint,
        model=trainer.model,
        feedback_bank=trainer.feedback_bank,
        map_location=device,
    )
    print(
        json.dumps(
            {
                "checkpoint_epoch": state["epoch"],
                "metrics": trainer.evaluate(test_loader),
                "active_ranks": bank.active_ranks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
