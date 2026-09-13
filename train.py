"""Single training entry point for BPTT, DFA/sDFA, PipeSDFA and LoDFA."""

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
from training.config import load_config
from training.data import build_loaders
from training.engine import Trainer
from training.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="validated dotted override"
    )
    parser.add_argument("--resume", help="last.pt or another complete checkpoint")
    parser.add_argument(
        "--smoke", action="store_true", help="replace data/model size with deterministic tiny data"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set, smoke=args.smoke)
    seed_everything(int(config["experiment"]["seed"]))
    device = choose_device(args.device)
    train_loader, test_loader = build_loaders(config)
    model = build_model(config)
    feedback_bank = FeedbackBank(model, config)
    trainer = Trainer(model, feedback_bank, config, device)
    if args.resume:
        trainer.resume(args.resume)
    summary = trainer.fit(train_loader, test_loader)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
