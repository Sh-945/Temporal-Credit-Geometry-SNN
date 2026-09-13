"""Cacheable worker for Paper 1 Experiment 03 stages A-C."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paper1_geometry.data import build_paper1_split, make_loaders  # noqa: E402
from analysis.paper1_geometry.diagnostics import (  # noqa: E402
    diagnose_ann_checkpoint,
    diagnose_snn_checkpoint,
)
from analysis.paper1_geometry.training import (  # noqa: E402
    GateInterventionTrainer,
    IndexedTrainer,
    TemporalANNTrainer,
    build_matched_ann,
    train_with_validation,
)
from analysis.soft_spectral.training import load_paired_model, materialize_dataset  # noqa: E402
from methods import FeedbackBank  # noqa: E402
from methods.gate_intervention import GateMode  # noqa: E402
from models import build_model  # noqa: E402
from models.temporal_ann_control import TemporalANN  # noqa: E402
from training.config import load_config  # noqa: E402
from training.data import build_datasets  # noqa: E402


def absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("train_ann", "train_gate", "train_snn", "diagnose_snn", "diagnose_ann"),
    )
    parser.add_argument("--config", default="configs/nmnist/experiment03_soft_spectral.yaml")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-state")
    parser.add_argument("--checkpoint")
    parser.add_argument("--q-reference-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--gate-mode", default=GateMode.ACTUAL.value)
    parser.add_argument(
        "--gate-layers",
        help="Comma-separated one-based hidden-layer indices; omitted means all layers.",
    )
    parser.add_argument("--is-bptt", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def _gate_layers(value: str | None) -> set[int] | None:
    if value is None:
        return None
    layers = {int(item.strip()) - 1 for item in value.split(",") if item.strip()}
    if not layers or min(layers) < 0:
        raise ValueError("--gate-layers must contain positive one-based indices")
    return layers


def _split_and_loaders(config, *, seed: int, materialize: bool):
    raw_train, raw_test = build_datasets(config)
    split = build_paper1_split(
        len(raw_train),
        split_seed=int(config.get("experiment03", {}).get("split_seed", 20260830)),
        probe_seed=20260831,
        validation_fraction=float(config.get("experiment03", {}).get("validation_fraction", .10)),
        probe_size=int(config.get("experiment03", {}).get("probe_size", 1024)),
    )
    if materialize:
        cache_path = os.environ.get("PAPER1_MATERIALIZED_CACHE")
        if cache_path and Path(cache_path).is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
            raw_train, raw_test = cached["train"], cached["test"]
        else:
            raw_train = materialize_dataset(raw_train)
            raw_test = materialize_dataset(raw_test)
    return split, make_loaders(raw_train, raw_test, split, config, seed=seed)


def _shared_materialized_cache_available() -> bool:
    cache_path = os.environ.get("PAPER1_MATERIALIZED_CACHE")
    return bool(cache_path and Path(cache_path).is_file())


def _write_diagnostic(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "geometry",
        "basis_generalization",
        "weight_updates",
        "subspace_comparisons",
        "parity",
    ):
        pd.DataFrame(result[name]).to_csv(output_dir / f"{name}.csv", index=False)
    torch.save(result["bases"], output_dir / "bases.pt")
    (output_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "rows": {name: len(result[name]) for name in result if name != "bases"}}, indent=2),
        encoding="utf-8",
    )


def train_ann(args, base_config, device: torch.device) -> None:
    if not args.initial_state:
        raise ValueError("train_ann requires --initial-state")
    config = copy.deepcopy(base_config)
    config["experiment"]["seed"] = args.seed
    config["experiment"]["name"] = f"paper1_temporal_ann_seed{args.seed}"
    config["method"]["name"] = "sdfa"
    config.setdefault("paper1", {})["model_kind"] = "temporal_ann"
    _split, loaders = _split_and_loaders(config, seed=args.seed, materialize=True)
    initial = torch.load(absolute(args.initial_state), map_location="cpu", weights_only=False)
    model, bank = build_matched_ann(config, initial, device)
    trainer = TemporalANNTrainer(model, bank, config, device)
    result = train_with_validation(
        trainer=trainer,
        loaders=loaders,
        config=config,
        output_dir=absolute(args.output_dir),
        method=args.method,
        seed=args.seed,
        epochs=args.epochs,
    )
    print(json.dumps({"status": "complete", **result["summary"]}), flush=True)


def train_gate(args, base_config, device: torch.device) -> None:
    if not args.initial_state:
        raise ValueError("train_gate requires --initial-state")
    config = copy.deepcopy(base_config)
    config["experiment"]["seed"] = args.seed
    config["experiment"]["name"] = f"paper1_{args.method}_seed{args.seed}"
    config["method"]["name"] = "sdfa"
    config.setdefault("paper1", {})["gate_mode"] = args.gate_mode
    config["paper1"]["gate_layers"] = (
        None
        if args.gate_layers is None
        else sorted(index + 1 for index in _gate_layers(args.gate_layers))
    )
    _split, loaders = _split_and_loaders(config, seed=args.seed, materialize=True)
    model, bank, _initial = load_paired_model(
        config=config,
        initial_state_path=absolute(args.initial_state),
        device=device,
    )
    trainer = GateInterventionTrainer(
        model,
        bank,
        config,
        device,
        gate_mode=args.gate_mode,
        seed=args.seed,
        intervention_layers=_gate_layers(args.gate_layers),
    )
    result = train_with_validation(
        trainer=trainer,
        loaders=loaders,
        config=config,
        output_dir=absolute(args.output_dir),
        method=args.method,
        seed=args.seed,
        epochs=args.epochs,
    )
    print(json.dumps({"status": "complete", **result["summary"]}), flush=True)


def train_snn(args, base_config, device: torch.device) -> None:
    """Train unchanged dense-DFA or matched-BPTT SNNs on indexed data."""

    if not args.initial_state:
        raise ValueError("train_snn requires --initial-state")
    config = copy.deepcopy(base_config)
    config["experiment"]["seed"] = args.seed
    config["experiment"]["name"] = f"paper1_{args.method}_seed{args.seed}"
    config["method"]["name"] = "bptt" if args.is_bptt else "sdfa"
    config.setdefault("paper1", {})["model_kind"] = "snn"
    _split, loaders = _split_and_loaders(config, seed=args.seed, materialize=True)
    model, bank, _initial = load_paired_model(
        config=config,
        initial_state_path=absolute(args.initial_state),
        device=device,
    )
    trainer = IndexedTrainer(model, bank, config, device)
    result = train_with_validation(
        trainer=trainer,
        loaders=loaders,
        config=config,
        output_dir=absolute(args.output_dir),
        method=args.method,
        seed=args.seed,
        epochs=args.epochs,
    )
    print(json.dumps({"status": "complete", **result["summary"]}), flush=True)


def diagnose_snn(args, device: torch.device) -> None:
    if not args.checkpoint:
        raise ValueError("diagnose_snn requires --checkpoint")
    state = torch.load(absolute(args.checkpoint), map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(config).to(device)
    bank = FeedbackBank(model, config).to(device)
    model.load_state_dict(state["model_state"])
    bank.load_state_dict(state["feedback_state"])
    reference_model = reference_bank = None
    if args.q_reference_checkpoint:
        reference_state = torch.load(
            absolute(args.q_reference_checkpoint), map_location="cpu", weights_only=False
        )
        reference_model = build_model(reference_state["config"]).to(device)
        reference_bank = FeedbackBank(reference_model, reference_state["config"]).to(device)
        reference_model.load_state_dict(reference_state["model_state"])
        reference_bank.load_state_dict(reference_state["feedback_state"])
    _split, loaders = _split_and_loaders(
        config,
        seed=args.seed,
        materialize=_shared_materialized_cache_available(),
    )
    result = diagnose_snn_checkpoint(
        model=model,
        feedback_bank=bank,
        loaders=loaders,
        device=device,
        dataset=str(config["data"]["dataset"]),
        method=args.method,
        seed=args.seed,
        epoch=args.epoch,
        is_bptt=args.is_bptt,
        gate_mode=args.gate_mode,
        intervention_layers=_gate_layers(args.gate_layers),
        q_reference_model=reference_model,
        q_reference_feedback_bank=reference_bank,
    )
    _write_diagnostic(absolute(args.output_dir), result)
    print(json.dumps({"status": "complete", "mode": "diagnose_snn", "method": args.method, "seed": args.seed, "epoch": args.epoch}), flush=True)


def diagnose_ann(args, device: torch.device) -> None:
    if not args.checkpoint:
        raise ValueError("diagnose_ann requires --checkpoint")
    state = torch.load(absolute(args.checkpoint), map_location="cpu", weights_only=False)
    config = state["config"]
    model = TemporalANN(
        list(config["model"]["input_shape"]),
        list(config["model"]["hidden_features"]),
        int(config["model"]["num_classes"]),
    ).to(device)
    bank = FeedbackBank(model, config).to(device)
    model.load_state_dict(state["model_state"])
    bank.load_state_dict(state["feedback_state"])
    _split, loaders = _split_and_loaders(
        config,
        seed=args.seed,
        materialize=_shared_materialized_cache_available(),
    )
    result = diagnose_ann_checkpoint(
        model=model,
        feedback_bank=bank,
        loaders=loaders,
        device=device,
        dataset=str(config["data"]["dataset"]),
        method=args.method,
        seed=args.seed,
        epoch=args.epoch,
    )
    _write_diagnostic(absolute(args.output_dir), result)
    print(json.dumps({"status": "complete", "mode": "diagnose_ann", "method": args.method, "seed": args.seed, "epoch": args.epoch}), flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = load_config(absolute(args.config))
    if args.mode == "train_ann":
        train_ann(args, config, device)
    elif args.mode == "train_gate":
        train_gate(args, config, device)
    elif args.mode == "train_snn":
        train_snn(args, config, device)
    elif args.mode == "diagnose_snn":
        diagnose_snn(args, device)
    else:
        diagnose_ann(args, device)


if __name__ == "__main__":
    main()
