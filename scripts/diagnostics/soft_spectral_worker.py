"""Run one independently cacheable Experiment 03 variant.

This entry point exists only to schedule causally independent method/seed pairs
concurrently on an under-utilised GPU.  It delegates all scientific work to the
same ``run_variant``/``run_cached`` implementation as the serial orchestrator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.soft_spectral.training import (  # noqa: E402
    Variant,
    deterministic_data_bundle,
    materialize_dataset,
    rank_schedule_from_rows,
)
from scripts.diagnostics.soft_spectral_training import run_cached  # noqa: E402
from training.config import load_config  # noqa: E402
from training.data import build_datasets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/nmnist/experiment03_soft_spectral.yaml"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("pilot", "full", "control"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selected-alpha", type=float)
    parser.add_argument("--learning-rate-scale", type=float, default=1.0)
    return parser.parse_args()


def absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def build_variant(args: argparse.Namespace, epochs: int) -> Variant:
    alpha = args.selected_alpha
    alpha_label = "UNSET" if alpha is None else f"{alpha:.2f}"
    variants = {
        "pilot_dense": Variant("pilot_dense", "Dense DFA", 1.0, "dense", epochs),
        "pilot_hard": Variant("pilot_hard", "Hard dynamic-r95", 0.0, "learned", epochs),
        "pilot_soft_025": Variant("pilot_soft_025", "Soft alpha=0.25", 0.25, "learned", epochs),
        "pilot_soft_050": Variant("pilot_soft_050", "Soft alpha=0.50", 0.50, "learned", epochs),
        "pilot_soft_075": Variant("pilot_soft_075", "Soft alpha=0.75", 0.75, "learned", epochs),
        "pilot_random_050": Variant("pilot_random_050", "Random soft alpha=0.50", 0.50, "random", epochs),
        "full_dense": Variant("full_dense", "Dense DFA", 1.0, "dense", epochs),
        "full_hard": Variant("full_hard", "Hard dynamic-r95", 0.0, "learned", epochs),
        "full_soft": Variant("full_soft", f"Soft alpha={alpha_label}", alpha, "learned", epochs),
        "full_random": Variant("full_random", f"Random soft alpha={alpha_label}", alpha, "random", epochs),
        "full_bptt": Variant("full_bptt", "Matched BPTT", None, "bptt", epochs),
        "full_dense_lrmatch": Variant(
            "full_dense_lrmatch",
            f"Dense DFA LRMatch x{args.learning_rate_scale:.4f}",
            1.0,
            "dense",
            epochs,
            learning_rate_scale=args.learning_rate_scale,
        ),
        "full_soft_normmatched": Variant(
            "full_soft_normmatched",
            f"Soft NormMatched alpha={alpha_label}",
            alpha,
            "learned",
            epochs,
            norm_matched=True,
        ),
    }
    if args.variant not in variants:
        raise ValueError(f"unknown variant: {args.variant}")
    variant = variants[args.variant]
    if variant.alpha is None and variant.basis_kind != "bptt":
        raise ValueError(f"--selected-alpha is required for {args.variant}")
    return variant


def main() -> None:
    args = parse_args()
    config = load_config(absolute(args.config))
    run_dir = absolute(args.run_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    experiment = config["experiment03"]
    epochs = int(experiment["pilot_epochs"]) if args.stage == "pilot" else 100
    variant = build_variant(args, epochs)
    raw_train, raw_test = build_datasets(config)
    bundle = deterministic_data_bundle(
        materialize_dataset(raw_train),
        materialize_dataset(raw_test),
        split_seed=int(experiment["split_seed"]),
        calibration_seed=int(experiment["calibration_seed"]),
        validation_fraction=float(experiment["validation_fraction"]),
        calibration_size=int(experiment["calibration_size"]),
        diagnostic_size=int(experiment["diagnostic_probe_size"]),
    )
    forced = None
    dependency = None
    if variant.key == "pilot_random_050":
        dependency = run_dir / "partial" / f"pilot_pilot_soft_050_seed{args.seed}.pt"
        source_method = "pilot_soft_050"
    elif variant.key == "full_random":
        dependency = run_dir / "partial" / f"full_full_soft_seed{args.seed}.pt"
        source_method = "full_soft"
    if dependency is not None:
        if not dependency.is_file():
            raise FileNotFoundError(f"required soft-rank dependency missing: {dependency}")
        source = torch.load(dependency, map_location="cpu", weights_only=False)
        forced = rank_schedule_from_rows(source["rank_schedule"], source_method, args.seed)
    result = run_cached(
        stage=args.stage,
        variant=variant,
        seed=args.seed,
        base_config=config,
        initial_state_path=run_dir / "initial_states" / f"seed_{args.seed}.pt",
        bundle=bundle,
        run_dir=run_dir,
        device=device,
        forced_rank_schedule=forced,
        run_test=args.stage != "pilot",
    )
    print(
        {
            "status": "worker_complete",
            "stage": args.stage,
            "variant": variant.key,
            "seed": args.seed,
            "epochs_completed": result["summary"]["epochs_completed"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
