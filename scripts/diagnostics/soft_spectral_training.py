"""Experiment 03: causal dynamic soft spectral-regulation training test."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.soft_spectral.report import generate_report  # noqa: E402
from analysis.soft_spectral.training import (  # noqa: E402
    Variant,
    baseline_parity,
    deterministic_data_bundle,
    make_loaders,
    materialize_dataset,
    rank_schedule_from_rows,
    run_variant,
    save_paired_initial_state,
    select_pilot_alpha,
)
from training.config import load_config  # noqa: E402
from training.data import build_datasets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/nmnist/experiment03_soft_spectral.yaml"
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stage", choices=("all", "pilot", "full", "report"), default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def write_environment(run_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        f"timestamp={datetime.now().astimezone().isoformat()}",
        f"command={' '.join(sys.argv)}",
        f"repository={ROOT}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda_runtime={torch.version.cuda}",
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        "nvidia_smi=",
        _command_output(["nvidia-smi"]),
    ]
    (run_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cache_path(run_dir: Path, stage: str, variant: Variant, seed: int) -> Path:
    return run_dir / "partial" / f"{stage}_{variant.key}_seed{seed}.pt"


def run_cached(
    *,
    stage: str,
    variant: Variant,
    seed: int,
    base_config: dict[str, Any],
    initial_state_path: Path,
    bundle,
    run_dir: Path,
    device: torch.device,
    forced_rank_schedule=None,
    run_test: bool,
):
    path = _cache_path(run_dir, stage, variant, seed)
    path.parent.mkdir(exist_ok=True)
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)
    result = run_variant(
        base_config=base_config,
        variant=variant,
        seed=seed,
        initial_state_path=initial_state_path,
        bundle=bundle,
        run_dir=run_dir,
        device=device,
        forced_rank_schedule=forced_rank_schedule,
        run_test=run_test,
    )
    torch.save(result, path)
    print(
        json.dumps(
            {
                "status": "variant_complete",
                "stage": stage,
                "method": variant.key,
                "seed": seed,
                "summary": result["summary"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def extend(all_rows: dict[str, list[dict[str, Any]]], result: dict[str, Any]) -> None:
    for key in all_rows:
        all_rows[key].extend(result.get(key, []))


def _alignment_by_layer(rows: list[dict[str, Any]], method: str, epoch: int) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    selected = frame[
        (frame.method == method)
        & (frame.epoch == epoch)
        & (frame.record_type == "weight_gradient")
    ]
    return selected.groupby("layer").gradient_cosine.mean().to_dict()


def _conditional_decisions(
    summaries: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.DataFrame(summaries)
    dense = frame[frame.method == "full_dense"].set_index("seed")
    soft = frame[frame.method == "full_soft"].set_index("seed")
    validation_gain = float(
        (soft.best_validation_accuracy - dense.best_validation_accuracy).mean()
    )
    dense_alignment = _alignment_by_layer(alignment_rows, "full_dense", 100)
    soft_alignment = _alignment_by_layer(alignment_rows, "full_soft", 100)
    relative = {
        layer: (soft_alignment[layer] - dense_alignment[layer])
        / max(abs(dense_alignment[layer]), 1e-12)
        for layer in dense_alignment
    }
    control = config["experiment03"]["automatic_controls"]
    alignment_20_layers = sum(
        value >= float(control["lrmatch_alignment_relative_gain"])
        for value in relative.values()
    )
    lrmatch = (
        validation_gain >= float(control["lrmatch_accuracy_gain_pp"]) / 100.0
        or alignment_20_layers >= 2
    )
    normmatched = (
        validation_gain
        >= -float(control["normmatched_validation_margin_pp"]) / 100.0
        and sum(
            value >= float(control["normmatched_alignment_relative_gain"])
            for value in relative.values()
        )
        >= 2
    )
    return {
        "validation_accuracy_gain_soft_minus_dense": validation_gain,
        "final_alignment_relative_gain_by_layer": relative,
        "trigger_lrmatch": bool(lrmatch),
        "trigger_normmatched": bool(normmatched),
        "thresholds": control,
    }


def _test_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for checkpoint, values in (
            ("final", summary.get("final_test")),
            ("best_validation", summary.get("best_validation_test")),
        ):
            if values is None:
                continue
            rows.append(
                {
                    "method": summary["method"],
                    "seed": summary["seed"],
                    "checkpoint": checkpoint,
                    "selection_basis": (
                        "final epoch" if checkpoint == "final" else "validation only"
                    ),
                    "test_loss": values["loss"],
                    "test_accuracy": values["accuracy"],
                    "best_validation_epoch": summary["best_validation_epoch"],
                }
            )
    return rows


def _paired_rows(summaries: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = pd.DataFrame(summaries)
    tests = pd.DataFrame(test_rows)
    rows = []
    for seed in sorted(summary.seed.unique()):
        dense = summary[(summary.seed == seed) & (summary.method == "full_dense")]
        if dense.empty:
            continue
        dense_row = dense.iloc[0]
        dense_test = tests[(tests.seed == seed) & (tests.method == "full_dense")]
        for method in sorted(summary[summary.seed == seed].method.unique()):
            if method == "full_dense":
                continue
            current = summary[(summary.seed == seed) & (summary.method == method)].iloc[0]
            current_test = tests[(tests.seed == seed) & (tests.method == method)]
            row = {
                "seed": seed,
                "method": method,
                "reference": "full_dense",
                "best_validation_accuracy_difference": current.best_validation_accuracy - dense_row.best_validation_accuracy,
                "convergence_epoch_difference": current.convergence_epoch_99pct_own_best - dense_row.convergence_epoch_99pct_own_best,
            }
            for checkpoint in ("final", "best_validation"):
                first = current_test[current_test.checkpoint == checkpoint]
                second = dense_test[dense_test.checkpoint == checkpoint]
                row[f"{checkpoint}_test_accuracy_difference"] = (
                    float(first.test_accuracy.iloc[0] - second.test_accuracy.iloc[0])
                    if not first.empty and not second.empty
                    else np.nan
                )
            rows.append(row)
    return rows


def _random_control_rows(
    summaries: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = pd.DataFrame(summaries)
    geo = pd.DataFrame(geometry)
    align = pd.DataFrame(alignment)
    rows = []
    for seed in sorted(summary.seed.unique()):
        soft = summary[(summary.seed == seed) & (summary.method == "full_soft")]
        random = summary[(summary.seed == seed) & (summary.method == "full_random")]
        if soft.empty or random.empty:
            continue
        for layer in ("hidden_1", "hidden_2", "hidden_3"):
            def value(frame, method, column):
                selected = frame[
                    (frame.method == method)
                    & (frame.seed == seed)
                    & (frame.epoch == 100)
                    & (frame.layer == layer)
                ]
                return float(selected[column].mean())
            rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "soft_minus_random_best_validation_accuracy": float(
                        soft.best_validation_accuracy.iloc[0]
                        - random.best_validation_accuracy.iloc[0]
                    ),
                    "soft_minus_random_bp_cosine": value(
                        align[align.record_type == "weight_gradient"], "full_soft", "gradient_cosine"
                    )
                    - value(
                        align[align.record_type == "weight_gradient"], "full_random", "gradient_cosine"
                    ),
                    "soft_minus_random_timestep_r95": value(
                        geo[(geo.signal_type == "filtered_dfa_delta") & (geo.temporal_mode == "timestep")],
                        "full_soft",
                        "r95",
                    )
                    - value(
                        geo[(geo.signal_type == "filtered_dfa_delta") & (geo.temporal_mode == "timestep")],
                        "full_random",
                        "r95",
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config_path = _absolute(args.config)
    config = load_config(config_path)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = _absolute(args.run_dir) if args.run_dir else (
        ROOT / "results" / "experiment03_soft_spectral" / f"experiment03_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("configs", "checkpoints", "figures", "initial_states", "partial"):
        (run_dir / folder).mkdir(exist_ok=True)
    shutil.copy2(config_path, run_dir / "configs" / "base_config.yaml")
    write_environment(run_dir, args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if args.stage == "report":
        verdict = generate_report(run_dir, config)
        print(json.dumps({"status": "report_complete", "verdict": verdict}))
        return

    if args.smoke:
        # Synthetic data keeps the exact causal plumbing while making CI cheap.
        smoke_config = config
        smoke_config["data"].update(
            {"dataset": "synthetic", "time_steps": 4, "train_samples": 48, "test_samples": 24, "batch_size": 8, "pin_memory": False}
        )
        smoke_config["model"]["input_shape"] = [2, 8, 8]
        smoke_config["model"]["hidden_features"] = [16, 12, 10]
        smoke_config["training"].update({"epochs": 2, "lr_step": 0})
        config = smoke_config
    raw_train, raw_test = build_datasets(config)
    cached_train = materialize_dataset(raw_train)
    cached_test = materialize_dataset(raw_test)
    experiment = config["experiment03"]
    calibration_size = 8 if args.smoke else int(experiment["calibration_size"])
    diagnostic_size = 8 if args.smoke else int(experiment["diagnostic_probe_size"])
    bundle = deterministic_data_bundle(
        cached_train,
        cached_test,
        split_seed=int(experiment["split_seed"]),
        calibration_seed=int(experiment["calibration_seed"]),
        validation_fraction=float(experiment["validation_fraction"]),
        calibration_size=calibration_size,
        diagnostic_size=diagnostic_size,
    )
    pd.DataFrame(bundle.split_rows).to_csv(run_dir / "data_split.csv", index=False)
    pd.DataFrame(bundle.calibration_rows).to_csv(
        run_dir / "spectral_calibration_indices.csv", index=False
    )
    pd.DataFrame(bundle.diagnostic_rows).to_csv(
        run_dir / "diagnostic_probe_indices.csv", index=False
    )
    seeds = [int(experiment["pilot_seed"])] if args.smoke else [
        int(value) for value in experiment["full_seeds"]
    ]
    initialization_rows = []
    for seed in seeds:
        path = run_dir / "initial_states" / f"seed_{seed}.pt"
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=False)
            initialization_rows.append(
                {key: value for key, value in state.items() if not key.endswith("_state")}
            )
        else:
            initialization_rows.append(
                save_paired_initial_state(config=config, seed=seed, path=path)
            )
    pd.DataFrame(initialization_rows).to_csv(
        run_dir / "paired_initialization.csv", index=False
    )
    pilot_seed = int(experiment["pilot_seed"])
    parity_loader = make_loaders(bundle, config, seed=pilot_seed)["train"]
    parity_rows = baseline_parity(
        config=config,
        initial_state_path=run_dir / "initial_states" / f"seed_{pilot_seed}.pt",
        batch=next(iter(parity_loader)),
        device=device,
    )
    pd.DataFrame(parity_rows).to_csv(run_dir / "baseline_parity.csv", index=False)
    if not all(row["passed"] for row in parity_rows):
        raise AssertionError("alpha=1 baseline parity failed; formal experiment stopped")

    row_keys = (
        "training_metrics",
        "rank_schedule",
        "basis_rotation",
        "runtime_metrics",
        "spectral_geometry",
        "bp_alignment",
        "gradient_norms",
        "gate_dynamics",
    )
    all_rows = {key: [] for key in row_keys}
    summaries: list[dict[str, Any]] = []
    pilot_epochs = 2 if args.smoke else int(experiment["pilot_epochs"])
    pilot_variants = [
        Variant("pilot_dense", "Dense DFA", 1.0, "dense", pilot_epochs),
        Variant("pilot_hard", "Hard dynamic-r95", 0.0, "learned", pilot_epochs),
        Variant("pilot_soft_025", "Soft alpha=0.25", 0.25, "learned", pilot_epochs),
        Variant("pilot_soft_050", "Soft alpha=0.50", 0.50, "learned", pilot_epochs),
        Variant("pilot_soft_075", "Soft alpha=0.75", 0.75, "learned", pilot_epochs),
    ]
    if args.stage in {"all", "pilot"}:
        for variant in pilot_variants:
            result = run_cached(
                stage="pilot",
                variant=variant,
                seed=pilot_seed,
                base_config=config,
                initial_state_path=run_dir / "initial_states" / f"seed_{pilot_seed}.pt",
                bundle=bundle,
                run_dir=run_dir,
                device=device,
                run_test=False,
            )
            extend(all_rows, result)
            summaries.append(result["summary"])
        soft050_schedule = rank_schedule_from_rows(
            all_rows["rank_schedule"], "pilot_soft_050", pilot_seed
        )
        random_variant = Variant(
            "pilot_random_050", "Random soft alpha=0.50", 0.50, "random", pilot_epochs
        )
        result = run_cached(
            stage="pilot",
            variant=random_variant,
            seed=pilot_seed,
            base_config=config,
            initial_state_path=run_dir / "initial_states" / f"seed_{pilot_seed}.pt",
            bundle=bundle,
            run_dir=run_dir,
            device=device,
            forced_rank_schedule=soft050_schedule,
            run_test=False,
        )
        extend(all_rows, result)
        summaries.append(result["summary"])
        if args.smoke:
            selection = {
                "criterion": "smoke plumbing only",
                "test_used_for_selection": False,
                "selected_alpha": 0.5,
                "selected_method": "pilot_soft_050",
                "candidates": [],
            }
        else:
            selection = select_pilot_alpha(all_rows["training_metrics"])
        (run_dir / "pilot_selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pilot_training = pd.DataFrame(all_rows["training_metrics"])
        pilot_summary_rows = []
        for summary in summaries:
            if not str(summary["method"]).startswith("pilot_"):
                continue
            tail = pilot_training[
                (pilot_training.method == summary["method"])
                & pilot_training.epoch.between(max(1, pilot_epochs - 9), pilot_epochs)
            ]
            pilot_summary_rows.append(
                {
                    **summary,
                    "selection_window_start": max(1, pilot_epochs - 9),
                    "selection_window_end": pilot_epochs,
                    "mean_validation_accuracy_selection_window": tail.validation_accuracy.mean(),
                    "mean_validation_loss_selection_window": tail.validation_loss.mean(),
                    "eligible_alpha_candidate": summary["method"] in {
                        "pilot_soft_025", "pilot_soft_050", "pilot_soft_075"
                    },
                    "selected": summary["method"] == selection["selected_method"],
                    "test_used_for_selection": False,
                }
            )
        pd.DataFrame(pilot_summary_rows).to_csv(
            run_dir / "pilot_results.csv", index=False
        )
    else:
        selection = json.loads((run_dir / "pilot_selection.json").read_text(encoding="utf-8"))
        for path in sorted((run_dir / "partial").glob("pilot_*.pt")):
            result = torch.load(path, map_location="cpu", weights_only=False)
            extend(all_rows, result)
            summaries.append(result["summary"])

    if args.stage == "pilot":
        for key, rows in all_rows.items():
            pd.DataFrame(rows).to_csv(run_dir / f"{key}.csv", index=False)
        print(json.dumps({"status": "pilot_complete", "selection": selection}))
        return

    selected_alpha = float(selection["selected_alpha"])
    full_epochs = 2 if args.smoke else 100
    if args.stage in {"all", "full"}:
        for seed in seeds:
            initial_path = run_dir / "initial_states" / f"seed_{seed}.pt"
            full_variants = [
                Variant("full_dense", "Dense DFA", 1.0, "dense", full_epochs),
                Variant("full_hard", "Hard dynamic-r95", 0.0, "learned", full_epochs),
                Variant("full_soft", f"Soft alpha={selected_alpha:.2f}", selected_alpha, "learned", full_epochs),
            ]
            for variant in full_variants:
                result = run_cached(
                    stage="full",
                    variant=variant,
                    seed=seed,
                    base_config=config,
                    initial_state_path=initial_path,
                    bundle=bundle,
                    run_dir=run_dir,
                    device=device,
                    run_test=True,
                )
                extend(all_rows, result)
                summaries.append(result["summary"])
            soft_schedule = rank_schedule_from_rows(
                all_rows["rank_schedule"], "full_soft", seed
            )
            for variant in (
                Variant("full_random", f"Random soft alpha={selected_alpha:.2f}", selected_alpha, "random", full_epochs),
                Variant("full_bptt", "Matched BPTT", None, "bptt", full_epochs),
            ):
                result = run_cached(
                    stage="full",
                    variant=variant,
                    seed=seed,
                    base_config=config,
                    initial_state_path=initial_path,
                    bundle=bundle,
                    run_dir=run_dir,
                    device=device,
                    forced_rank_schedule=(soft_schedule if variant.basis_kind == "random" else None),
                    run_test=True,
                )
                extend(all_rows, result)
                summaries.append(result["summary"])

        soft_training_norms = pd.DataFrame(all_rows["gradient_norms"])
        selected_norms = soft_training_norms[
            (soft_training_norms.method == "full_soft")
            & (soft_training_norms.record_type == "training_batch")
        ]
        s_global = float(selected_norms.weight_gradient_norm_ratio.median())
        decisions = (
            {
                "validation_accuracy_gain_soft_minus_dense": float("nan"),
                "final_alignment_relative_gain_by_layer": {},
                "trigger_lrmatch": False,
                "trigger_normmatched": False,
                "thresholds": config["experiment03"]["automatic_controls"],
            }
            if args.smoke
            else _conditional_decisions(
                summaries, all_rows["bp_alignment"], config
            )
        )
        decisions["s_global_median_weight_gradient_norm_ratio"] = s_global
        controls_run = []
        if decisions["trigger_lrmatch"]:
            controls_run.append("full_dense_lrmatch")
            variant = Variant(
                "full_dense_lrmatch",
                f"Dense DFA LRMatch x{s_global:.4f}",
                1.0,
                "dense",
                full_epochs,
                learning_rate_scale=s_global,
            )
            for seed in seeds:
                result = run_cached(
                    stage="control",
                    variant=variant,
                    seed=seed,
                    base_config=config,
                    initial_state_path=run_dir / "initial_states" / f"seed_{seed}.pt",
                    bundle=bundle,
                    run_dir=run_dir,
                    device=device,
                    run_test=True,
                )
                extend(all_rows, result)
                summaries.append(result["summary"])
        if decisions["trigger_normmatched"]:
            controls_run.append("full_soft_normmatched")
            variant = Variant(
                "full_soft_normmatched",
                f"Soft NormMatched alpha={selected_alpha:.2f}",
                selected_alpha,
                "learned",
                full_epochs,
                norm_matched=True,
            )
            for seed in seeds:
                result = run_cached(
                    stage="control",
                    variant=variant,
                    seed=seed,
                    base_config=config,
                    initial_state_path=run_dir / "initial_states" / f"seed_{seed}.pt",
                    bundle=bundle,
                    run_dir=run_dir,
                    device=device,
                    run_test=True,
                )
                extend(all_rows, result)
                summaries.append(result["summary"])
        decisions["controls_run"] = controls_run
        (run_dir / "automatic_control_decisions.json").write_text(
            json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    for key, rows in all_rows.items():
        pd.DataFrame(rows).to_csv(run_dir / f"{key}.csv", index=False)
    pd.DataFrame(summaries).to_csv(run_dir / "variant_summaries.csv", index=False)
    test_rows = _test_rows(summaries)
    pd.DataFrame(test_rows).to_csv(run_dir / "test_results.csv", index=False)
    paired_rows = _paired_rows(summaries, test_rows)
    pd.DataFrame(paired_rows).to_csv(
        run_dir / "paired_seed_differences.csv", index=False
    )
    random_rows = (
        []
        if args.smoke
        else _random_control_rows(
            summaries, all_rows["spectral_geometry"], all_rows["bp_alignment"]
        )
    )
    pd.DataFrame(random_rows).to_csv(run_dir / "random_control.csv", index=False)
    verdict = None if args.smoke else generate_report(run_dir, config)
    manifest = {
        "status": "smoke_complete" if args.smoke else "complete",
        "run_dir": str(run_dir),
        "pilot_selection": selection,
        "model_selection_based_only_on_validation": True,
        "test_evaluation_schedule": ["final", "best_validation"],
        "paired_seeds": seeds,
        "future_basis_leakage": False,
        "basis_lag_rule": "epoch t training uses basis fitted after epoch t-1",
        "calibration_is_training_only": True,
        "hypotheses_preregistered": ["H1", "H2", "H3", "H4", "H5"],
        "verdict": verdict,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "verdict": verdict, "run_dir": str(run_dir)}), flush=True)


if __name__ == "__main__":
    main()
