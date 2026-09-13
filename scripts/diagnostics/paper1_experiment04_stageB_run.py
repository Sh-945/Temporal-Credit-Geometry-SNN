"""Run Experiment 04 DVS offline decompositions and H3-only causal training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paper1_geometry.diagnostics import _capture_snn_dfa_layer  # noqa: E402
from analysis.paper1_geometry.training import GateInterventionTrainer  # noqa: E402
from analysis.soft_spectral.training import load_paired_model  # noqa: E402
from scripts.diagnostics.paper1_worker import _split_and_loaders  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.config import load_config  # noqa: E402
from training.engine import Trainer, _one_hot  # noqa: E402


SEEDS = (20260830, 20260831, 20260901)
EPOCHS = (0, 10, 25, 50, 75, 100)


@dataclass(frozen=True)
class Variant:
    method: str
    gate_mode: str


VARIANTS = (
    Variant("dvs_h3_mean_gate", "temporal_mean_normmatched"),
    Variant("dvs_h3_shuffled_gate", "timestep_shuffled"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument(
        "--cache",
        default=Path(tempfile.gettempdir()) / "paper1_dvs_materialized.pt",
        type=Path,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-workers", type=int, default=2)
    parser.add_argument("--training-workers", type=int, default=1)
    return parser.parse_args()


def checkpoint(root: Path, epoch: int) -> Path:
    return root / ("init.pt" if epoch == 0 else f"epoch_{epoch:03d}.pt")


def source_seed_dir(source_run: Path, seed: int) -> Path:
    return source_run / "checkpoints" / "stageD_cross_dataset" / "dvs_dfa" / f"seed_{seed}"


def paired_initial_state(source_run: Path, seed: int) -> Path:
    return (
        source_run / "checkpoints" / "stageD_cross_dataset" / "initial_states"
        / f"seed_{seed}.pt"
    )


def variant_seed_dir(run_dir: Path, variant: Variant, seed: int) -> Path:
    return run_dir / "checkpoints" / "stageB_h3" / variant.method / f"seed_{seed}"


def run_logged(command: list[str], *, log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{tail}")


def diagnostic_command(
    *, seed: int, method: str, state: Path, destination: Path, epoch: int,
    gate_mode: str, device: str, q_reference: Path | None = None,
    target_h3: bool = False,
) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts/diagnostics/paper1_worker.py"),
        "diagnose_snn", "--seed", str(seed), "--method", method,
        "--checkpoint", str(state), "--output-dir", str(destination),
        "--epoch", str(epoch), "--gate-mode", gate_mode, "--device", device,
    ]
    if q_reference is not None:
        command.extend(["--q-reference-checkpoint", str(q_reference)])
    if target_h3:
        command.extend(["--gate-layers", "3"])
    return command


def offline_job(
    run_dir: Path, source_run: Path, seed: int, mode: str,
    *, device: str, env: dict[str, str],
) -> dict:
    source = source_seed_dir(source_run, seed)
    destination = run_dir / "stageA_counterfactual" / mode / f"seed_{seed}"
    if (destination / "complete.json").is_file():
        return {"status": "skipped", "mode": mode, "seed": seed}
    if mode == "mean_gate":
        gate_mode, q_reference = "temporal_mean_unmatched", None
    elif mode == "fixed_reference_q":
        gate_mode, q_reference = "actual", checkpoint(source, 0)
    else:  # pragma: no cover
        raise AssertionError(mode)
    run_logged(
        diagnostic_command(
            seed=seed, method=f"offline_{mode}", state=checkpoint(source, 100),
            destination=destination, epoch=100, gate_mode=gate_mode,
            device=device, q_reference=q_reference,
        ),
        log=run_dir / "logs" / f"offline_{mode}_seed_{seed}.log", env=env,
    )
    return {"status": "complete", "mode": mode, "seed": seed}


def actual_final_job(
    run_dir: Path, source_run: Path, seed: int, *, device: str, env: dict[str, str]
) -> dict:
    destination = (
        run_dir / "stageB_h3" / "diagnostics" / "dvs_actual"
        / f"seed_{seed}" / "epoch_100"
    )
    if (destination / "complete.json").is_file():
        return {"status": "skipped", "method": "dvs_actual", "seed": seed, "epoch": 100}
    run_logged(
        diagnostic_command(
            seed=seed, method="dvs_actual",
            state=checkpoint(source_seed_dir(source_run, seed), 100),
            destination=destination, epoch=100, gate_mode="actual", device=device,
        ),
        log=run_dir / "logs" / f"diag_dvs_actual_seed_{seed}_epoch_100.log", env=env,
    )
    return {"status": "complete", "method": "dvs_actual", "seed": seed, "epoch": 100}


def _model_gradients(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def baseline_parity(
    run_dir: Path, source_run: Path, *, device: torch.device, env: dict[str, str]
) -> dict:
    output = run_dir / "baseline_parity.json"
    if output.is_file():
        audit = json.loads(output.read_text(encoding="utf-8"))
        if audit.get("passed"):
            return audit
    seed = SEEDS[0]
    initial_path = paired_initial_state(source_run, seed)
    state = torch.load(
        checkpoint(source_seed_dir(source_run, seed), 0),
        map_location="cpu", weights_only=False,
    )
    config = state["config"]
    _split, loaders = _split_and_loaders(config, seed=seed, materialize=True)
    batch = next(iter(loaders["train"]))
    raw_samples, raw_target, raw_indices = batch
    samples = raw_samples.transpose(0, 1).contiguous().to(device, dtype=torch.float32)
    target = raw_target.to(device)
    indices = raw_indices.to(device)

    production_model, production_bank, _ = load_paired_model(
        config=config, initial_state_path=initial_path, device=device
    )
    off_model, off_bank, _ = load_paired_model(
        config=config, initial_state_path=initial_path, device=device
    )
    production = Trainer(production_model, production_bank, config, device)
    off = GateInterventionTrainer(
        off_model, off_bank, config, device, gate_mode="off", seed=seed,
        intervention_layers={2},
    )
    off._sample_indices = indices
    with torch.no_grad():
        first_output = production_model(samples, detach_temporal=True)
        second_output = off_model(samples, detach_temporal=True)
    forward_max_abs = float((first_output - second_output).abs().max())
    production.optimizer.zero_grad(set_to_none=True)
    production_values = production._local_batch(samples, target)
    production_values["total"].backward()
    off.optimizer.zero_grad(set_to_none=True)
    off_values = off._local_batch(samples, target)
    off_values["total"].backward()
    production_gradients = _model_gradients(production_model)
    off_gradients = _model_gradients(off_model)
    gradient_rows = []
    for name in sorted(production_gradients):
        first, second = production_gradients[name], off_gradients[name]
        gradient_rows.append(
            {
                "parameter": name,
                "max_abs": float((first - second).abs().max()),
                "relative": float(
                    torch.linalg.vector_norm(first - second)
                    / (torch.linalg.vector_norm(first) + 1e-30)
                ),
            }
        )
    with torch.no_grad():
        output_values, hidden_inputs, _ = production_model.forward_with_cache(
            samples, detach_temporal=True
        )
        error = output_values.mean(dim=0) - _one_hot(target, production_model.num_classes)
    delta_rows = []
    for layer_index, layer_input in enumerate(hidden_inputs):
        capture = _capture_snn_dfa_layer(
            production_model, production_bank, layer_index, layer_input, error,
            gate_mode="actual", seed=seed, epoch=0, sample_indices=indices,
        )
        delta_rows.append(
            {
                "layer": f"hidden_{layer_index + 1}",
                "production_q_gate_relative": capture["parity_relative"],
            }
        )
    max_gradient_relative = max(row["relative"] for row in gradient_rows)
    max_gradient_abs = max(row["max_abs"] for row in gradient_rows)
    max_delta_relative = max(row["production_q_gate_relative"] for row in delta_rows)
    audit = {
        "passed": (
            forward_max_abs == 0.0
            and max_gradient_abs <= 1e-12
            and max_gradient_relative <= 1e-6
            and max_delta_relative < 1e-5
        ),
        "scope": "real DVS-Gesture batch; intervention OFF dispatches to the unchanged production Trainer path",
        "seed": seed,
        "batch_samples": int(target.numel()),
        "forward_max_abs": forward_max_abs,
        "total_loss_max_abs": abs(float(production_values["total"].detach()) - float(off_values["total"].detach())),
        "max_weight_gradient_relative": max_gradient_relative,
        "max_weight_gradient_abs": max_gradient_abs,
        "max_production_delta_q_gate_relative": max_delta_relative,
        "tolerances": {
            "forward_max_abs": 0.0,
            "weight_gradient_max_abs": 1e-12,
            "weight_gradient_relative": 1e-6,
            "production_delta_relative": 1e-5,
        },
        "weight_gradients": gradient_rows,
        "delta_parity": delta_rows,
    }
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["passed"]:
        raise AssertionError(f"DVS baseline parity failed: {audit}")
    return audit


def train_job(
    run_dir: Path, source_run: Path, variant: Variant, seed: int,
    *, device: str, env: dict[str, str],
) -> dict:
    destination = variant_seed_dir(run_dir, variant, seed)
    summary = destination / "variant_summary.json"
    if summary.is_file():
        return {"status": "skipped", "method": variant.method, "seed": seed}
    command = [
        sys.executable, str(ROOT / "scripts/diagnostics/paper1_worker.py"),
        "train_gate", "--config", "configs/dvs_gesture/paper1_experiment03.yaml",
        "--seed", str(seed), "--method", variant.method, "--epochs", "100",
        "--initial-state", str(paired_initial_state(source_run, seed)),
        "--output-dir", str(destination), "--device", device,
        "--gate-mode", variant.gate_mode, "--gate-layers", "3",
    ]
    run_logged(
        command, log=run_dir / "logs" / f"train_{variant.method}_seed_{seed}.log", env=env
    )
    return {"status": "complete", "method": variant.method, "seed": seed}


def diagnostic_job(
    run_dir: Path, variant: Variant, seed: int, epoch: int,
    *, device: str, env: dict[str, str],
) -> dict:
    destination = (
        run_dir / "stageB_h3" / "diagnostics" / variant.method
        / f"seed_{seed}" / f"epoch_{epoch:03d}"
    )
    if (destination / "complete.json").is_file():
        return {"status": "skipped", "method": variant.method, "seed": seed, "epoch": epoch}
    run_logged(
        diagnostic_command(
            seed=seed, method=variant.method,
            state=checkpoint(variant_seed_dir(run_dir, variant, seed), epoch),
            destination=destination, epoch=epoch, gate_mode=variant.gate_mode,
            device=device, target_h3=True,
        ),
        log=run_dir / "logs" / f"diag_{variant.method}_seed_{seed}_epoch_{epoch:03d}.log",
        env=env,
    )
    return {"status": "complete", "method": variant.method, "seed": seed, "epoch": epoch}


def parallel(jobs, worker, *, max_workers: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, *job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


def main() -> None:
    args = parse_args()
    run_dir, source_run = args.run_dir.resolve(), args.source_run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    env = os.environ.copy()
    env["PAPER1_MATERIALIZED_CACHE"] = str(args.cache)

    offline_jobs = [(run_dir, source_run, seed, mode) for seed in SEEDS for mode in ("mean_gate", "fixed_reference_q")]
    parallel(
        offline_jobs,
        lambda run, source, seed, mode: offline_job(
            run, source, seed, mode, device=args.device, env=env
        ),
        max_workers=max(1, args.diagnostic_workers),
    )
    parallel(
        [(run_dir, source_run, seed) for seed in SEEDS],
        lambda run, source, seed: actual_final_job(
            run, source, seed, device=args.device, env=env
        ),
        max_workers=max(1, args.diagnostic_workers),
    )
    audit = baseline_parity(run_dir, source_run, device=torch.device(args.device), env=env)
    print(json.dumps({"status": "baseline_parity", "passed": audit["passed"]}), flush=True)

    for variant in VARIANTS:
        jobs = [(run_dir, source_run, variant, seed) for seed in SEEDS]
        parallel(
            jobs,
            lambda run, source, selected, seed: train_job(
                run, source, selected, seed, device=args.device, env=env
            ),
            max_workers=max(1, args.training_workers),
        )

    jobs = [(run_dir, variant, seed, epoch) for variant in VARIANTS for seed in SEEDS for epoch in EPOCHS]
    parallel(
        jobs,
        lambda run, selected, seed, epoch: diagnostic_job(
            run, selected, seed, epoch, device=args.device, env=env
        ),
        max_workers=max(1, args.diagnostic_workers),
    )
    done = {
        "status": "complete", "offline_diagnostics": len(offline_jobs),
        "training_variants": len(VARIANTS) * len(SEEDS),
        "stageB_diagnostics": len(jobs), "baseline_parity": audit,
    }
    (run_dir / "stageB_execution_summary.json").write_text(
        json.dumps(done, indent=2), encoding="utf-8"
    )
    print(json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
