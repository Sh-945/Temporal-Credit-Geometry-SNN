"""Training paths for the temporal ANN control and causal gate variants."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Collection

import torch
import torch.nn.functional as F
from torch import Tensor

from evaluation import classification_accuracy
from methods import FeedbackBank
from methods.gate_intervention import (
    GateMode,
    apply_gate_intervention,
    deterministic_time_permutations,
)
from models.temporal_ann_control import TemporalANN
from training.checkpoint import load_checkpoint, save_checkpoint
from training.engine import Trainer


DIAGNOSTIC_EPOCHS = {10, 25, 50, 75, 100}


def _one_hot(target: Tensor, classes: int) -> Tensor:
    return F.one_hot(target, num_classes=classes).to(torch.float32)


class GateInterventionTrainer(Trainer):
    """Production-equivalent sDFA replay with an optional gradient-only gate."""

    def __init__(
        self,
        model,
        feedback_bank,
        config: dict[str, Any],
        device: torch.device,
        *,
        gate_mode: GateMode | str,
        seed: int,
        intervention_layers: Collection[int] | None = None,
    ) -> None:
        super().__init__(model, feedback_bank, config, device)
        self.gate_mode = GateMode(gate_mode)
        self.seed = int(seed)
        self.intervention_layers = (
            None
            if intervention_layers is None
            else frozenset(int(index) for index in intervention_layers)
        )
        self.completed_epoch = 0
        self._sample_indices: Tensor | None = None
        self.last_gate_rows: list[dict[str, Any]] = []

    def set_epoch(self, completed_epoch: int) -> None:
        self.completed_epoch = int(completed_epoch)

    def _prepare(self, batch) -> tuple[Tensor, Tensor]:
        if len(batch) != 3:
            raise ValueError("gate intervention loader must return sample, label, index")
        samples, target, indices = batch
        self._sample_indices = indices.to(self.device, non_blocking=True)
        samples = samples.transpose(0, 1).contiguous().to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        return samples, target.to(self.device, non_blocking=True)

    def _local_batch(self, samples: Tensor, target: Tensor) -> dict[str, Tensor | float]:
        if self.gate_mode == GateMode.OFF:
            return super()._local_batch(samples, target)
        if self._sample_indices is None:
            raise RuntimeError("sample indices were not prepared")
        temporal_mode = self.config["method"].get("temporal_mode", "pointwise")
        pointwise = temporal_mode == "pointwise"
        with torch.no_grad():
            forward_output, hidden_inputs, readout_input = self.model.forward_with_cache(
                samples, detach_temporal=pointwise
            )
            desired = _one_hot(target, self.model.num_classes).to(self.device)
            output_error = forward_output.mean(dim=0) - desired

        output_replay = self.model.replay_readout(readout_input.detach())
        task_loss, mean_output = self._task_loss(output_replay, target)
        local_losses: list[Tensor] = []
        self.last_gate_rows = []
        time_permutations = (
            deterministic_time_permutations(
                samples.shape[0],
                seed=self.seed,
                epoch=self.completed_epoch,
                sample_indices=self._sample_indices,
            )
            if self.gate_mode == GateMode.TIMESTEP_SHUFFLED
            else None
        )
        for index, layer_input in enumerate(hidden_inputs):
            effective_gate_mode = (
                self.gate_mode
                if self.intervention_layers is None or index in self.intervention_layers
                else GateMode.ACTUAL
            )
            kind = self.model.hidden_specs[index]["kind"]
            if kind == "conv":
                module = self.model.conv_layers[index].conv
            elif getattr(self.model, "architecture", None) == "conv":
                module = self.model.fc_hidden.linear
            else:
                module = self.model.hidden_layers[index].linear
            captured: list[Tensor] = []
            handle = module.register_forward_hook(
                lambda _module, _inputs, output, destination=captured: destination.append(output)
            )
            try:
                spikes = self.model.replay_hidden(
                    index, layer_input.detach(), detach_temporal=pointwise
                )
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError("expected one hidden current during intervention replay")
            current = captured[0]
            current_view = current.reshape_as(spikes)
            actual_gate = torch.autograd.grad(
                spikes,
                current,
                grad_outputs=torch.ones_like(spikes),
                retain_graph=True,
                create_graph=False,
            )[0].reshape_as(spikes).detach()
            applied_gate = apply_gate_intervention(
                actual_gate,
                effective_gate_mode,
                seed=self.seed,
                epoch=self.completed_epoch,
                sample_indices=self._sample_indices,
                time_permutations=time_permutations,
            )
            teaching = self.feedback_bank.project(index, output_error).detach()
            if kind == "conv":
                height, width = spikes.shape[-2:]
                production_teaching = teaching[:, :, None, None] / float(height * width)
            else:
                production_teaching = teaching
            desired_delta = production_teaching.unsqueeze(0) * applied_gate
            injected_proxy = (current_view * desired_delta).mean()
            production_proxy = (spikes * production_teaching.unsqueeze(0)).mean()
            # Preserve the production scalar value while replacing only dL/dI.
            local_losses.append(
                injected_proxy + (production_proxy - injected_proxy).detach()
            )
            self.last_gate_rows.append(
                {
                    "layer": f"hidden_{index + 1}",
                    "gate_mode": effective_gate_mode.value,
                    "actual_norm": float(torch.linalg.vector_norm(actual_gate)),
                    "applied_norm": float(torch.linalg.vector_norm(applied_gate)),
                    "norm_ratio": float(
                        torch.linalg.vector_norm(applied_gate)
                        / (torch.linalg.vector_norm(actual_gate) + 1e-30)
                    ),
                }
            )
        local_loss = torch.stack(local_losses).sum()
        structure_loss, structure_parts = self.feedback_bank.regularization(self.model)
        total = (
            task_loss
            + float(self.config["method"].get("local_weight", 1.0)) * local_loss
            + structure_loss
        )
        return {
            "total": total,
            "task": task_loss.detach(),
            "local": local_loss.detach(),
            "structure": structure_loss.detach(),
            "mean_output": mean_output.detach(),
            **structure_parts,
        }


class IndexedTrainer(Trainer):
    """Unchanged production Trainer accepting source indices from a loader."""

    def _prepare(self, batch) -> tuple[Tensor, Tensor]:
        samples, target = batch[:2]
        samples = samples.transpose(0, 1).contiguous().to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        return samples, target.to(self.device, non_blocking=True)


class TemporalANNTrainer:
    def __init__(
        self,
        model: TemporalANN,
        feedback_bank: FeedbackBank,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.feedback_bank = feedback_bank.to(device)
        self.config = config
        self.device = device
        parameters = [
            value
            for value in list(model.parameters()) + list(feedback_bank.parameters())
            if value.requires_grad
        ]
        training = config["training"]
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        step_size = int(training.get("lr_step", 0))
        self.scheduler = (
            torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=float(training.get("lr_gamma", 0.1)),
            )
            if step_size > 0
            else None
        )

    def _prepare(self, batch) -> tuple[Tensor, Tensor]:
        samples, target = batch[:2]
        return (
            samples.transpose(0, 1).contiguous().to(
                self.device, dtype=torch.float32, non_blocking=True
            ),
            target.to(self.device, non_blocking=True),
        )

    def _task_loss(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        mean_output = logits.mean(dim=0)
        desired = _one_hot(target, self.model.num_classes).to(mean_output.device)
        loss = 0.5 * (mean_output - desired).square().sum(dim=1).mean()
        return loss, mean_output

    def _batch(self, samples: Tensor, target: Tensor) -> dict[str, Tensor | float]:
        with torch.no_grad():
            provisional, hidden_inputs, _currents, readout_input = self.model.forward_with_cache(samples)
            desired = _one_hot(target, self.model.num_classes).to(self.device)
            output_error = provisional.mean(dim=0) - desired
        output_replay = self.model.replay_readout(readout_input.detach())
        task_loss, mean_output = self._task_loss(output_replay, target)
        local_losses: list[Tensor] = []
        for index, layer_input in enumerate(hidden_inputs):
            activation, _current = self.model.replay_hidden(index, layer_input.detach())
            teaching = self.feedback_bank.project(index, output_error).detach()
            local_losses.append((activation * teaching.unsqueeze(0)).mean())
        local_loss = torch.stack(local_losses).sum()
        return {
            "total": task_loss + float(self.config["method"].get("local_weight", 1.0)) * local_loss,
            "task": task_loss.detach(),
            "local": local_loss.detach(),
            "structure": 0.0,
            "mean_output": mean_output.detach(),
        }

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        self.feedback_bank.train()
        sums = {"loss": 0.0, "task_loss": 0.0, "local_loss": 0.0, "accuracy": 0.0, "samples": 0}
        for batch in loader:
            samples, target = self._prepare(batch)
            self.optimizer.zero_grad(set_to_none=True)
            values = self._batch(samples, target)
            total = values["total"]
            assert isinstance(total, Tensor)
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite temporal ANN loss")
            total.backward()
            clip = float(self.config["training"].get("gradient_clip", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.feedback_bank.parameters()), clip
                )
            self.optimizer.step()
            count = int(target.numel())
            sums["samples"] += count
            sums["loss"] += float(total.detach()) * count
            sums["task_loss"] += float(values["task"]) * count
            sums["local_loss"] += float(values["local"]) * count
            sums["accuracy"] += float(classification_accuracy(values["mean_output"], target)) * count
        count = max(1, sums.pop("samples"))
        return {key: value / count for key, value in sums.items()}

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        self.feedback_bank.eval()
        loss_sum = accuracy_sum = 0.0
        sample_count = 0
        for batch in loader:
            samples, target = self._prepare(batch)
            logits = self.model(samples)
            loss, mean_output = self._task_loss(logits, target)
            count = int(target.numel())
            sample_count += count
            loss_sum += float(loss) * count
            accuracy_sum += float(classification_accuracy(mean_output, target)) * count
        count = max(1, sample_count)
        return {"loss": loss_sum / count, "accuracy": accuracy_sum / count}


def build_matched_ann(
    config: dict[str, Any], initial_state: dict[str, Any], device: torch.device
) -> tuple[TemporalANN, FeedbackBank]:
    model = TemporalANN(
        list(config["model"]["input_shape"]),
        list(config["model"]["hidden_features"]),
        int(config["model"]["num_classes"]),
    )
    model.load_matched_snn_state(initial_state["model_state"])
    bank = FeedbackBank(model, config)
    bank.load_state_dict(initial_state["feedback_state"])
    return model.to(device), bank.to(device)


def train_with_validation(
    *,
    trainer,
    loaders,
    config: dict[str, Any],
    output_dir: Path,
    method: str,
    seed: int,
    epochs: int = 100,
) -> dict[str, Any]:
    """Train without test-based selection and save all diagnostic checkpoints."""

    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "init.pt",
        epoch=-1,
        model=trainer.model,
        feedback_bank=trainer.feedback_bank,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        config=config,
        metrics={"stage": "paired_initialization", "method": method, "seed": seed},
    )
    rows: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    start = time.perf_counter()
    for completed_epoch in range(1, int(epochs) + 1):
        if hasattr(trainer, "set_epoch"):
            trainer.set_epoch(completed_epoch)
        train = trainer.train_epoch(loaders["train"])
        validation = trainer.evaluate(loaders["validation"])
        row = {
            "method": method,
            "seed": seed,
            "epoch": completed_epoch,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        rows.append(row)
        improved = validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] <= best_loss
        )
        if improved:
            best_accuracy, best_loss, best_epoch = validation["accuracy"], validation["loss"], completed_epoch
            save_checkpoint(
                output_dir / "best_validation.pt",
                epoch=completed_epoch - 1,
                model=trainer.model,
                feedback_bank=trainer.feedback_bank,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                config=config,
                metrics=row,
            )
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        save_checkpoint(
            output_dir / "last.pt",
            epoch=completed_epoch - 1,
            model=trainer.model,
            feedback_bank=trainer.feedback_bank,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            config=config,
            metrics=row,
        )
        if completed_epoch in DIAGNOSTIC_EPOCHS:
            save_checkpoint(
                output_dir / f"epoch_{completed_epoch:03d}.pt",
                epoch=completed_epoch - 1,
                model=trainer.model,
                feedback_bank=trainer.feedback_bank,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                config=config,
                metrics=row,
            )
    final_test = trainer.evaluate(loaders["test"])
    load_checkpoint(
        output_dir / "best_validation.pt",
        model=trainer.model,
        feedback_bank=trainer.feedback_bank,
        map_location=trainer.device,
    )
    best_test = trainer.evaluate(loaders["test"])
    validation_values = [row["validation_accuracy"] for row in rows]
    threshold = 0.99 * max(validation_values)
    convergence_epoch = next(row["epoch"] for row in rows if row["validation_accuracy"] >= threshold)
    summary = {
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "best_validation_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
        "best_validation_test": best_test,
        "final_test": final_test,
        "convergence_epoch_99pct_own_best": convergence_epoch,
        "wall_seconds": time.perf_counter() - start,
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (output_dir / "variant_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return {"training_metrics": rows, "summary": summary}
