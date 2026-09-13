"""BPTT and replay-based DFA/sDFA/LoDFA training engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from evaluation import classification_accuracy
from methods import FeedbackBank
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import resolve_repo_path


def _one_hot(target: Tensor, classes: int) -> Tensor:
    return F.one_hot(target, num_classes=classes).to(dtype=torch.float32)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        feedback_bank: FeedbackBank,
        config: dict[str, Any],
        device: torch.device,
        spectral_filter=None,
    ) -> None:
        self.model = model.to(device)
        self.feedback_bank = feedback_bank.to(device)
        self.config = config
        self.device = device
        self.spectral_filter = spectral_filter
        self.last_epoch_spectral_rows: list[dict[str, Any]] = []
        parameters = list(model.parameters()) + list(feedback_bank.parameters())
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
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
        self.start_epoch = 0

    def _prepare(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        samples, target = batch
        # DataLoader is batch-major; all model and method equations are time-major.
        samples = samples.transpose(0, 1).contiguous().to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        return samples, target.to(self.device, non_blocking=True)

    def _task_loss(self, output_spikes: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        mean_output = output_spikes.mean(dim=0)
        desired = _one_hot(target, self.model.num_classes).to(mean_output.device)
        return 0.5 * (mean_output - desired).square().sum(dim=1).mean(), mean_output

    def _bptt_batch(self, samples: Tensor, target: Tensor) -> dict[str, Tensor | float]:
        output = self.model(samples, detach_temporal=False)
        task_loss, mean_output = self._task_loss(output, target)
        return {
            "total": task_loss,
            "task": task_loss.detach(),
            "local": 0.0,
            "structure": 0.0,
            "mean_output": mean_output.detach(),
            "alignment": 0.0,
            "orthogonal": 0.0,
            "hoyer": 0.0,
        }

    def _local_batch(self, samples: Tensor, target: Tensor) -> dict[str, Tensor | float]:
        temporal_mode = self.config["method"].get("temporal_mode", "pointwise")
        pointwise = temporal_mode == "pointwise"
        with torch.no_grad():
            forward_output, hidden_inputs, readout_input = self.model.forward_with_cache(
                samples, detach_temporal=pointwise
            )
            provisional_mean = forward_output.mean(dim=0)
            desired = _one_hot(target, self.model.num_classes).to(self.device)
            output_error = provisional_mean - desired

        # Replay the output layer for the global task MSE.
        output_replay = self.model.replay_readout(readout_input.detach())
        task_loss, mean_output = self._task_loss(output_replay, target)

        local_losses: list[Tensor] = []
        for index, (spec, layer_input) in enumerate(
            zip(self.model.hidden_specs, hidden_inputs)
        ):
            if self.spectral_filter is None:
                local_spikes = self.model.replay_hidden(
                    index, layer_input.detach(), detach_temporal=pointwise
                )
            else:
                if spec["kind"] != "fc":
                    raise ValueError(
                        "post-gate spectral regulation currently supports FC hidden layers"
                    )
                linear = self.model.hidden_layers[index].linear
                with self.spectral_filter.intercept_linear(index, linear):
                    local_spikes = self.model.replay_hidden(
                        index, layer_input.detach(), detach_temporal=pointwise
                    )
            # Patent stop-gradient: the proxy loss updates the forward layer, not B.
            teaching = self.feedback_bank.project(index, output_error).detach()
            if spec["kind"] == "conv":
                height, width = local_spikes.shape[-2:]
                teaching = teaching[:, :, None, None] / float(height * width)
            local_losses.append((local_spikes * teaching.unsqueeze(0)).mean())
        local_loss = torch.stack(local_losses).sum()
        structure_loss, structure_parts = self.feedback_bank.regularization(self.model)
        local_weight = float(self.config["method"].get("local_weight", 1.0))
        total = task_loss + local_weight * local_loss + structure_loss
        return {
            "total": total,
            "task": task_loss.detach(),
            "local": local_loss.detach(),
            "structure": structure_loss.detach(),
            "mean_output": mean_output.detach(),
            **structure_parts,
        }

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        self.feedback_bank.train()
        sums = {
            "loss": 0.0,
            "task_loss": 0.0,
            "local_loss": 0.0,
            "structure_loss": 0.0,
            "accuracy": 0.0,
            "samples": 0,
        }
        self.last_epoch_spectral_rows = []
        for batch_index, batch in enumerate(loader):
            samples, target = self._prepare(batch)
            self.optimizer.zero_grad(set_to_none=True)
            if self.feedback_bank.method_name == "bptt":
                values = self._bptt_batch(samples, target)
            else:
                values = self._local_batch(samples, target)
            total = values["total"]
            assert isinstance(total, Tensor)
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite training loss")
            total.backward()
            if self.spectral_filter is not None:
                for row in self.spectral_filter.drain_batch_diagnostics():
                    self.last_epoch_spectral_rows.append(
                        {"batch_index": batch_index, **row}
                    )
            clip = float(self.config["training"].get("gradient_clip", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.feedback_bank.parameters()),
                    clip,
                )
            self.optimizer.step()
            count = target.numel()
            sums["samples"] += count
            sums["loss"] += float(total.detach()) * count
            sums["task_loss"] += float(values["task"]) * count
            sums["local_loss"] += float(values["local"]) * count
            sums["structure_loss"] += float(values["structure"]) * count
            sums["accuracy"] += (
                float(classification_accuracy(values["mean_output"], target)) * count
            )
        samples = max(1, sums.pop("samples"))
        return {key: value / samples for key, value in sums.items()}

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        self.feedback_bank.eval()
        total_loss = total_accuracy = 0.0
        sample_count = 0
        for batch in loader:
            samples, target = self._prepare(batch)
            output = self.model(samples, detach_temporal=False)
            loss, mean_output = self._task_loss(output, target)
            count = target.numel()
            sample_count += count
            total_loss += float(loss) * count
            total_accuracy += float(classification_accuracy(mean_output, target)) * count
        denominator = max(1, sample_count)
        return {"loss": total_loss / denominator, "accuracy": total_accuracy / denominator}

    def resume(self, checkpoint_path: str | Path) -> dict[str, Any]:
        state = load_checkpoint(
            checkpoint_path,
            model=self.model,
            feedback_bank=self.feedback_bank,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            restore_rng=True,
            map_location=self.device,
        )
        self.start_epoch = int(state["epoch"]) + 1
        return state

    def fit(self, train_loader, test_loader) -> dict[str, Any]:
        output_dir = resolve_repo_path(self.config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {key: value for key, value in self.config.items() if not key.startswith("_")},
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        history_path = output_dir / "metrics.jsonl"
        epochs = int(self.config["training"]["epochs"])
        best_accuracy = -1.0
        best_record: dict[str, Any] = {}
        if self.start_epoch > 0 and history_path.is_file():
            with history_path.open("r", encoding="utf-8") as handle:
                previous = [json.loads(line) for line in handle if line.strip()]
            if previous:
                best_record = max(previous, key=lambda item: item["test"]["accuracy"])
                best_accuracy = float(best_record["test"]["accuracy"])
        checkpoint_epochs = {
            int(value)
            for value in self.config["training"].get("checkpoint_epochs", [])
        }
        for epoch in range(self.start_epoch, epochs):
            active_ranks = self.feedback_bank.update_dynamic_ranks(epoch, epochs)
            train_metrics = self.train_epoch(train_loader)
            test_metrics = self.evaluate(test_loader)
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "test": test_metrics,
                "active_ranks": active_ranks,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Persist the state expected at the start of the next epoch.
            if self.scheduler is not None:
                self.scheduler.step()
            if test_metrics["accuracy"] >= best_accuracy:
                best_accuracy = test_metrics["accuracy"]
                best_record = record
                save_checkpoint(
                    output_dir / "best.pt",
                    epoch=epoch,
                    model=self.model,
                    feedback_bank=self.feedback_bank,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config,
                    metrics=record,
                )
            completed_epoch = epoch + 1
            if completed_epoch in checkpoint_epochs:
                save_checkpoint(
                    output_dir / f"epoch_{completed_epoch:03d}.pt",
                    epoch=epoch,
                    model=self.model,
                    feedback_bank=self.feedback_bank,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config,
                    metrics=record,
                )
            save_checkpoint(
                output_dir / "last.pt",
                epoch=epoch,
                model=self.model,
                feedback_bank=self.feedback_bank,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                config=self.config,
                metrics=record,
            )
            print(json.dumps(record, ensure_ascii=False))
        summary = {
            "status": "complete",
            "experiment": self.config["experiment"]["name"],
            "dataset": self.config["data"]["dataset"],
            "model": self.config["model"]["architecture"],
            "neuron": self.config["neuron"].get("type", "lif"),
            "time_steps": self.config["data"]["time_steps"],
            "method": self.feedback_bank.method_name,
            "rank": self.feedback_bank.active_ranks,
            "energy_threshold": self.config["method"]
            .get("feedback", {})
            .get("dynamic_rank", {})
            .get("energy_threshold"),
            "loss_weights": self.config["method"]
            .get("feedback", {})
            .get("loss_weights", {}),
            "seed": self.config["experiment"]["seed"],
            "best_accuracy": best_accuracy,
            "best_epoch": best_record.get("epoch"),
            "config": "config.yaml",
            "checkpoint": "best.pt",
            "log": "metrics.jsonl",
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return summary
