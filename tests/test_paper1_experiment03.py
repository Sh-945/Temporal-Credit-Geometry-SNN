from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from analysis.feedback_expansion.capture import capture_layer_signals
from analysis.paper1_geometry.data import IndexedSubset, build_paper1_split
from analysis.paper1_geometry.diagnostics import diagnose_ann_checkpoint, diagnose_snn_checkpoint
from analysis.paper1_geometry.training import GateInterventionTrainer, build_matched_ann
from methods import FeedbackBank
from methods.gate_intervention import (
    GateMode,
    apply_gate_intervention,
    deterministic_time_permutation,
)
from models import build_model
from models.temporal_ann_control import TemporalANN
from training.config import load_config
from training.engine import Trainer, _one_hot
from training.seed import seed_everything


ROOT = Path(__file__).resolve().parents[1]


class Paper1Experiment03Tests(unittest.TestCase):
    def _config(self):
        config = load_config(ROOT / "configs/nmnist/sdfa.yaml")
        config["data"].update({"dataset": "synthetic", "time_steps": 4, "batch_size": 3})
        config["model"]["input_shape"] = [2, 4, 4]
        config["model"]["hidden_features"] = [9, 8, 7]
        config["training"]["gradient_clip"] = 0.0
        return config

    def test_probe_order_is_deterministic_and_not_sorted_before_fit_eval_split(self):
        first = build_paper1_split(2000, probe_size=1024)
        second = build_paper1_split(2000, probe_size=1024)
        self.assertTrue((first.probe_indices == second.probe_indices).all())
        self.assertFalse((first.probe_indices == sorted(first.probe_indices)).all())
        self.assertFalse(set(first.basis_fit_indices) & set(first.basis_eval_indices))

    def test_temporal_ann_uses_one_shared_parameter_set(self):
        model = TemporalANN([2, 4, 4], [9, 8, 7], 10)
        self.assertEqual(len(model.hidden_layers), 3)
        samples = torch.rand(5, 3, 2, 4, 4)
        output = model(samples)
        self.assertEqual(tuple(output.shape), (5, 3, 10))
        output.sum().backward()
        self.assertIsNotNone(model.hidden_layers[0].weight.grad)
        self.assertEqual(len(list(model.parameters())), 8)

    def test_mean_gate_is_time_constant_and_norm_matched(self):
        gate = torch.rand(6, 4, 9)
        indices = torch.arange(4)
        applied = apply_gate_intervention(
            gate,
            GateMode.TEMPORAL_MEAN,
            seed=5,
            epoch=2,
            sample_indices=indices,
        )
        self.assertTrue(torch.allclose(applied, applied[:1].expand_as(applied)))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(applied),
                torch.linalg.vector_norm(gate),
                rtol=1e-6,
                atol=1e-7,
            )
        )

    def test_shuffled_gate_is_reproducible_and_preserves_values(self):
        gate = torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(5, 3, 4)
        indices = torch.tensor([19, 21, 44])
        first = apply_gate_intervention(
            gate,
            GateMode.TIMESTEP_SHUFFLED,
            seed=7,
            epoch=8,
            sample_indices=indices,
        )
        second = apply_gate_intervention(
            gate,
            GateMode.TIMESTEP_SHUFFLED,
            seed=7,
            epoch=8,
            sample_indices=indices,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first.sort(dim=0).values, gate.sort(dim=0).values))
        self.assertEqual(
            deterministic_time_permutation(5, seed=7, epoch=8, sample_index=19).tolist(),
            deterministic_time_permutation(5, seed=7, epoch=8, sample_index=19).tolist(),
        )

    def test_actual_gate_path_matches_production_delta_and_weight_gradient(self):
        config = self._config()
        seed_everything(11)
        production_model = build_model(config)
        production_bank = FeedbackBank(production_model, config)
        intervention_model = copy.deepcopy(production_model)
        intervention_bank = copy.deepcopy(production_bank)
        device = torch.device("cpu")
        production = Trainer(production_model, production_bank, config, device)
        intervention = GateInterventionTrainer(
            intervention_model,
            intervention_bank,
            config,
            device,
            gate_mode=GateMode.ACTUAL,
            seed=11,
        )
        samples = torch.rand(4, 3, 2, 4, 4)
        target = torch.tensor([1, 2, 3])
        intervention._sample_indices = torch.tensor([5, 6, 7])
        with torch.no_grad():
            first = production_model(samples, detach_temporal=True)
            second = intervention_model(samples, detach_temporal=True)
        self.assertTrue(torch.equal(first, second))

        production.optimizer.zero_grad(set_to_none=True)
        production_values = production._local_batch(samples, target)
        production_values["total"].backward()
        intervention.optimizer.zero_grad(set_to_none=True)
        intervention_values = intervention._local_batch(samples, target)
        intervention_values["total"].backward()
        for first_layer, second_layer in zip(
            production_model.hidden_layers, intervention_model.hidden_layers
        ):
            self.assertTrue(
                torch.allclose(
                    first_layer.linear.weight.grad,
                    second_layer.linear.weight.grad,
                    rtol=1e-5,
                    atol=1e-7,
                )
            )

        off_model = copy.deepcopy(production_model)
        off_bank = copy.deepcopy(production_bank)
        off = GateInterventionTrainer(
            off_model,
            off_bank,
            config,
            device,
            gate_mode=GateMode.OFF,
            seed=11,
        )
        reference_model = copy.deepcopy(production_model)
        reference_bank = copy.deepcopy(production_bank)
        reference = Trainer(reference_model, reference_bank, config, device)
        off._sample_indices = torch.tensor([5, 6, 7])
        off.optimizer.zero_grad(set_to_none=True)
        off._local_batch(samples, target)["total"].backward()
        reference.optimizer.zero_grad(set_to_none=True)
        reference._local_batch(samples, target)["total"].backward()
        for first_layer, second_layer in zip(off_model.hidden_layers, reference_model.hidden_layers):
            self.assertTrue(torch.equal(first_layer.linear.weight.grad, second_layer.linear.weight.grad))

        with torch.no_grad():
            output, hidden_inputs, _ = production_model.forward_with_cache(
                samples, detach_temporal=True
            )
            error = output.mean(0) - _one_hot(target, production_model.num_classes)
        capture = capture_layer_signals(
            production_model,
            production_bank,
            0,
            hidden_inputs[0],
            error,
            detach_temporal=True,
        )
        applied = apply_gate_intervention(
            capture.gate,
            GateMode.ACTUAL,
            seed=11,
            epoch=0,
            sample_indices=torch.tensor([5, 6, 7]),
        )
        self.assertTrue(torch.allclose(capture.delta, capture.q.unsqueeze(0) * applied))

    def test_all_three_paradigms_emit_matched_current_geometry(self):
        config = self._config()
        seed_everything(13)
        model = build_model(config)
        bank = FeedbackBank(model, config)
        frames = torch.rand(16, 4, 2, 4, 4)
        labels = torch.arange(16) % 10
        dataset = TensorDataset(frames, labels)
        loaders = {
            "basis_fit": DataLoader(IndexedSubset(dataset, list(range(8))), batch_size=4),
            "basis_eval": DataLoader(IndexedSubset(dataset, list(range(8, 16))), batch_size=4),
        }
        snn = diagnose_snn_checkpoint(
            model=model,
            feedback_bank=bank,
            loaders=loaders,
            device=torch.device("cpu"),
            dataset="synthetic",
            method="dfa_actual",
            seed=13,
            epoch=0,
            is_bptt=False,
        )
        self.assertTrue(snn["geometry"])
        self.assertEqual({row["ambient_dim"] for row in snn["geometry"]}, {7, 8, 9})
        initial = {
            "model_state": copy.deepcopy(model.state_dict()),
            "feedback_state": copy.deepcopy(bank.state_dict()),
        }
        ann, ann_bank = build_matched_ann(config, initial, torch.device("cpu"))
        ann_result = diagnose_ann_checkpoint(
            model=ann,
            feedback_bank=ann_bank,
            loaders=loaders,
            device=torch.device("cpu"),
            dataset="synthetic",
            method="ann_dfa",
            seed=13,
            epoch=0,
        )
        self.assertTrue(ann_result["geometry"])

    def test_conv_snn_uses_channel_level_matched_current_geometry(self):
        config = load_config(ROOT / "configs/dvs_gesture/baseline.yaml")
        config["data"].update({"dataset": "synthetic", "time_steps": 3, "batch_size": 2})
        config["model"].update(
            {
                "input_shape": [2, 8, 8],
                "channels": [3, 4],
                "hidden_features": 5,
                "num_classes": 11,
            }
        )
        config["method"].update({"name": "sdfa", "temporal_mode": "pointwise"})
        seed_everything(17)
        model = build_model(config)
        bank = FeedbackBank(model, config)
        dataset = TensorDataset(torch.rand(8, 3, 2, 8, 8), torch.arange(8) % 11)
        loaders = {
            "basis_fit": DataLoader(IndexedSubset(dataset, list(range(4))), batch_size=2),
            "basis_eval": DataLoader(IndexedSubset(dataset, list(range(4, 8))), batch_size=2),
        }
        result = diagnose_snn_checkpoint(
            model=model,
            feedback_bank=bank,
            loaders=loaders,
            device=torch.device("cpu"),
            dataset="synthetic_dvs",
            method="conv_dfa",
            seed=17,
            epoch=0,
            is_bptt=False,
        )
        self.assertEqual(
            {row["ambient_dim"] for row in result["geometry"]}, {3, 4, 5}
        )
        self.assertLess(
            max(row["production_delta_relative_error"] for row in result["parity"]),
            1e-5,
        )
        production_model = copy.deepcopy(model)
        production_bank = copy.deepcopy(bank)
        intervention_model = copy.deepcopy(model)
        intervention_bank = copy.deepcopy(bank)
        production = Trainer(production_model, production_bank, config, torch.device("cpu"))
        intervention = GateInterventionTrainer(
            intervention_model,
            intervention_bank,
            config,
            torch.device("cpu"),
            gate_mode=GateMode.ACTUAL,
            seed=17,
        )
        samples = torch.rand(3, 2, 2, 8, 8)
        target = torch.tensor([1, 2])
        intervention._sample_indices = torch.tensor([4, 5])
        production.optimizer.zero_grad(set_to_none=True)
        production._local_batch(samples, target)["total"].backward()
        intervention.optimizer.zero_grad(set_to_none=True)
        intervention._local_batch(samples, target)["total"].backward()
        production_weights = [layer.conv.weight for layer in production_model.conv_layers] + [
            production_model.fc_hidden.linear.weight
        ]
        intervention_weights = [layer.conv.weight for layer in intervention_model.conv_layers] + [
            intervention_model.fc_hidden.linear.weight
        ]
        for first, second in zip(production_weights, intervention_weights):
            self.assertTrue(torch.allclose(first.grad, second.grad, rtol=1e-5, atol=1e-7))

    def test_targeted_gate_intervention_preserves_unselected_layers(self):
        config = self._config()
        seed_everything(23)
        base_model = build_model(config)
        base_bank = FeedbackBank(base_model, config)
        actual_model, mean_model = copy.deepcopy(base_model), copy.deepcopy(base_model)
        actual_bank, mean_bank = copy.deepcopy(base_bank), copy.deepcopy(base_bank)
        actual = GateInterventionTrainer(
            actual_model,
            actual_bank,
            config,
            torch.device("cpu"),
            gate_mode=GateMode.ACTUAL,
            seed=23,
        )
        targeted = GateInterventionTrainer(
            mean_model,
            mean_bank,
            config,
            torch.device("cpu"),
            gate_mode=GateMode.TEMPORAL_MEAN,
            seed=23,
            intervention_layers={2},
        )
        samples = torch.rand(4, 3, 2, 4, 4)
        target = torch.tensor([1, 2, 3])
        for trainer in (actual, targeted):
            trainer._sample_indices = torch.tensor([4, 5, 6])
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer._local_batch(samples, target)["total"].backward()
        self.assertEqual(
            [row["gate_mode"] for row in targeted.last_gate_rows],
            ["actual", "actual", "temporal_mean_normmatched"],
        )
        for index in (0, 1):
            self.assertTrue(
                torch.allclose(
                    actual_model.hidden_layers[index].linear.weight.grad,
                    mean_model.hidden_layers[index].linear.weight.grad,
                    rtol=1e-6,
                    atol=1e-8,
                )
            )


if __name__ == "__main__":
    unittest.main()
