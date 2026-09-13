from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from analysis.feedback_subspace.diagnostic import _capture_post_gate_delta
from analysis.feedback_subspace.metrics import DualCovariance, spectrum_statistics
from methods import FeedbackBank
from models import build_model
from training.config import load_config
from training.engine import Trainer
from training.seed import seed_everything


ROOT = Path(__file__).resolve().parents[1]


class FeedbackSubspaceTests(unittest.TestCase):
    def _config(self):
        config = load_config(
            ROOT / "configs/nmnist/feedback_subspace_4fc_sdfa.yaml"
        )
        config["data"].update(
            {
                "dataset": "synthetic",
                "root": ".",
                "time_steps": 3,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            }
        )
        config["model"]["input_shape"] = [2, 4, 4]
        config["model"]["hidden_features"] = [8, 7, 6]
        config["training"]["epochs"] = 1
        return config

    def test_streaming_centered_spectrum_recovers_known_rank(self):
        torch.manual_seed(3)
        factors = torch.randn(40, 3)
        basis = torch.randn(3, 12)
        observations = factors @ basis + 2.0
        covariance = DualCovariance(12)
        covariance.update(observations[:17])
        covariance.update(observations[17:])
        metrics, _singular, _energy, _cumulative = spectrum_statistics(
            covariance.raw,
            algebraic_max_dim=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(metrics["numerical_rank"], 3)
        self.assertLessEqual(metrics["r99"], 3)

    def test_captured_delta_is_exact_surrogate_gated_teaching(self):
        config = self._config()
        seed_everything(11)
        model = build_model(config)
        bank = FeedbackBank(model, config)
        samples = torch.rand(3, 2, 2, 4, 4)
        target = torch.tensor([1, 2])
        with torch.no_grad():
            output, hidden_inputs, _ = model.forward_with_cache(
                samples, detach_temporal=True
            )
            error = output.mean(dim=0) - F.one_hot(target, 10).float()
            teaching = bank.project(0, error)
        actual = _capture_post_gate_delta(
            model,
            0,
            hidden_inputs[0],
            teaching,
            detach_temporal=True,
        )

        layer = model.hidden_layers[0]
        with torch.no_grad():
            time, batch = hidden_inputs[0].shape[:2]
            current = layer.linear(hidden_inputs[0].reshape(time * batch, -1)).reshape(
                time, batch, -1
            )
            membrane = torch.zeros_like(current[0])
            previous_spike = torch.zeros_like(current[0])
            expected = []
            for step_current in current:
                membrane = layer.neuron.decay * membrane * (1.0 - previous_spike) + step_current
                difference = membrane - layer.neuron.threshold
                gate = 1.0 / (1.0 + layer.neuron.surrogate_beta * difference.abs()).square()
                expected.append(teaching * gate)
                previous_spike = (difference >= 0).float()
                membrane = membrane.detach()
                previous_spike = previous_spike.detach()
        self.assertTrue(torch.allclose(actual, torch.stack(expected), atol=1e-6, rtol=1e-5))

    def test_offline_capture_does_not_change_next_parameter_update(self):
        config = self._config()
        seed_everything(23)
        model_a = build_model(config)
        bank_a = FeedbackBank(model_a, config)
        model_b = copy.deepcopy(model_a)
        bank_b = copy.deepcopy(bank_a)
        trainer_a = Trainer(model_a, bank_a, config, torch.device("cpu"))
        trainer_b = Trainer(model_b, bank_b, config, torch.device("cpu"))
        samples = torch.rand(3, 2, 2, 4, 4)
        target = torch.tensor([0, 3])

        with torch.no_grad():
            output, hidden_inputs, _ = model_b.forward_with_cache(
                samples, detach_temporal=True
            )
            error = output.mean(dim=0) - F.one_hot(target, 10).float()
            teaching = bank_b.project(0, error)
        before = {name: value.clone() for name, value in model_b.state_dict().items()}
        _capture_post_gate_delta(
            model_b, 0, hidden_inputs[0], teaching, detach_temporal=True
        )
        for name, value in model_b.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))
        self.assertTrue(all(parameter.grad is None for parameter in model_b.parameters()))

        values_a = trainer_a._local_batch(samples, target)
        values_b = trainer_b._local_batch(samples, target)
        self.assertTrue(torch.equal(values_a["mean_output"], values_b["mean_output"]))
        self.assertTrue(torch.equal(values_a["total"], values_b["total"]))
        values_a["total"].backward()
        values_b["total"].backward()
        trainer_a.optimizer.step()
        trainer_b.optimizer.step()
        for value_a, value_b in zip(model_a.state_dict().values(), model_b.state_dict().values()):
            self.assertTrue(torch.equal(value_a, value_b))


if __name__ == "__main__":
    unittest.main()
