from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from analysis.feedback_subspace.metrics import OnlineCovariance
from analysis.update_relevance.core import (
    capture_bptt_updates,
    capture_dfa_layer_update,
    pca_basis,
    principal_subspace_overlap,
    project_update,
    random_orthogonal_basis,
    virtual_sgd_step,
)
from methods import FeedbackBank
from models import build_model
from training.config import load_config
from training.seed import seed_everything


ROOT = Path(__file__).resolve().parents[1]


class UpdateRelevanceTests(unittest.TestCase):
    def _config(self):
        config = load_config(
            ROOT / "configs/nmnist/feedback_subspace_4fc_sdfa.yaml"
        )
        config["data"].update(
            {"dataset": "synthetic", "time_steps": 3, "batch_size": 2}
        )
        config["model"]["input_shape"] = [2, 4, 4]
        config["model"]["hidden_features"] = [8, 7, 6]
        return config

    def _capture(self):
        config = self._config()
        seed_everything(41)
        model = build_model(config)
        bank = FeedbackBank(model, config)
        samples = torch.rand(3, 2, 2, 4, 4)
        target = torch.tensor([1, 4])
        with torch.no_grad():
            output, hidden_inputs, _ = model.forward_with_cache(
                samples, detach_temporal=True
            )
            desired = F.one_hot(target, num_classes=10).float()
            error = output.mean(dim=0) - desired
        capture = capture_dfa_layer_update(
            model,
            bank,
            0,
            hidden_inputs[0],
            error,
            detach_temporal=True,
        )
        return model, bank, samples, target, capture

    def test_chain_rule_gradient_has_production_parity(self):
        _model, _bank, _samples, _target, capture = self._capture()
        self.assertLess(capture["parity_weight_max_abs"], 1e-7)
        self.assertLess(capture["parity_weight_relative"], 1e-6)
        self.assertLess(capture["parity_bias_max_abs"], 1e-7)

    def test_identity_projector_exactly_preserves_update(self):
        _model, _bank, _samples, _target, capture = self._capture()
        dimension = capture["delta"].shape[-1]
        projected = project_update(
            capture["delta"],
            capture["weight_gradient"],
            capture["bias_gradient"],
            torch.eye(dimension),
            norm_matching=False,
        )
        self.assertTrue(
            torch.equal(projected["weight_gradient"], capture["weight_gradient"])
        )
        self.assertAlmostEqual(projected["gradient_cosine"], 1.0, places=6)
        self.assertAlmostEqual(projected["relative_error"], 0.0, places=7)

    def test_random_basis_is_orthonormal_and_deterministic(self):
        first = random_orthogonal_basis(20, 7, 123, torch.device("cpu"))
        second = random_orthogonal_basis(20, 7, 123, torch.device("cpu"))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.allclose(first.T @ first, torch.eye(7), atol=1e-6, rtol=1e-6)
        )

    def test_pca_and_principal_overlap_recover_known_subspace(self):
        generator = torch.Generator().manual_seed(8)
        basis, _ = torch.linalg.qr(torch.randn(15, 3, generator=generator))
        observations = torch.randn(80, 3, generator=generator) @ basis.T
        covariance = OnlineCovariance(15)
        covariance.update(observations)
        recovered, _values, metrics = pca_basis(covariance, torch.device("cpu"))
        overlap, angles, _cosines = principal_subspace_overlap(
            basis, recovered, 3
        )
        self.assertEqual(metrics["numerical_rank"], 3)
        self.assertGreater(overlap, 0.9999)
        self.assertLess(float(angles.max()), 0.1)

    def test_bptt_is_diagnostic_and_virtual_step_restores_weights(self):
        model, _bank, samples, target, capture = self._capture()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        bptt = capture_bptt_updates(model, samples, target)
        self.assertEqual(len(bptt["weight_gradient"]), 3)
        self.assertEqual(tuple(bptt["delta"][0].shape), (3, 2, 8))
        with virtual_sgd_step(
            model,
            0,
            capture["weight_gradient"],
            capture["bias_gradient"],
            1e-3,
        ):
            self.assertFalse(
                torch.equal(
                    model.hidden_layers[0].linear.weight,
                    before["hidden_layers.0.linear.weight"],
                )
            )
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))


if __name__ == "__main__":
    unittest.main()
