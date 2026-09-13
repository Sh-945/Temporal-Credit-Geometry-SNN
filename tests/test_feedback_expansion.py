from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd

from analysis.feedback_expansion.capture import capture_layer_signals
from analysis.feedback_expansion.core import (
    direction_energy,
    energy_in_basis,
    mean_pair_cosine,
    merge_covariances,
    pca_from_covariance,
    principal_subspace_metrics,
    scatter_about,
    temporal_coherence,
)
from analysis.feedback_expansion.report import _spearman_with_p
from analysis.feedback_subspace.metrics import OnlineCovariance
from methods import FeedbackBank
from models import build_model
from training.config import load_config
from training.seed import seed_everything


ROOT = Path(__file__).resolve().parents[1]


class FeedbackExpansionTests(unittest.TestCase):
    def _config(self):
        config = load_config(ROOT / "configs/nmnist/feedback_subspace_4fc_sdfa.yaml")
        config["data"].update({"dataset": "synthetic", "time_steps": 3, "batch_size": 4})
        config["model"]["input_shape"] = [2, 4, 4]
        config["model"]["hidden_features"] = [8, 7, 6]
        return config

    def test_exact_gate_multiplies_q_to_actual_delta(self):
        config = self._config()
        seed_everything(7)
        model = build_model(config)
        bank = FeedbackBank(model, config)
        samples = torch.rand(3, 4, 2, 4, 4)
        target = torch.tensor([0, 1, 2, 3])
        with torch.no_grad():
            output, hidden_inputs, _ = model.forward_with_cache(samples, detach_temporal=True)
            error = output.mean(0) - F.one_hot(target, 10).float()
        capture = capture_layer_signals(
            model, bank, 0, hidden_inputs[0], error, detach_temporal=True
        )
        self.assertLess(capture.parity_max_abs, 1e-6)
        self.assertLess(capture.parity_relative, 1e-5)
        self.assertTrue(torch.all(capture.gate >= 0))
        self.assertTrue(torch.all(capture.gate <= 1))

    def test_fit_center_is_used_for_heldout_energy(self):
        fit = OnlineCovariance(4)
        evaluation = OnlineCovariance(4)
        fit.update(torch.tensor([[0.0, 0, 0, 0], [2.0, 0, 0, 0]]))
        evaluation.update(torch.tensor([[5.0, 1, 0, 0], [7.0, -1, 0, 0]]))
        scatter = scatter_about(evaluation, fit.mean)
        captured = energy_in_basis(scatter, torch.eye(4)[:, :1], 1)
        self.assertGreater(captured, 0.95)

    def test_merge_matches_one_pass_covariance(self):
        generator = torch.Generator().manual_seed(3)
        values = torch.randn(50, 9, generator=generator)
        first, second = OnlineCovariance(9), OnlineCovariance(9)
        first.update(values[:13])
        second.update(values[13:])
        merged = merge_covariances([first, second])
        direct = OnlineCovariance(9)
        direct.update(values)
        self.assertEqual(merged.count, direct.count)
        self.assertTrue(torch.allclose(merged.mean, direct.mean, atol=1e-7))
        self.assertTrue(torch.allclose(merged.m2, direct.m2, atol=1e-5))

    def test_pca_overlap_and_direction_energy(self):
        generator = torch.Generator().manual_seed(9)
        basis, _ = torch.linalg.qr(torch.randn(12, 3, generator=generator))
        values = torch.randn(100, 3, generator=generator) @ basis.T
        covariance = OnlineCovariance(12)
        covariance.update(values)
        summary = pca_from_covariance(covariance, torch.device("cpu"), max_basis=3)
        metrics = principal_subspace_metrics(basis, summary.basis, 3)
        energies = direction_energy(covariance.m2, summary.basis)
        self.assertGreater(metrics["overlap"], 0.999)
        self.assertAlmostEqual(float(energies.sum()), 1.0, places=5)

    def test_temporal_coherence_extremes(self):
        identical = torch.ones(4, 3, 5)
        coherence, cosine = temporal_coherence(identical)
        self.assertTrue(torch.allclose(coherence, torch.ones_like(coherence)))
        self.assertTrue(torch.allclose(cosine, torch.ones_like(cosine)))
        cancel = identical.clone()
        cancel[2:] *= -1
        coherence, _ = temporal_coherence(cancel)
        self.assertTrue(torch.allclose(coherence, torch.zeros_like(coherence)))

    def test_pair_sampling_separates_known_class_vectors(self):
        vectors = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]])
        labels = torch.tensor([0, 0, 1, 1])
        same, different, same_n, different_n = mean_pair_cosine(
            vectors, labels, seed=2, pairs=100
        )
        self.assertEqual(same, 1.0)
        self.assertEqual(different, 0.0)
        self.assertGreater(same_n, 0)
        self.assertGreater(different_n, 0)

    def test_spearman_implementation_handles_ties_and_perfect_order(self):
        rho, p_value = _spearman_with_p(
            pd.Series([1, 2, 2, 4, 5]), pd.Series([10, 20, 20, 40, 50])
        )
        self.assertAlmostEqual(rho, 1.0, places=12)
        self.assertEqual(p_value, 0.0)


if __name__ == "__main__":
    unittest.main()
