from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from analysis.update_relevance.core import capture_dfa_layer_update
from methods import (
    DynamicBasisTracker,
    FeedbackBank,
    SoftSpectralFilter,
    SpectralProjector,
)
from models import build_model
from training.config import load_config
from training.engine import Trainer
from training.seed import seed_everything


ROOT = Path(__file__).resolve().parents[1]


class SpectralFeedbackTests(unittest.TestCase):
    def _config(self):
        config = load_config(ROOT / "configs/nmnist/feedback_subspace_4fc_sdfa.yaml")
        config["data"].update({"dataset": "synthetic", "time_steps": 3, "batch_size": 4})
        config["model"]["input_shape"] = [2, 4, 4]
        config["model"]["hidden_features"] = [8, 7, 6]
        config["training"]["gradient_clip"] = 0.0
        return config

    def test_projection_components_are_orthogonal(self):
        generator = torch.Generator().manual_seed(4)
        basis, _ = torch.linalg.qr(torch.randn(12, 5, generator=generator))
        delta = torch.randn(3, 7, 12, generator=generator)
        projector = SpectralProjector(basis, alpha=0.4)
        parallel, tail, filtered = projector.components(delta)
        cosine = torch.sum(parallel * tail) / (
            torch.linalg.vector_norm(parallel) * torch.linalg.vector_norm(tail)
        )
        self.assertLess(abs(float(cosine)), 1e-6)
        self.assertTrue(torch.allclose(filtered, parallel + 0.4 * tail))

    def test_alpha_one_bypasses_hook_and_is_bitwise_dense(self):
        config = self._config()
        seed_everything(12)
        dense_model = build_model(config)
        dense_bank = FeedbackBank(dense_model, config)
        filtered_model = copy.deepcopy(dense_model)
        filtered_bank = copy.deepcopy(dense_bank)
        spectral = SoftSpectralFilter([8, 7, 6], alpha=1.0)
        for index, dimension in enumerate((8, 7, 6)):
            spectral.set_basis(index, torch.eye(dimension)[:, : max(1, dimension // 2)], 0)
        dense = Trainer(dense_model, dense_bank, config, torch.device("cpu"))
        filtered = Trainer(
            filtered_model,
            filtered_bank,
            config,
            torch.device("cpu"),
            spectral_filter=spectral,
        )
        samples = torch.rand(3, 4, 2, 4, 4)
        target = torch.tensor([0, 1, 2, 3])
        dense_values = dense._local_batch(samples, target)
        filtered_values = filtered._local_batch(samples, target)
        self.assertTrue(torch.equal(dense_values["total"], filtered_values["total"]))
        dense_values["total"].backward()
        filtered_values["total"].backward()
        for first, second in zip(dense_model.parameters(), filtered_model.parameters()):
            self.assertTrue(torch.equal(first.grad, second.grad))
        dense.optimizer.step()
        filtered.optimizer.step()
        for first, second in zip(dense_model.state_dict().values(), filtered_model.state_dict().values()):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(spectral.drain_batch_diagnostics(), [])

    def test_hook_filters_exact_post_gate_delta_before_weight_gradient(self):
        config = self._config()
        seed_everything(21)
        model = build_model(config)
        bank = FeedbackBank(model, config)
        reference_model = copy.deepcopy(model)
        reference_bank = copy.deepcopy(bank)
        basis, _ = torch.linalg.qr(torch.randn(8, 3))
        spectral = SoftSpectralFilter([8, 7, 6], alpha=0.25)
        spectral.set_basis(0, basis, 0)
        trainer = Trainer(
            model, bank, config, torch.device("cpu"), spectral_filter=spectral
        )
        samples = torch.rand(3, 4, 2, 4, 4)
        target = torch.tensor([0, 2, 4, 6])

        with torch.no_grad():
            output, hidden_inputs, _ = reference_model.forward_with_cache(
                samples, detach_temporal=True
            )
            error = output.mean(0) - F.one_hot(target, 10).float()
        capture = capture_dfa_layer_update(
            reference_model,
            reference_bank,
            0,
            hidden_inputs[0],
            error,
            detach_temporal=True,
        )
        filtered_delta, diagnostics = spectral.filter_tensor(0, capture["delta"])
        expected = torch.einsum(
            "tbo,tbi->oi",
            filtered_delta / filtered_delta.numel(),
            hidden_inputs[0],
        )
        values = trainer._local_batch(samples, target)
        values["total"].backward()
        actual = model.hidden_layers[0].linear.weight.grad
        self.assertTrue(torch.allclose(actual, expected, atol=1e-8, rtol=2e-5))
        self.assertLess(abs(diagnostics["parallel_tail_cosine"]), 1e-6)

    def test_eigh_nonconvergence_retries_cpu_float64(self):
        tracker = DynamicBasisTracker([8], energy_threshold=0.95)
        generator = torch.Generator().manual_seed(31)
        samples = torch.randn(32, 8, generator=generator)
        scatter = samples.T @ samples
        real_eigh = torch.linalg.eigh
        dtypes = []

        def flaky_eigh(matrix):
            dtypes.append(matrix.dtype)
            if matrix.dtype == torch.float32:
                raise RuntimeError("linalg.eigh: algorithm failed to converge")
            return real_eigh(matrix)

        with patch("torch.linalg.eigh", side_effect=flaky_eigh):
            with self.assertWarns(RuntimeWarning):
                basis, eigenvalues, rank = tracker.learned_basis(
                    scatter, device=torch.device("cpu")
                )
        self.assertEqual(dtypes, [torch.float32, torch.float64])
        self.assertEqual(basis.dtype, torch.float32)
        self.assertEqual(eigenvalues.dtype, torch.float64)
        self.assertEqual(basis.shape, (8, rank))
        self.assertTrue(
            torch.allclose(
                basis.T @ basis,
                torch.eye(rank),
                atol=1e-5,
                rtol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
