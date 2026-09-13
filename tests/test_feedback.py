from __future__ import annotations

import unittest

import torch

from methods.lodfa import LowRankFeedback, select_energy_rank
from methods.regularization import feedback_regularization


class FeedbackTests(unittest.TestCase):
    def test_factorized_projection_matches_materialized_matrix(self):
        torch.manual_seed(1)
        feedback = LowRankFeedback(8, 5, rank=3, trainable=True)
        error = torch.randn(4, 5)
        self.assertTrue(
            torch.allclose(
                feedback(error),
                error @ feedback.materialize().transpose(0, 1),
                atol=1e-6,
            )
        )
        self.assertEqual(torch.linalg.matrix_rank(feedback.materialize()).item(), 3)

    def test_energy_threshold_selects_smallest_rank(self):
        values = torch.tensor([4.0, 3.0, 0.1])
        self.assertEqual(select_energy_rank(values, 0.60), 1)
        self.assertEqual(select_energy_rank(values, 0.95), 2)

    def test_active_rank_changes_feedback_shape_capacity(self):
        feedback = LowRankFeedback(9, 6, rank=4, trainable=True)
        feedback.set_active_rank(2)
        self.assertEqual(feedback.active_rank, 2)
        self.assertLessEqual(torch.linalg.matrix_rank(feedback.materialize()).item(), 2)

    def test_structural_losses_reach_feedback_parameters(self):
        feedback = LowRankFeedback(8, 5, rank=3, trainable=True)
        losses = feedback_regularization(feedback, torch.randn(8, 10))
        total = losses["alignment"] + losses["orthogonal"] + losses["hoyer"]
        total.backward()
        self.assertIsNotNone(feedback.B_U.grad)
        self.assertIsNotNone(feedback.B_S.grad)
        self.assertIsNotNone(feedback.B_V.grad)


if __name__ == "__main__":
    unittest.main()
