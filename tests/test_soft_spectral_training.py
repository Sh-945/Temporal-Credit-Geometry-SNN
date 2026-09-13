from __future__ import annotations

import unittest

import torch
from torch.utils.data import TensorDataset

from analysis.soft_spectral.training import (
    deterministic_data_bundle,
    rank_schedule_from_rows,
    select_pilot_alpha,
)
from methods import DynamicBasisTracker


class SoftSpectralTrainingTests(unittest.TestCase):
    def test_data_split_is_deterministic_disjoint_and_training_only_calibration(self):
        dataset = TensorDataset(torch.zeros(100, 2), torch.arange(100) % 10)
        first = deterministic_data_bundle(
            dataset,
            dataset,
            split_seed=3,
            calibration_seed=4,
            validation_fraction=0.10,
            calibration_size=12,
            diagnostic_size=7,
        )
        second = deterministic_data_bundle(
            dataset,
            dataset,
            split_seed=3,
            calibration_seed=4,
            validation_fraction=0.10,
            calibration_size=12,
            diagnostic_size=7,
        )
        self.assertEqual(first.split_rows, second.split_rows)
        training = {row["dataset_index"] for row in first.split_rows if row["split"] == "training"}
        validation = {row["dataset_index"] for row in first.split_rows if row["split"] == "validation"}
        calibration = {row["dataset_index"] for row in first.calibration_rows}
        diagnostic = {row["dataset_index"] for row in first.diagnostic_rows}
        self.assertFalse(training & validation)
        self.assertTrue(calibration <= training)
        self.assertFalse(calibration & diagnostic)

    def test_pilot_selection_uses_accuracy_then_loss_then_larger_alpha(self):
        rows = []
        values = {
            "pilot_soft_025": (0.9500, 0.10),
            "pilot_soft_050": (0.9504, 0.09),
            "pilot_soft_075": (0.9504, 0.09),
        }
        for method, (accuracy, loss) in values.items():
            for epoch in range(41, 51):
                rows.append(
                    {
                        "method": method,
                        "epoch": epoch,
                        "validation_accuracy": accuracy,
                        "validation_loss": loss,
                        "unstable": False,
                    }
                )
        selected = select_pilot_alpha(rows)
        self.assertEqual(selected["selected_alpha"], 0.75)
        self.assertFalse(selected["test_used_for_selection"])

    def test_energy_rank_and_random_schedule_are_exactly_matched(self):
        eigenvalues = torch.tensor([5.0, 3.0, 1.0, 1.0])
        self.assertEqual(DynamicBasisTracker.energy_rank(eigenvalues, 0.80), 2)
        rows = []
        for epoch, ranks in ((1, [4, 3, 2]), (2, [5, 4, 3])):
            for layer_index, rank in enumerate(ranks, start=1):
                rows.append(
                    {
                        "method": "full_soft",
                        "seed": 11,
                        "source_epoch": epoch,
                        "layer": f"hidden_{layer_index}",
                        "k": rank,
                    }
                )
        self.assertEqual(
            rank_schedule_from_rows(rows, "full_soft", 11),
            {1: [4, 3, 2], 2: [5, 4, 3]},
        )


if __name__ == "__main__":
    unittest.main()
