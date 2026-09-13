from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from methods import FeedbackBank
from models import build_model
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import load_config
from training.engine import Trainer


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def _config(self, relative: str):
        return load_config(ROOT / relative, smoke=True)

    def test_fc_local_backward_preserves_time_and_updates_hidden(self):
        config = self._config("configs/nmnist/lodfa.yaml")
        model = build_model(config)
        bank = FeedbackBank(model, config)
        trainer = Trainer(model, bank, config, torch.device("cpu"))
        samples = torch.rand(4, 4, 2, 8, 8).transpose(0, 1)
        target = torch.tensor([0, 1, 2, 3])
        values = trainer._local_batch(samples, target)
        values["total"].backward()
        gradient = model.hidden_layers[0].linear.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_conv_feedback_is_channel_level_and_broadcasts(self):
        config = self._config("configs/dvs_gesture/lodfa.yaml")
        model = build_model(config)
        bank = FeedbackBank(model, config)
        trainer = Trainer(model, bank, config, torch.device("cpu"))
        samples = torch.rand(4, 2, 2, 8, 8)
        target = torch.tensor([0, 1])
        values = trainer._local_batch(samples, target)
        values["total"].backward()
        self.assertEqual(bank.layers[0].hidden_dim, 4)
        gradient = model.conv_layers[0].conv.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_checkpoint_round_trip_includes_feedback_optimizer_and_config(self):
        config = self._config("configs/nmnist/lodfa.yaml")
        model = build_model(config)
        bank = FeedbackBank(model, config)
        trainer = Trainer(model, bank, config, torch.device("cpu"))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "roundtrip.pt"
            save_checkpoint(
                path,
                epoch=0,
                model=model,
                feedback_bank=bank,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                config=config,
                metrics={"accuracy": 0.5},
            )
            before = bank.layers[0].materialize().clone()
            with torch.no_grad():
                bank.layers[0].B_S.zero_()
            state = load_checkpoint(
                path,
                model=model,
                feedback_bank=bank,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
            )
            self.assertTrue(torch.allclose(before, bank.layers[0].materialize()))
            self.assertEqual(state["config"]["method"]["name"], "lodfa")
            self.assertIn("optimizer_state", state)


if __name__ == "__main__":
    unittest.main()
