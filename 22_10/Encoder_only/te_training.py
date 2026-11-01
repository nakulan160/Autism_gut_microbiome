"""Focused correctness checks. Run: python -m unittest -v test_training.py"""
import csv
import gzip
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

import train_masked_abundance as training


def profile(run_id, ids, values):
    return training.Profile(run_id, np.asarray(ids, dtype=np.int64),
                            np.asarray(values, dtype=np.float32), tuple(map(str, values)))


class TrainingCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "profiles.tsv.gz"
        self.rows = [
            ["run_id", "Species A", "Unknown", "Species B", "Genus C"],
            ["R1", "0.10000", "99", "0", ""],
            ["R2", "", "0.21", "", "1.2"],
            ["R3", "", "", "", "1.00"],
        ]
        with gzip.open(self.input, "wt", newline="") as handle:
            csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(self.rows)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temporary.cleanup()

    def tiny_config(self, output, epochs):
        return training.Config(input=str(self.input), output_dir=str(output), epochs=epochs,
                               d_model=8, num_heads=2, num_layers=2, d_ff=16,
                               head_hidden=(12, 16), encoder_dropout=0.1,
                               accumulate_samples=2, checkpoint_every=1, plot_every=2,
                               diagnostic_samples=3, threads=1, progress_every=1000,
                               warmup_updates=4, device="cpu")

    def test_filter_preserves_original_cells_and_never_zero_fills(self):
        vocab = training.build_vocabulary(self.input)
        self.assertEqual(vocab.labels, ("Species A", "Species B", "Genus C"))
        filtered = self.root / "filtered.tsv.gz"
        profiles, audit = training.load_profiles(self.input, vocab, filtered)
        with gzip.open(filtered, "rt", newline="") as handle:
            actual = list(csv.reader(handle, delimiter="\t"))
        self.assertEqual(actual, [[row[i] for i in (0, 1, 3, 4)] for row in self.rows])
        self.assertEqual(profiles[0].abundance_text, ("0.10000", "0"))
        self.assertEqual(profiles[0].species_ids.tolist(), [1, 2])
        self.assertEqual(profiles[1].species_ids.tolist(), [3])
        self.assertEqual(audit["reported_zero_cells_retained"], 1)
        self.assertEqual(audit["removed_reported_cells"], 2)
        self.assertEqual(len(profiles), 3)
        # Fingerprint is independent of gzip bytes and removal of the excluded column.
        filtered_vocab = training.build_vocabulary(filtered)
        _, filtered_audit = training.load_profiles(filtered, filtered_vocab)
        self.assertEqual(audit["data_sha256"], filtered_audit["data_sha256"])

    def test_mask_removes_by_saved_mask_not_by_numeric_zero(self):
        p = profile("R", [1, 2, 3], [0, .2, .3])
        masked = training.mask_profile(p, .1, None, [2])
        self.assertEqual(masked.visible_species_ids.tolist(), [1, 3])
        self.assertEqual(masked.visible_abundances[0], 0)
        self.assertEqual(masked.masked_species_ids.tolist(), [2])
        self.assertEqual(masked.target_text, ("0.2",))
        zero_target = training.mask_profile(p, .1, None, [1])
        self.assertEqual(zero_target.targets.tolist(), [0])
        self.assertEqual(zero_target.visible_species_ids.tolist(), [2, 3])

    def test_mask_count_and_cls_only_input(self):
        for n, expected in ((1, 1), (2, 1), (9, 1), (10, 1), (40, 4)):
            p = profile("R", list(range(1, n + 1)), [.1] * n)
            masked = training.mask_profile(p, .1, np.random.default_rng(4))
            self.assertEqual(len(masked.targets), expected)
            self.assertEqual(len(masked.visible_abundances), n - expected)
        masked = training.mask_profile(profile("single", [3], [.1]), .1, np.random.default_rng(0))
        values, ids, targets = training.prepare_batch(masked, torch.device("cpu"))
        self.assertEqual(tuple(values.shape), (1, 0, 1))
        model = training.build_model(self.tiny_config(self.root / "out", 1), 3, torch.device("cpu"))
        loss, _ = training.compute_masked_loss(model(values), ids, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.cls_token.grad)

    def test_four_masked_entries_are_averaged_and_match_exact_ids(self):
        prediction = torch.zeros(1, 3000, requires_grad=True)
        with torch.no_grad():
            prediction[0, [6, 24, 620, 2999]] = torch.tensor([.12, .04, .03, .01])
            prediction[0, 7] = 999  # Adjacent incorrect index must not enter the loss.
        ids = torch.tensor([621, 7, 3000, 25])  # Deliberately not sorted.
        targets = torch.tensor([.02, .10, .01, .05])
        loss, selected = training.compute_masked_loss(prediction, ids, targets)
        self.assertAlmostEqual(float(loss.detach()), .00015, places=9)
        torch.testing.assert_close(selected, torch.tensor([.03, .12, .01, .04]))
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 7]), 0)
        self.assertEqual(float(prediction.grad[0, 100]), 0)
        self.assertNotEqual(float(prediction.grad[0, 6]), 0)
        for invalid in (0, 3001):
            with self.assertRaises(ValueError):
                training.compute_masked_loss(prediction, torch.tensor([invalid]), torch.tensor([.1]))

    def test_no_position_signal_and_cls_is_reused(self):
        model = training.build_model(self.tiny_config(self.root / "out", 1), 3, torch.device("cpu"))
        model.eval()
        values = torch.tensor([[[.1], [.3], [.2]]])
        original_cls = model.cls_token.detach().clone()
        with torch.no_grad():
            first = model(values)
            permuted = model(values[:, [2, 0, 1], :])
            representation = model.encode_sample(values)
        torch.testing.assert_close(first, permuted, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(model.cls_token.detach(), original_cls, rtol=0, atol=0)
        self.assertEqual(tuple(representation.shape), (1, 8))

    def test_accumulation_averages_samples_including_last_partial_group(self):
        class ConstantHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([.1, .2, .3]))
            def forward(self, values):
                return self.weight.unsqueeze(0)

        class RecordingOptimizer(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=1)
                self.gradients = []
            def step(self, closure=None):
                self.gradients.append(self.param_groups[0]["params"][0].grad.clone())

        profiles = [profile(str(i), [1, 2, 3], [.01*i, .01*i+.02, .01*i+.03]) for i in range(17)]
        config = replace(self.tiny_config(self.root / "accum", 1), accumulate_samples=16,
                         save_mask_logs=False)
        model = ConstantHead()
        optimizer = RecordingOptimizer(model.parameters())
        expected_rng = np.random.default_rng(55)
        order = expected_rng.permutation(17)
        expected_gradients, expected_losses = [], []
        for start in range(0, 17, 16):
            group = order[start:start+16]
            gradient = torch.zeros(3)
            for index in group:
                masked = training.mask_profile(profiles[int(index)], .1, expected_rng)
                selected = int(masked.masked_species_ids[0]) - 1
                error = model.weight.detach()[selected] - float(masked.targets[0])
                expected_losses.append(float(error.square()))
                gradient[selected] += 2 * error / len(group)
            expected_gradients.append(gradient)
        metrics, step = training.train_one_epoch(model, optimizer, profiles, config,
                                                 np.random.default_rng(55), torch.device("cpu"), 1, 0)
        self.assertEqual(step, 2)
        self.assertEqual(len(optimizer.gradients), 2)
        for actual, expected in zip(optimizer.gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(metrics["train_mse"], sum(expected_losses) / 17, places=8)

    def test_checkpoint_resume_matches_uninterrupted_training_and_plot_schedule(self):
        uninterrupted = self.root / "uninterrupted"
        resumed = self.root / "resumed"
        training.run_training(self.tiny_config(uninterrupted, 2))
        training.run_training(self.tiny_config(resumed, 1))
        self.assertFalse((resumed / "plots").exists())
        training.run_training(self.tiny_config(resumed, 2), resume="latest")
        full = torch.load(uninterrupted / "checkpoints/epoch_0002.pt", weights_only=False)
        split = torch.load(resumed / "checkpoints/epoch_0002.pt", weights_only=False)
        for key, value in full["model_state_dict"].items():
            torch.testing.assert_close(value, split["model_state_dict"][key], rtol=0, atol=0)
        self.assertEqual(full["global_step"], 4)
        self.assertEqual(full["lr_schedule"], split["lr_schedule"])
        self.assertEqual(full["diagnostic_plan"], split["diagnostic_plan"])
        for a, b in zip(full["history"], split["history"]):
            for key in ("epoch", "train_mse", "train_mae", "train_rmse", "masked_entries"):
                self.assertEqual(a[key], b[key])
        self.assertEqual(len(list((resumed / "plots").glob("*.png"))), 3)
        self.assertTrue(all("0002" in path.name for path in (resumed / "plots").glob("*.png")))
        with (resumed / "epoch_metrics.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)
        for epoch in (1, 2):
            name = f"masked_targets/epoch_{epoch:04d}.tsv.gz"
            with gzip.open(uninterrupted / name, "rt") as a, gzip.open(resumed / name, "rt") as b:
                self.assertEqual(a.read(), b.read())
        vocab = training.build_vocabulary(self.input)
        profiles, audit = training.load_profiles(self.input, vocab)
        cfg = self.tiny_config(resumed, 2)
        model = training.build_model(cfg, vocab.size, torch.device("cpu"))
        optimizer = training.build_optimizer(model, cfg)
        wrong = replace(vocab, labels=tuple(reversed(vocab.labels)))
        with self.assertRaisesRegex(ValueError, "identities/order"):
            training.load_checkpoint(resumed / "checkpoints/epoch_0002.pt", model, optimizer,
                                     cfg, wrong, audit["data_sha256"], np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "identical retained profiles"):
            training.load_checkpoint(resumed / "checkpoints/epoch_0002.pt", model, optimizer,
                                     cfg, vocab, "wrong-data", np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
