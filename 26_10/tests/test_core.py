import sys, unittest, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from embeddings import abundance_bins
from losses import masked_loss
from masking import prepare_sample, choose_mask
from split import make_split
from model import MicrobiomeEncoder
from config import load_config


class CoreTests(unittest.TestCase):
    def test_round_and_small_values(self):
        self.assertEqual(
            abundance_bins(
                torch.tensor([0.0, 0.03, 0.49, 0.5, 88.304, 88.5, 100.0])
            ).tolist(),
            [0, 0, 0, 1, 88, 89, 100],
        )

    def test_exact_species_loss(self):
        p = torch.tensor([100.0, 2.0, 100.0, 6.0], requires_grad=True)
        loss = masked_loss(p, torch.tensor([2, 4]), torch.tensor([1.0, 4.0]))
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        self.assertEqual(p.grad.tolist(), [0.0, 1.0, 0.0, 2.0])

    def test_removal_keeps_reported_zero(self):
        r = dict(ids=torch.tensor([1, 2, 3]), values=torch.tensor([0.0, 0.03, 2.0]))
        ids, values, ti, t = prepare_sample(r, torch.tensor([2]), "cpu")
        self.assertEqual(ids.tolist(), [1, 2])
        self.assertEqual(values[0], 0)

    def test_split(self):
        a, b = make_split(100, 0.2, 42)
        self.assertEqual((len(a), len(b)), (80, 20))
        self.assertFalse(set(a) & set(b))
        self.assertEqual((a, b), make_split(100, 0.2, 42))

    def test_cls_only_and_permutation(self):
        c = load_config(Path(__file__).resolve().parents[1] / "settings.json")
        c.update(
            species_dim=8,
            abundance_dim=8,
            layers=1,
            heads=2,
            feedforward=32,
            head_dims=[16],
            dropout=0.0,
        )
        m = MicrobiomeEncoder(c, 4).eval()
        self.assertEqual(
            m(torch.tensor([], dtype=torch.long), torch.tensor([])).shape, (4,)
        )
        ids = torch.tensor([1, 3])
        a = torch.tensor([0.03, 88.3])
        torch.testing.assert_close(
            m(ids, a), m(ids.flip(0), a.flip(0)), atol=1e-6, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
