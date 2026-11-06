from __future__ import annotations

import unittest

import torch
from torch import nn

from breaching.cases.defenses import (
    QuantizationDefense,
    prune_gradient,
    quantize_gradient,
)
from breaching.cases.users import UserSingleStep


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, inputs, **kwargs):
        return self.fc2(self.relu(self.fc1(inputs)))


class DummyDataLoader:
    def __init__(self):
        self.name = "dummy_dataset"

    def __iter__(self):
        yield {
            "inputs": torch.randn(4, 10),
            "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        }

    def __len__(self):
        return 1


class TestQuantizationDefense(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantize_gradient_roundtrip(self):
        grad = torch.randn(16, 32, 3, 3, dtype=torch.float32)
        qg = quantize_gradient(grad, nbits=4, block_size=64, stochastic=False)

        self.assertEqual(qg.shape, grad.shape)
        self.assertEqual(qg.dtype, grad.dtype)

        # High directional retention
        cos_sim = torch.cosine_similarity(grad.flatten(), qg.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.95)

    def test_quantize_int8(self):
        grad = torch.randn(128, dtype=torch.float32)
        qg = quantize_gradient(grad, nbits=8, block_size=32)
        cos_sim = torch.cosine_similarity(grad.flatten(), qg.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.99)

    def test_stochastic_quantization_unbiased(self):
        grad = torch.randn(64, dtype=torch.float32)
        # Average of 500 stochastic quantizations should approximate original gradient
        samples = [
            quantize_gradient(grad, nbits=4, block_size=64, stochastic=True)
            for _ in range(500)
        ]
        mean_sample = torch.stack(samples).mean(dim=0)
        error = torch.norm(mean_sample - grad) / torch.norm(grad)
        self.assertLess(error.item(), 0.15)

    def test_preserve_norm(self):
        grad = torch.randn(100, dtype=torch.float32) * 5.0
        orig_norm = torch.norm(grad, 2).item()
        qg = quantize_gradient(grad, nbits=4, block_size=32, preserve_norm=True)
        qg_norm = torch.norm(qg, 2).item()
        self.assertAlmostEqual(orig_norm, qg_norm, places=4)

    def test_prune_gradient_topk(self):
        grad = torch.randn(100, dtype=torch.float32)
        pg = prune_gradient(grad, sparsity=0.9, method="topk")
        zeros = (pg == 0.0).sum().item()
        self.assertEqual(zeros, 90)

    def test_prune_gradient_random(self):
        grad = torch.randn(100, dtype=torch.float32)
        pg = prune_gradient(grad, sparsity=0.8, method="random")
        zeros = (pg == 0.0).sum().item()
        self.assertEqual(zeros, 80)

    def test_quantization_defense_class(self):
        defense = QuantizationDefense(nbits=4, block_size=64)
        grads = [torch.randn(32, 10), torch.randn(10)]
        orig_clone = [g.clone() for g in grads]

        defended = defense(grads, in_place=False)
        self.assertEqual(len(defended), 2)
        for d, o in zip(defended, orig_clone):
            self.assertEqual(d.shape, o.shape)
            cos_sim = torch.cosine_similarity(d.flatten(), o.flatten(), dim=0).item()
            self.assertGreater(cos_sim, 0.95)

        stats = defense.calc_compression_stats(grads)
        self.assertIn("compression_ratio", stats)
        self.assertGreater(stats["compression_ratio"], 5.0)

    def test_gradient_matching_inversion_distortion(self):
        # Inversion attack objective: || grad_true - grad_candidate ||^2
        grad_true = torch.randn(100, dtype=torch.float32)
        qg = quantize_gradient(grad_true, nbits=4, block_size=32)

        distortion = torch.norm(grad_true - qg, 2).item()
        self.assertGreater(distortion, 0.0)

    def test_user_integration(self):
        model = DummyModel()
        loss_fn = nn.CrossEntropyLoss()
        dataloader = DummyDataLoader()
        setup = {"device": torch.device("cpu"), "dtype": torch.float}

        cfg_user = {
            "num_data_points": 4,
            "provide_labels": True,
            "provide_num_data_points": True,
            "provide_buffers": False,
            "local_diff_privacy": {
                "gradient_noise": 0.0,
                "input_noise": 0.0,
                "distribution": "gaussian",
                "per_example_clipping": 0.0,
            },
            "quantization": {
                "nbits": 4,
                "block_size": 32,
                "stochastic": False,
            },
        }

        user = UserSingleStep(
            model, loss_fn, dataloader, setup, idx=0, cfg_user=cfg_user
        )
        self.assertTrue(any("quantization" in r.lower() for r in user.defense_repr))

        server_payload = {
            "parameters": [p.clone() for p in model.parameters()],
            "buffers": None,
        }
        shared_data, _true_user_data = user.compute_local_updates(server_payload)

        self.assertIn("gradients", shared_data)
        for g in shared_data["gradients"]:
            self.assertFalse(torch.isnan(g).any())

    def test_empty_tensor(self):
        grad = torch.empty((0,), dtype=torch.float32)
        qg = quantize_gradient(grad, nbits=4, block_size=64)
        self.assertEqual(qg.numel(), 0)


if __name__ == "__main__":
    unittest.main()
