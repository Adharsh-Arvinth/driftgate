"""
test_transmission.py — Unit tests for communication control, layer selection, and compression
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import CommConfig, ModelConfig
from state_manager import ClientState
from compression import topk_sparsify, random_sparsify, quantize_params, compute_compression_stats
from communication_controller import CommunicationController


class TestTransmissionAndCompression(unittest.TestCase):

    def setUp(self):
        self.comm_cfg = CommConfig(
            compression_method="topk",
            topk_fraction=0.5,
            transient_suppress_transmission=True,
            persistent_selective_transmission=True,
        )
        self.model_cfg = ModelConfig()
        self.total_params = 1000
        self.controller = CommunicationController(self.comm_cfg, self.model_cfg, self.total_params)

        self.full_params = {
            "W_0": np.ones((10, 10), dtype=np.float32),
            "b_0": np.ones((10,), dtype=np.float32),
            "W_1": np.ones((10, 10), dtype=np.float32),
            "b_1": np.ones((10,), dtype=np.float32),
            "W_2": np.ones((10, 5), dtype=np.float32),
            "b_2": np.ones((5,), dtype=np.float32),
            "W_3": np.ones((5, 2), dtype=np.float32),
            "b_3": np.ones((2,), dtype=np.float32),
        }
        self.layer_groups = {"base": [0, 1], "head": [2, 3]}

    def test_transient_suppression(self):
        prepared, mode, sent = self.controller.prepare_update(
            client_id=0,
            client_state=ClientState.TRANSIENT,
            full_params=self.full_params,
            affected_layer_groups=["head"],
            layer_groups=self.layer_groups,
            round_num=1,
        )
        self.assertIsNone(prepared, "Transient client payload must be None")
        self.assertEqual(mode, "none")
        self.assertEqual(sent, 0, "Transient client must send 0 parameters")

    def test_persistent_selective_transmission(self):
        prepared, mode, sent = self.controller.prepare_update(
            client_id=1,
            client_state=ClientState.PERSISTENT,
            full_params=self.full_params,
            affected_layer_groups=["head"],
            layer_groups=self.layer_groups,
            round_num=1,
        )
        self.assertIsNotNone(prepared)
        self.assertEqual(mode, "selective")
        # Only head layers (W_2, b_2, W_3, b_3) must be in payload
        for k in prepared.keys():
            layer_idx = int(k.split("_")[1])
            self.assertIn(layer_idx, [2, 3], "Base layers must not be transmitted under persistent drift")

    def test_topk_compression_math(self):
        params = {"W": np.array([[1.0, 10.0], [0.1, 50.0]], dtype=np.float32)}
        sparse = topk_sparsify(params, k_fraction=0.5)
        # Should retain only top 2 largest: 50.0 and 10.0
        flat = sparse["W"].flatten()
        self.assertEqual(np.count_nonzero(flat), 2)
        self.assertIn(50.0, flat)
        self.assertIn(10.0, flat)
        self.assertNotIn(0.1, flat[flat != 0])

    def test_quantization(self):
        params = {"W": np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32)}
        quant = quantize_params(params, bits=8)
        self.assertEqual(quant["W"].shape, params["W"].shape)
        # Numerical error within 8-bit precision
        diff = np.max(np.abs(quant["W"] - params["W"]))
        self.assertLess(diff, 0.02)

    def test_comm_stats_accounting(self):
        # Simulate transient and selective updates
        self.controller.prepare_update(0, ClientState.TRANSIENT, self.full_params, ["head"], self.layer_groups, 1)
        self.controller.prepare_update(1, ClientState.PERSISTENT, self.full_params, ["head"], self.layer_groups, 1)
        
        stats = self.controller.get_stats()
        summary = stats.get_overall_summary()
        self.assertGreater(summary["suppressed_client_rounds"], 0)
        self.assertGreater(summary["selective_client_rounds"], 0)
        self.assertGreater(summary["bandwidth_reduction_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
