"""
test_aggregation.py — Unit tests for episode memory and server aggregation
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import AggregationConfig
from state_manager import ClientState
from episode_memory import DriftEpisode, DriftEpisodeManager
from reliability_engine import ClientReliabilityEngine
from aggregation import AggregationEngine


class TestAggregationAndEpisodeMemory(unittest.TestCase):

    def setUp(self):
        self.cfg = AggregationConfig(
            base_method="episode_weighted",
            episode_memory_depth=5,
            episode_recency_decay=0.7,
            selective_layer_aggregation=True,
            min_weight_floor=0.05,
        )
        self.num_clients = 4
        self.agg_engine = AggregationEngine(self.cfg, self.num_clients)

        self.global_params = {
            "W_0": np.zeros((4, 4), dtype=np.float32),
            "b_0": np.zeros((4,), dtype=np.float32),
            "W_1": np.zeros((4, 2), dtype=np.float32),
            "b_1": np.zeros((2,), dtype=np.float32),
        }
        self.layer_groups = {"base": [0], "head": [1]}

    def test_episode_memory_scoring_and_recurrence(self):
        mgr = DriftEpisodeManager(self.num_clients, self.cfg)
        
        # Client 0: smooth recovery, non-recurring
        mgr.open_episode(0, 1, "persistent", 0.8, ["head"], "selective", {1: 2.0})
        mgr.record_recovery(0, 3, 0.90)
        mgr.close_episode(0, 4, 0.92)
        weight_clean = mgr.get_episode_weight(0)
        self.assertGreaterEqual(weight_clean, 0.80)

        # Client 1: recurring unmitigated oscillation
        mgr.open_episode(1, 1, "persistent", 0.9, ["head"], "selective", {1: 3.0})
        mgr.close_episode(1, 3, 0.50)
        # Recurrence
        mgr.open_episode(1, 6, "persistent", 0.9, ["head"], "selective", {1: 3.0})
        mgr.close_episode(1, 8, 0.55)
        weight_recurring = mgr.get_episode_weight(1)
        self.assertLess(weight_recurring, weight_clean, "Recurring drifter must receive lower aggregation weight")

    def test_reliability_engine(self):
        rel_engine = ClientReliabilityEngine(self.num_clients, window_size=5)
        
        # Client 0: stable gradients
        for _ in range(5):
            rel_engine.update(0, grad_norm=1.0, accuracy=0.85, was_suppressed=False)
        rel_0 = rel_engine.get_reliability(0)

        # Client 1: erratic and suppressed
        for i in range(5):
            rel_engine.update(1, grad_norm=float(i * 10), accuracy=0.40, was_suppressed=(i % 2 == 0))
        rel_1 = rel_engine.get_reliability(1)

        self.assertGreater(rel_0, rel_1, "Stable client must have strictly higher reliability score")

    def test_partial_layer_reconstruction(self):
        # Client 0 sends full update
        upd_0 = {
            "W_0": np.ones((4, 4), dtype=np.float32) * 1.0,
            "b_0": np.ones((4,), dtype=np.float32) * 1.0,
            "W_1": np.ones((4, 2), dtype=np.float32) * 1.0,
            "b_1": np.ones((2,), dtype=np.float32) * 1.0,
        }
        # Client 1 sends selective update (head only)
        upd_1 = {
            "W_1": np.ones((4, 2), dtype=np.float32) * 3.0,
            "b_1": np.ones((2,), dtype=np.float32) * 3.0,
        }
        # Client 2 suppressed
        upd_2 = None

        client_updates = {0: upd_0, 1: upd_1, 2: upd_2}
        data_sizes = {0: 100, 1: 100, 2: 100}
        states = {0: ClientState.STABLE, 1: ClientState.PERSISTENT, 2: ClientState.TRANSIENT}
        modifiers = {0: 1.0, 1: 0.7, 2: 0.0}

        new_global, diag = self.agg_engine.aggregate(
            client_updates=client_updates,
            client_data_sizes=data_sizes,
            client_states=states,
            state_modifiers=modifiers,
            global_params=self.global_params,
            layer_groups=self.layer_groups,
        )

        self.assertIn("W_0", new_global)
        self.assertIn("W_1", new_global)
        # Base layer W_0 was only transmitted by client 0 -> value should be 1.0
        np.testing.assert_allclose(new_global["W_0"], 1.0, atol=1e-5)
        # Head layer W_1 received updates from both client 0 (1.0) and client 1 (3.0) -> blended
        self.assertTrue(1.0 < float(new_global["W_1"][0, 0]) < 3.0)


if __name__ == "__main__":
    unittest.main()
