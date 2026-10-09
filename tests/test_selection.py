"""
test_selection.py — Unit tests for Client Selection Strategies
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import SelectionConfig
from state_manager import ClientState
from client_selector import (
    DriftGateSelector,
    RandomSelector,
    PowerOfDSelector,
    OortSelector,
    create_selector,
)


class TestClientSelection(unittest.TestCase):

    def setUp(self):
        self.cfg = SelectionConfig(clients_per_round=3, seed=42)
        self.num_clients = 8

    def test_driftgate_selector_excludes_transient_and_drifting(self):
        selector = DriftGateSelector(self.cfg, self.num_clients)
        
        client_states = {cid: ClientState.STABLE for cid in range(self.num_clients)}
        client_states[0] = ClientState.TRANSIENT
        client_states[1] = ClientState.DRIFTING
        
        drift_severities = {cid: 0.1 for cid in range(self.num_clients)}
        reliabilities = {cid: 0.8 for cid in range(self.num_clients)}
        comm_budgets = {cid: 0.3 for cid in range(self.num_clients)}

        selected, diag = selector.select(
            round_num=1,
            client_states=client_states,
            drift_severities=drift_severities,
            reliabilities=reliabilities,
            comm_budgets=comm_budgets,
        )

        self.assertEqual(len(selected), 3)
        self.assertNotIn(0, selected, "Transient clients must not be selected")
        self.assertNotIn(1, selected, "Drifting clients awaiting verification must not be selected")

    def test_driftgate_prioritizes_persistent_and_reliable(self):
        selector = DriftGateSelector(self.cfg, self.num_clients)
        
        client_states = {cid: ClientState.STABLE for cid in range(self.num_clients)}
        # Client 2 is persistent with high drift severity and high reliability
        client_states[2] = ClientState.PERSISTENT
        
        drift_severities = {cid: 0.0 for cid in range(self.num_clients)}
        drift_severities[2] = 0.95
        
        reliabilities = {cid: 0.6 for cid in range(self.num_clients)}
        reliabilities[2] = 0.98
        
        comm_budgets = {cid: 0.5 for cid in range(self.num_clients)}
        comm_budgets[2] = 0.2  # Transmitting fewer layers is bandwidth-efficient

        selected, _ = selector.select(
            round_num=2,
            client_states=client_states,
            drift_severities=drift_severities,
            reliabilities=reliabilities,
            comm_budgets=comm_budgets,
        )

        self.assertIn(2, selected, "High-reliability persistent drifter must be prioritized")

    def test_baseline_selectors(self):
        # Random selector
        rand_sel = create_selector("random", self.cfg, self.num_clients)
        s_rand, _ = rand_sel.select(1)
        self.assertEqual(len(s_rand), 3)

        # Power-of-d
        pow_sel = create_selector("pow_d", self.cfg, self.num_clients)
        s_pow, _ = pow_sel.select(1, data_sizes={i: 100 * (i + 1) for i in range(self.num_clients)})
        self.assertEqual(len(s_pow), 3)

        # Oort selector
        oort_sel = create_selector("oort", self.cfg, self.num_clients)
        s_oort, _ = oort_sel.select(1)
        self.assertEqual(len(s_oort), 3)


if __name__ == "__main__":
    unittest.main()
