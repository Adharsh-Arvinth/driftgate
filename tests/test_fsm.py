"""
test_fsm.py — Unit tests for Client Finite State Machine (FSM)
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import ClientFSMConfig
from drift_detector import DriftSignal, DriftType
from state_manager import ClientFSM, ClientState, FederationStateManager


class TestClientFSM(unittest.TestCase):

    def setUp(self):
        self.cfg = ClientFSMConfig(
            classification_wait_rounds=3,
            transient_suppression_rounds=4,
            recovery_accuracy_threshold=0.75,
            max_adapting_rounds=6,
            min_persistent_selective_rounds=2,
        )
        self.fsm = ClientFSM(client_id=0, cfg=self.cfg)

    def test_initial_state(self):
        self.assertEqual(self.fsm.state, ClientState.STABLE)
        self.assertTrue(self.fsm.should_transmit())
        self.assertEqual(self.fsm.transmission_mode(), "full")

    def test_transient_lifecycle(self):
        # 1. Stable -> Drifting
        s1 = self.fsm.update(DriftSignal.DRIFT, DriftType.UNKNOWN, 0.5, 0.8, 1)
        self.assertEqual(s1, ClientState.DRIFTING)
        self.assertFalse(self.fsm.should_transmit())

        # 2. Drifting -> Transient classification
        for r in range(2, 2 + self.cfg.classification_wait_rounds):
            self.fsm.update(DriftSignal.DRIFT, DriftType.TRANSIENT, 0.8, 0.7, r)
        self.assertEqual(self.fsm.state, ClientState.TRANSIENT)
        self.assertEqual(self.fsm.transmission_mode(), "none")
        self.assertEqual(self.fsm.get_weight_modifier(), 0.0)

        # 3. Transient suppression rounds elapse -> Stable
        for r in range(10, 10 + self.cfg.transient_suppression_rounds):
            self.fsm.update(DriftSignal.STABLE, DriftType.NONE, 0.0, 0.8, r)
        self.assertEqual(self.fsm.state, ClientState.STABLE)
        self.assertTrue(self.fsm.should_transmit())

    def test_persistent_and_recovery_lifecycle(self):
        # 1. Stable -> Drifting
        self.fsm.update(DriftSignal.DRIFT, DriftType.UNKNOWN, 0.5, 0.7, 1)
        
        # 2. Drifting -> Persistent classification
        for r in range(2, 2 + self.cfg.classification_wait_rounds):
            self.fsm.update(DriftSignal.DRIFT, DriftType.PERSISTENT, 0.9, 0.5, r)
        self.assertEqual(self.fsm.state, ClientState.PERSISTENT)
        self.assertEqual(self.fsm.transmission_mode(), "selective")

        # 3. Persistent -> Adapting (drift resolves)
        for r in range(10, 10 + self.cfg.min_persistent_selective_rounds):
            self.fsm.update(DriftSignal.STABLE, DriftType.NONE, 0.0, 0.6, r)
        self.assertEqual(self.fsm.state, ClientState.ADAPTING)

        # 4. Adapting -> Eligible (accuracy recovered)
        self.fsm.update(DriftSignal.STABLE, DriftType.NONE, 0.0, 0.88, 15)
        self.assertEqual(self.fsm.state, ClientState.ELIGIBLE)

        # 5. Eligible -> Stable (confirmed)
        self.fsm.update(DriftSignal.STABLE, DriftType.NONE, 0.0, 0.90, 16)
        self.assertEqual(self.fsm.state, ClientState.STABLE)

    def test_tick_idle_for_unselected_client(self):
        # Client in Transient should advance timer during idle rounds without signal resets
        self.fsm.state = ClientState.TRANSIENT
        self.fsm.rounds_in_state = 0
        for r in range(1, self.cfg.transient_suppression_rounds + 1):
            self.fsm.tick_idle(r)
        self.assertEqual(self.fsm.state, ClientState.STABLE)

    def test_state_manager_registry(self):
        mgr = FederationStateManager(num_clients=5, cfg=self.cfg)
        self.assertEqual(len(mgr.get_eligible_clients()), 5)
        
        # Drift client 0 into transient
        mgr.fsms[0].state = ClientState.TRANSIENT
        eligible = mgr.get_eligible_clients()
        self.assertNotIn(0, eligible)
        self.assertEqual(len(eligible), 4)


if __name__ == "__main__":
    unittest.main()
