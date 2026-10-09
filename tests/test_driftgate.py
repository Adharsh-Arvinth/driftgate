import sys
import os
import unittest
import numpy as np

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import get_fast_config, Config
from data import FederationDataLoader, SyntheticSensorDataset
from drift_detector import ADWIN, PageHinkley, DDM, TransienceClassifier, DriftType, DriftSignal
from state_manager import ClientFSM, ClientState, FederationStateManager
from model import NumpyMLP
from communication_controller import CommunicationController, topk_sparsify
from aggregation import DriftEpisodeManager, AggregationEngine
from client_selector import DriftGateSelector
from federation import FederationRunner


class TestDriftGateComponents(unittest.TestCase):

    def test_adwin_detector(self):
        adwin = ADWIN(delta=0.01)
        # Stationary signal
        for _ in range(30):
            adwin.update(1.0 + np.random.normal(0, 0.05))
        self.assertFalse(adwin.is_change_detected)
        # Sudden jump
        drift_found = False
        for _ in range(20):
            if adwin.update(5.0 + np.random.normal(0, 0.05)):
                drift_found = True
                break
        self.assertTrue(drift_found, "ADWIN should detect distribution mean jump")

    def test_transience_classifier(self):
        tc = TransienceClassifier(transient_window=3, persistence_threshold=0.6)
        # Transient drift: fires once then reverts
        tc.observe(True, 1)
        tc.observe(False, 2)
        res = tc.observe(False, 3)
        self.assertEqual(res, DriftType.TRANSIENT)

        # Persistent drift: continues firing
        tc_p = TransienceClassifier(transient_window=3, persistence_threshold=0.6)
        tc_p.observe(True, 1)
        tc_p.observe(True, 2)
        res_p = tc_p.observe(True, 3)
        self.assertEqual(res_p, DriftType.PERSISTENT)

    def test_client_fsm_transitions(self):
        cfg = get_fast_config().fsm
        fsm = ClientFSM(client_id=0, cfg=cfg)
        self.assertEqual(fsm.state, ClientState.STABLE)

        # Trigger drift
        fsm.update(DriftSignal.DRIFT, DriftType.UNKNOWN, 0.5, 0.8, 1)
        self.assertEqual(fsm.state, ClientState.DRIFTING)
        self.assertFalse(fsm.should_transmit())

        # Classify as transient
        for r in range(2, 2 + cfg.classification_wait_rounds):
            fsm.update(DriftSignal.DRIFT, DriftType.TRANSIENT, 0.8, 0.75, r)
        self.assertEqual(fsm.state, ClientState.TRANSIENT)
        self.assertEqual(fsm.transmission_mode(), "none")

    def test_topk_compression(self):
        params = {"w": np.array([[1.0, 5.0], [0.1, 10.0]])}
        sparse = topk_sparsify(params, k_fraction=0.5)
        flat = sparse["w"].flatten()
        non_zeros = np.count_nonzero(flat)
        self.assertEqual(non_zeros, 2)
        self.assertEqual(sparse["w"][1, 1], 10.0)

    def test_end_to_end_simulation(self):
        cfg = get_fast_config()
        cfg.training.num_rounds = 2
        cfg.data.num_clients = 3
        cfg.selection.clients_per_round = 2
        runner = FederationRunner(cfg, seed=42)
        res = runner.run()
        self.assertIn("final_eval", res)
        self.assertGreater(res["final_eval"]["accuracy"], 0.0)
        self.assertIn("comm_stats", res)


if __name__ == "__main__":
    unittest.main()
