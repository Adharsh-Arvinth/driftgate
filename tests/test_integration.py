"""
test_integration.py — End-to-end integration tests for DriftGate federation
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import get_fast_config
from federation import FederationRunner, BaselineRunner
from state_manager import ClientState


class TestDriftGateIntegration(unittest.TestCase):

    def test_end_to_end_driftgate_mixed_scenario(self):
        cfg = get_fast_config()
        cfg.data.num_clients = 6
        cfg.training.num_rounds = 8
        cfg.selection.clients_per_round = 3
        cfg.drift.global_drift_round = 3  # Drift onset at round 3
        cfg.drift.drift_scenario = "mixed"
        cfg.drift.drift_severity = 0.85

        runner = FederationRunner(cfg, seed=42)
        res = runner.run()

        # 1. Verification of execution
        self.assertIn("final_eval", res)
        self.assertIn("comm_stats", res)
        self.assertGreater(res["final_eval"]["accuracy"], 0.0)

        # 2. State history verification (FSM transitions must have occurred)
        state_history = res.get("state_history", [])
        self.assertGreater(len(state_history), 0)
        
        # Check that drift states were activated during the run
        any_drifting_or_adaptive = any(
            (r.get("drifting", 0) > 0 or 
             r.get("transient", 0) > 0 or 
             r.get("persistent", 0) > 0 or 
             r.get("adapting", 0) > 0)
            for r in state_history
        )
        self.assertTrue(
            any_drifting_or_adaptive,
            f"FSM must trigger drift or adaptive states during mixed drift run. States: {state_history}"
        )

        # 3. Communication savings verification
        comm_stats = res["comm_stats"]
        self.assertGreater(comm_stats["bandwidth_reduction_pct"], 0.0)

    def test_baseline_runners(self):
        cfg = get_fast_config()
        cfg.data.num_clients = 4
        cfg.training.num_rounds = 2
        cfg.selection.clients_per_round = 2

        # FedAvg
        cfg_avg = BaselineRunner.apply_baseline_config(cfg, "fedavg")
        res_avg = FederationRunner(cfg_avg, seed=42).run()
        self.assertGreater(res_avg["final_eval"]["accuracy"], 0.0)

        # FedProx
        cfg_prox = BaselineRunner.apply_baseline_config(cfg, "fedprox")
        res_prox = FederationRunner(cfg_prox, seed=42).run()
        self.assertGreater(res_prox["final_eval"]["accuracy"], 0.0)

        # Oort
        cfg_oort = BaselineRunner.apply_baseline_config(cfg, "oort")
        res_oort = FederationRunner(cfg_oort, seed=42).run()
        self.assertGreater(res_oort["final_eval"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
