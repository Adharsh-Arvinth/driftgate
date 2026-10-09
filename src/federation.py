"""
federation.py — Main federation loop orchestrating clients and server

Runs the complete federated learning simulation:
  1. Initialize data loader, clients, and server
  2. For each round:
     a. Server selects clients
     b. Server broadcasts global model
     c. Selected clients perform local training (with drift injection)
     d. Server processes results (FSM updates, comm control, aggregation)
     e. Server evaluates global model
     f. Metrics logged
  3. Return full experiment results
"""

import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from config import Config, get_default_config
from data import FederationDataLoader
from client import FederatedClient
from server import FederationServer


class FederationRunner:
    """
    Orchestrates the complete DriftGate federated learning simulation.
    """

    def __init__(self, cfg: Config, seed: Optional[int] = None):
        self.cfg = cfg
        self.seed = seed if seed is not None else cfg.experiment.seeds[0]
        self._set_seed(self.seed)

        # Data
        self.data_loader = FederationDataLoader(cfg.data, cfg.drift)
        self.n_features = self.data_loader.n_features
        self.n_classes = self.data_loader.n_classes

        print(f"[FederationRunner] n_features={self.n_features}, n_classes={self.n_classes}")

        # Server
        self.server = FederationServer(
            cfg=cfg,
            n_features=self.n_features,
            n_classes=self.n_classes,
            num_clients=cfg.data.num_clients,
        )

        # Clients
        self.clients: Dict[int, FederatedClient] = {}
        for i, client_data in enumerate(self.data_loader.clients):
            self.clients[i] = FederatedClient(
                client_id=i,
                dataset=client_data,
                cfg=cfg,
                n_features=self.n_features,
                n_classes=self.n_classes,
            )

        # Results storage
        self.results: List[Dict] = []
        self._global_metrics: List[Dict] = []

    def _set_seed(self, seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    def run(self) -> Dict[str, Any]:
        """Run the full federation for cfg.training.num_rounds rounds."""
        cfg = self.cfg
        num_rounds = cfg.training.num_rounds
        drift_engine = self.data_loader.drift_engine
        X_test = self.data_loader.X_test
        y_test = self.data_loader.y_test

        print(f"\n{'='*60}")
        print(f"DriftGate Federation Run")
        print(f"Strategy: {cfg.selection.strategy}")
        print(f"Rounds: {num_rounds} | Clients: {cfg.data.num_clients}")
        print(f"Drift scenario: {cfg.drift.drift_scenario}")
        print(f"{'='*60}\n")

        for round_num in range(1, num_rounds + 1):
            t_start = time.time()

            # ── Client selection ──────────────────────────────────────────────
            selected_ids, sel_diag = self.server.select_clients(round_num)

            # ── Client training ───────────────────────────────────────────────
            global_params = self.server.get_global_params()
            client_results = []

            for cid in selected_ids:
                client = self.clients[cid]
                drift_X, drift_y, drift_frac = self.data_loader.get_drift_samples(
                    client_id=cid,
                    round_num=round_num,
                    batch_size=cfg.data.samples_per_client_per_round,
                )
                result = client.train_round(
                    global_params=global_params,
                    round_num=round_num,
                    drift_X=drift_X,
                    drift_y=drift_y,
                    drift_fraction=drift_frac,
                )
                client_results.append(result)

            # ── Server processing ─────────────────────────────────────────────
            round_metrics = self.server.process_round(round_num, client_results)

            # ── Global evaluation ─────────────────────────────────────────────
            global_eval = self.server.evaluate_global_model(X_test, y_test)

            t_elapsed = time.time() - t_start

            # ── Record metrics ────────────────────────────────────────────────
            round_record = {
                "round": round_num,
                "accuracy": global_eval["accuracy"],
                "f1": global_eval["f1"],
                "precision": global_eval["precision"],
                "recall": global_eval["recall"],
                "n_selected": len(selected_ids),
                "selected_ids": selected_ids,
                "round_time": t_elapsed,
                **round_metrics,
                "sel_diag": sel_diag,
            }
            self._global_metrics.append(round_record)

            # ── Progress logging ──────────────────────────────────────────────
            if round_num % 5 == 0 or round_num == 1:
                state_counts = round_metrics["state_counts"]
                comm = round_metrics.get("comm_summary", {})
                eff = comm.get("communication_efficiency", 0.0)
                print(
                    f"R{round_num:3d} | "
                    f"Acc={global_eval['accuracy']:.3f} | "
                    f"F1={global_eval['f1']:.3f} | "
                    f"Selected={len(selected_ids)} | "
                    f"CommEff={eff:.2f} | "
                    f"Stable={state_counts.get('stable',0)} "
                    f"Trans={state_counts.get('transient',0)} "
                    f"Pers={state_counts.get('persistent',0)} "
                    f"Adapt={state_counts.get('adapting',0)}"
                )

        # ── Final results compilation ─────────────────────────────────────────
        comm_stats = self.server.get_communication_stats()
        final_eval = self.server.evaluate_global_model(X_test, y_test)

        print(f"\n{'='*60}")
        print(f"Run Complete | Final Accuracy: {final_eval['accuracy']:.4f} | F1: {final_eval['f1']:.4f}")
        print(f"Bandwidth Reduction: {comm_stats.get('bandwidth_reduction_pct', 0):.1f}%")
        print(f"{'='*60}\n")

        return {
            "config": cfg.to_dict(),
            "round_metrics": self._global_metrics,
            "final_eval": final_eval,
            "comm_stats": comm_stats,
            "state_history": self.server.get_state_history(),
            "episode_summary": self.server.get_episode_summary(),
            "client_summaries": self.server.get_client_summaries(),
            "seed": self.seed,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline runner (same federation loop with different strategy config)
# ──────────────────────────────────────────────────────────────────────────────
class BaselineRunner(FederationRunner):
    """
    Runs baseline FL algorithms using the same simulation infrastructure.
    Configures the system to behave as the specified baseline.
    """

    BASELINE_CONFIGS = {
        "fedavg": {
            "selection.strategy": "random",
            "aggregation.base_method": "fedavg",
            "comm.compression_method": "none",
            "comm.transient_suppress_transmission": False,
            "comm.persistent_selective_transmission": False,
            "aggregation.selective_layer_aggregation": False,
        },
        "fedprox": {
            "selection.strategy": "random",
            "aggregation.base_method": "fedprox",
            "comm.compression_method": "none",
            "comm.transient_suppress_transmission": False,
            "comm.persistent_selective_transmission": False,
            "aggregation.selective_layer_aggregation": False,
        },
        "feddrift": {
            # FedDrift: random selection + clustering-style aggregation
            # Simplified: uses drift detection but no transience classifier
            "selection.strategy": "random",
            "aggregation.base_method": "fedavg",
            "comm.compression_method": "none",
            "comm.transient_suppress_transmission": False,
            "comm.persistent_selective_transmission": False,
            "aggregation.selective_layer_aggregation": False,
        },
        "oort": {
            "selection.strategy": "oort",
            "aggregation.base_method": "fedavg",
            "comm.compression_method": "none",
            "comm.transient_suppress_transmission": False,
            "comm.persistent_selective_transmission": False,
            "aggregation.selective_layer_aggregation": False,
        },
    }

    @classmethod
    def apply_baseline_config(cls, cfg: Config, baseline_name: str) -> Config:
        """Apply baseline-specific config overrides."""
        overrides = cls.BASELINE_CONFIGS.get(baseline_name, {})
        import copy
        cfg_copy = copy.deepcopy(cfg)

        for path, value in overrides.items():
            parts = path.split(".")
            obj = cfg_copy
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        return cfg_copy
