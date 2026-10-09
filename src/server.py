"""
server.py — Federation server implementation for DriftGate

The server orchestrates:
    1. Model initialization and broadcasting
    2. Client selection (using DriftGate or baseline selector)
    3. Receiving client updates and drift reports
    4. FSM state updates for all clients
    5. Communication controller (prepare selective updates)
    6. Episode memory aggregation
    7. Global model update
    8. Metrics collection
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from config import Config
from model import NumpyMLP, create_model
from state_manager import FederationStateManager, ClientState
from aggregation import AggregationEngine
from client_selector import create_selector, DriftGateSelector
from communication_controller import CommunicationController
from client import ClientResult
from drift_detector import DriftSignal, DriftType


class FederationServer:
    """
    Central server for the DriftGate federated learning system.

    Implements the full server-side protocol:
        Round r:
          1. Select K clients using DriftGate selector (FSM-gated)
          2. Broadcast global model to selected clients
          3. Receive ClientResults from clients
          4. Update FSM states for all reporting clients
          5. Apply communication controller (select which params to aggregate)
          6. Update episode manager with drift episode info
          7. Run episode-weighted aggregation
          8. Compute global metrics
    """

    def __init__(
        self,
        cfg: Config,
        n_features: int,
        n_classes: int,
        num_clients: int,
    ):
        self.cfg = cfg
        self.n_features = n_features
        self.n_classes = n_classes
        self.num_clients = num_clients

        # Global model
        self.global_model = NumpyMLP(
            input_dim=n_features,
            hidden_dims=cfg.model.hidden_dims,
            n_classes=n_classes,
        )
        self.global_params = self.global_model.get_parameters()

        # Subsystems
        self.state_manager = FederationStateManager(num_clients, cfg.fsm)
        self.aggregation_engine = AggregationEngine(cfg.aggregation, num_clients)
        self.selector = create_selector(cfg.selection.strategy, cfg.selection, num_clients)
        self.comm_controller = CommunicationController(
            cfg.comm,
            cfg.model,
            total_params=self.global_model.count_parameters(),
        )

        # Per-round metrics
        self._round_metrics: List[Dict] = []
        self._current_round: int = 0

        # Client data sizes (set when first results received)
        self._client_data_sizes: Dict[int, int] = {cid: 1 for cid in range(num_clients)}

        # Drift severity tracking
        self._drift_severities: Dict[int, float] = {cid: 0.0 for cid in range(num_clients)}

    def get_global_params(self) -> Dict[str, np.ndarray]:
        """Return current global model parameters."""
        return {k: v.copy() for k, v in self.global_params.items()}

    def select_clients(self, round_num: int) -> Tuple[List[int], Dict]:
        """Run client selection for this round."""
        # Get current states
        client_states = {
            cid: self.state_manager.get_state(cid)
            for cid in range(self.num_clients)
        }

        # Get reliabilities
        reliabilities = self.aggregation_engine.reliability_engine.get_all_reliabilities()

        # Get comm budgets (fraction of params each client would send)
        # Selective clients send fewer params → lower budget fraction
        comm_budgets = {}
        for cid in range(self.num_clients):
            state = client_states[cid]
            if state == ClientState.PERSISTENT:
                comm_budgets[cid] = self.cfg.comm.topk_fraction * 0.5  # Selective
            elif state == ClientState.TRANSIENT:
                comm_budgets[cid] = 0.0  # Suppressed
            else:
                comm_budgets[cid] = self.cfg.comm.topk_fraction  # Full with compression

        if hasattr(self.selector, "select"):
            selected, sel_diag = self.selector.select(
                round_num=round_num,
                client_states=client_states,
                drift_severities=self._drift_severities,
                reliabilities=reliabilities,
                comm_budgets=comm_budgets,
                data_sizes=self._client_data_sizes,
            )
        else:
            selected = list(range(min(self.cfg.selection.clients_per_round, self.num_clients)))
            sel_diag = {}

        return selected, sel_diag

    def process_round(
        self,
        round_num: int,
        client_results: List[ClientResult],
    ) -> Dict[str, Any]:
        """
        Process all client results for a round.

        Steps:
            1. Update FSM states
            2. Update episode manager
            3. Prepare comm-controlled updates
            4. Run aggregation
            5. Collect metrics
        """
        self._current_round = round_num

        # ── Step 1: Update data sizes ─────────────────────────────────────────
        for result in client_results:
            self._client_data_sizes[result.client_id] = max(1, result.n_samples)
            self._drift_severities[result.client_id] = result.drift_severity

        # ── Step 2: Update FSM states ─────────────────────────────────────────
        for result in client_results:
            cid = result.client_id
            new_state = self.state_manager.update_client(
                client_id=cid,
                drift_signal=result.drift_signal,
                drift_type=result.drift_type,
                drift_confidence=result.drift_confidence,
                local_accuracy=result.local_accuracy,
                round_num=round_num,
            )

        # Advance state/timers for non-reporting clients without injecting artificial stable signals
        reporting_ids = {r.client_id for r in client_results}
        for cid in range(self.num_clients):
            if cid not in reporting_ids:
                self.state_manager.tick_unselected_client(cid, round_num)

        self.state_manager.record_round(round_num)

        # ── Step 3: Update episode manager and reliability ────────────────────
        for result in client_results:
            cid = result.client_id
            fsm_state = self.state_manager.get_state(cid)

            self.aggregation_engine.update_episode_manager(
                client_id=cid,
                fsm_state=fsm_state,
                round_num=round_num,
                drift_severity=result.drift_severity,
                drift_type=result.drift_type.value if hasattr(result.drift_type, "value") else str(result.drift_type),
                affected_groups=result.affected_layer_groups,
                layer_divergence=result.layer_divergence_scores,
                accuracy=result.local_accuracy,
                transmission_mode=self.state_manager.fsms[cid].transmission_mode(),
            )

            grad_norm = np.mean(list(result.layer_grad_norms.values())) if result.layer_grad_norms else None
            was_suppressed = (fsm_state == ClientState.TRANSIENT)
            self.aggregation_engine.update_reliability(
                client_id=cid,
                grad_norm=grad_norm,
                accuracy=result.local_accuracy,
                was_suppressed=was_suppressed,
            )

        # ── Step 4: Communication control ─────────────────────────────────────
        prepared_updates: Dict[int, Optional[Dict[str, np.ndarray]]] = {}
        for result in client_results:
            cid = result.client_id
            fsm_state = self.state_manager.get_state(cid)

            prepared, mode, params_sent = self.comm_controller.prepare_update(
                client_id=cid,
                client_state=fsm_state,
                full_params=result.trained_params or {},
                affected_layer_groups=result.affected_layer_groups,
                layer_groups=self.cfg.model.layer_groups,
                round_num=round_num,
            )
            prepared_updates[cid] = prepared

        # ── Step 5: Episode-weighted aggregation ─────────────────────────────
        client_states_dict = {
            cid: self.state_manager.get_state(cid)
            for cid in range(self.num_clients)
        }
        state_modifiers = {
            cid: self.state_manager.fsms[cid].get_weight_modifier()
            for cid in range(self.num_clients)
        }

        new_global, agg_diag = self.aggregation_engine.aggregate(
            client_updates=prepared_updates,
            client_data_sizes=self._client_data_sizes,
            client_states=client_states_dict,
            state_modifiers=state_modifiers,
            global_params=self.global_params,
            layer_groups=self.cfg.model.layer_groups,
        )

        self.global_params = new_global
        self.global_model.set_parameters(new_global)

        # ── Step 6: Collect round metrics ─────────────────────────────────────
        comm_summary = self.comm_controller.stats.get_round_summary(round_num)
        state_counts = self.state_manager.get_state_counts()

        metrics = {
            "round": round_num,
            "n_clients_reporting": len(client_results),
            "n_active": agg_diag.get("n_active", 0),
            "state_counts": state_counts,
            "comm_summary": comm_summary,
            "aggregation": agg_diag,
            "avg_local_accuracy": float(np.mean([r.local_accuracy for r in client_results])) if client_results else 0.0,
            "avg_drift_severity": float(np.mean([r.drift_severity for r in client_results])) if client_results else 0.0,
        }
        self._round_metrics.append(metrics)
        return metrics

    def evaluate_global_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate global model on the global test set."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        preds = self.global_model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        precision = float(precision_score(y_test, preds, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, preds, average="weighted", zero_division=0))
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def get_round_metrics(self) -> List[Dict]:
        return self._round_metrics

    def get_communication_stats(self) -> Dict:
        return self.comm_controller.stats.get_overall_summary()

    def get_state_history(self) -> List[Dict]:
        return self.state_manager.get_round_history()

    def get_episode_summary(self) -> Dict:
        return self.aggregation_engine.episode_manager.get_summary()

    def get_client_summaries(self) -> List[Dict]:
        return self.state_manager.get_client_summaries()
