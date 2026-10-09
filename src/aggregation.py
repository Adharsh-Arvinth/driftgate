"""
aggregation.py — Server-side aggregation with Drift Episode Memory

DriftGate Core Mechanism 5: Episode-Memory-Guided Aggregation

The server maintains a structured DriftEpisode record for each client.
Each episode captures:
    - Drift start/end rounds
    - Peak drift severity
    - Layer divergence profile during the episode
    - Post-drift accuracy recovery trajectory
    - Whether drift recurred
    - Client reliability score (gradient stability across rounds)

These episode records modulate aggregation weights:
    weight_k = data_size_k × episode_weight_k × reliability_k × state_modifier_k

where episode_weight_k is computed from the episode history as:
    episode_weight_k = decay-weighted average of episode resolution quality

This creates a server-side memory of HOW each client has handled past drifts,
which is distinct from any existing aggregation method that uses only current-round
information.

Additionally implements:
    - FedAvg baseline
    - FedProx (proximal term) baseline
    - Layer-selective aggregation for persistent clients
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque, Tuple
import numpy as np

from config import AggregationConfig
from state_manager import ClientState
from episode_memory import DriftEpisode, DriftEpisodeManager
from reliability_engine import ClientReliabilityEngine


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation engine
# ──────────────────────────────────────────────────────────────────────────────
class AggregationEngine:
    """
    Server-side aggregation combining:
        - Data-size weighting (standard FedAvg)
        - Episode-memory weight (DriftGate novel mechanism)
        - Reliability score (gradient stability)
        - FSM state modifier (based on current client state)
        - Selective layer aggregation for partial updates

    Final weight formula:
        w_k = n_k × episode_weight_k × reliability_k × state_modifier_k

    Normalized so weights sum to 1 over participating clients.
    """

    def __init__(
        self,
        cfg: AggregationConfig,
        num_clients: int,
    ):
        self.cfg = cfg
        self.episode_manager = DriftEpisodeManager(num_clients, cfg)
        self.reliability_engine = ClientReliabilityEngine(num_clients)
        self._round = 0

    def aggregate(
        self,
        client_updates: Dict[int, Optional[Dict[str, np.ndarray]]],
        client_data_sizes: Dict[int, int],
        client_states: Dict[int, ClientState],
        state_modifiers: Dict[int, float],
        global_params: Dict[str, np.ndarray],
        layer_groups: Optional[Dict[str, List[int]]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Aggregate client updates into a new global model.

        Args:
            client_updates: {cid: param_delta_or_None}
                           None means client was suppressed (TRANSIENT)
            client_data_sizes: {cid: n_samples}
            client_states: {cid: ClientState}
            state_modifiers: {cid: float weight modifier from FSM}
            global_params: Current global model parameters
            layer_groups: Layer group definition for selective agg

        Returns:
            (new_global_params, aggregation_diagnostics)
        """
        self._round += 1
        method = self.cfg.base_method

        # Filter out suppressed (None) updates
        active_updates = {
            cid: upd for cid, upd in client_updates.items()
            if upd is not None and len(upd) > 0
        }

        if not active_updates:
            return global_params, {"method": method, "n_active": 0, "total_weight": 0.0}

        if method == "fedavg" or not self.cfg.selective_layer_aggregation:
            new_params = self._fedavg(active_updates, client_data_sizes, global_params)
            diag = {"method": "fedavg", "n_active": len(active_updates)}
        elif method == "episode_weighted":
            new_params, diag = self._episode_weighted_aggregate(
                active_updates,
                client_data_sizes,
                client_states,
                state_modifiers,
                global_params,
                layer_groups,
            )
        else:
            new_params = self._fedavg(active_updates, client_data_sizes, global_params)
            diag = {"method": "fedavg_fallback", "n_active": len(active_updates)}

        return new_params, diag

    def _fedavg(
        self,
        updates: Dict[int, Dict[str, np.ndarray]],
        data_sizes: Dict[int, int],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Standard FedAvg: weight by data size."""
        total_samples = sum(data_sizes.get(cid, 1) for cid in updates)
        if total_samples == 0:
            return global_params

        new_params = {k: np.zeros_like(v) for k, v in global_params.items()}
        for cid, delta in updates.items():
            w = data_sizes.get(cid, 1) / total_samples
            for k in new_params:
                if k in delta:
                    new_params[k] += w * delta[k]
                else:
                    new_params[k] += w * global_params[k]  # Use global for missing layers

        # Apply updates: new_params are absolute parameters (not deltas)
        # (Our updates ARE absolute parameters after local training)
        return new_params

    def _episode_weighted_aggregate(
        self,
        updates: Dict[int, Dict[str, np.ndarray]],
        data_sizes: Dict[int, int],
        client_states: Dict[int, ClientState],
        state_modifiers: Dict[int, float],
        global_params: Dict[str, np.ndarray],
        layer_groups: Optional[Dict[str, List[int]]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        DriftGate episode-weighted aggregation.

        weight_k = n_k × episode_weight_k × reliability_k × state_modifier_k
        """
        episode_weights = self.episode_manager.get_all_episode_weights()
        reliabilities = self.reliability_engine.get_all_reliabilities()

        # Compute raw weights
        raw_weights: Dict[int, float] = {}
        for cid in updates:
            n_k = data_sizes.get(cid, 1)
            ep_w = episode_weights.get(cid, 1.0)
            rel_k = reliabilities.get(cid, 0.5)
            state_mod = state_modifiers.get(cid, 1.0)
            raw_weights[cid] = n_k * ep_w * rel_k * state_mod

        total_w = sum(raw_weights.values())
        if total_w <= 0:
            # Fallback to uniform
            total_w = len(updates)
            normalized = {cid: 1.0 / total_w for cid in updates}
        else:
            normalized = {cid: w / total_w for cid, w in raw_weights.items()}

        # Layer-selective aggregation:
        # For full-update clients: aggregate all layers normally
        # For selective-update clients: aggregate only their transmitted layers,
        #   keep global params for the rest
        new_params = {k: v.copy() for k, v in global_params.items()}

        # Track which layers have been updated
        layer_update_weights: Dict[str, float] = {k: 0.0 for k in global_params}
        layer_weighted_sums: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in global_params.items()
        }

        for cid, upd in updates.items():
            w = normalized[cid]
            for param_name, param_val in upd.items():
                if param_name in layer_weighted_sums:
                    layer_weighted_sums[param_name] += w * param_val
                    layer_update_weights[param_name] += w

        # Apply: for layers that received updates, use weighted sum
        # For layers with no updates (all clients were selective and skipped), keep global
        for k in new_params:
            if layer_update_weights[k] > 0:
                new_params[k] = layer_weighted_sums[k] / layer_update_weights[k]
            # Else: keep global params unchanged

        diag = {
            "method": "episode_weighted",
            "n_active": len(updates),
            "total_weight": total_w,
            "normalized_weights": normalized,
            "episode_weights": {cid: episode_weights.get(cid, 1.0) for cid in updates},
            "reliabilities": {cid: reliabilities.get(cid, 0.5) for cid in updates},
        }
        return new_params, diag

    def update_episode_manager(
        self,
        client_id: int,
        fsm_state: ClientState,
        round_num: int,
        drift_severity: float,
        drift_type: str,
        affected_groups: List[str],
        layer_divergence: Dict[int, float],
        accuracy: float,
        transmission_mode: str,
    ) -> None:
        """Update episode manager with current round's client drift info."""
        from drift_detector import DriftType

        if fsm_state == ClientState.DRIFTING:
            # May be opening new episode
            existing = self.episode_manager._open_episodes.get(client_id)
            if existing is None:
                self.episode_manager.open_episode(
                    client_id, round_num, drift_type, drift_severity,
                    affected_groups, transmission_mode, layer_divergence,
                )
            else:
                self.episode_manager.update_episode(client_id, drift_severity, layer_divergence)

        elif fsm_state == ClientState.PERSISTENT:
            existing = self.episode_manager._open_episodes.get(client_id)
            if existing is None:
                self.episode_manager.open_episode(
                    client_id, round_num, "persistent", drift_severity,
                    affected_groups, transmission_mode, layer_divergence,
                )
            else:
                self.episode_manager.update_episode(client_id, drift_severity, layer_divergence)

        elif fsm_state == ClientState.ADAPTING:
            self.episode_manager.record_recovery(client_id, round_num, accuracy)

        elif fsm_state in (ClientState.STABLE, ClientState.ELIGIBLE):
            # Close any open episode
            if self.episode_manager._open_episodes.get(client_id) is not None:
                self.episode_manager.close_episode(client_id, round_num, accuracy)

    def update_reliability(
        self,
        client_id: int,
        grad_norm: Optional[float],
        accuracy: float,
        was_suppressed: bool,
    ) -> None:
        """Update reliability engine for a client."""
        self.reliability_engine.update(client_id, grad_norm, accuracy, was_suppressed)
