"""
communication_controller.py — Selective layer transmission and compression

DriftGate Core Mechanism 4: State-Driven Selective Transmission

Controls what parameters each client transmits based on:
  1. FSM state (STABLE→full, PERSISTENT→selective, TRANSIENT→none, ADAPTING→head)
  2. Layer divergence profile (which layers are drift-affected)
  3. Communication budget (top-k sparsification)
  4. Round budget accounting

Key insight (DriftGate differentiator):
  - TRANSIENT clients: NO transmission. Local adaptation only.
    They apply local SGD but their update never reaches the server.
    This saves communication AND prevents polluting the global model
    with short-lived noise.
  - PERSISTENT clients: Only drift-affected layer groups transmitted.
    Stable base layers (already well-trained) are NOT transmitted.
    Only the "head" or specific affected layers are sent.
  - STABLE / ELIGIBLE clients: Full transmission (with optional top-k compression).
  - ADAPTING clients: Head-only transmission during recovery phase.

Communication accounting:
  Every transmission is measured in bytes and parameter counts.
  Comparison with a FedAvg baseline (all clients, all params) is tracked.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from config import CommConfig, ModelConfig
from state_manager import ClientState
from compression import topk_sparsify, random_sparsify, quantize_params, compute_compression_stats


# ──────────────────────────────────────────────────────────────────────────────
# Communication accounting
# ──────────────────────────────────────────────────────────────────────────────
class CommStats:
    """
    Tracks communication statistics for a federation run.
    Computes transmitted bytes, parameter counts, and efficiency metrics.
    """

    def __init__(self, bytes_per_float: int = 4):
        self.bytes_per_float = bytes_per_float
        # Per-round stats: {round_num: {client_id: {params_sent, bytes_sent, mode}}}
        self._round_stats: Dict[int, Dict[int, Dict[str, Any]]] = {}
        # Totals
        self.total_params_sent: int = 0
        self.total_bytes_sent: int = 0
        self.total_params_baseline: int = 0  # FedAvg equivalent
        self.total_bytes_baseline: int = 0

        # Suppressed clients (TRANSIENT mode)
        self.total_suppressed_rounds: int = 0
        # Selective-only clients (PERSISTENT/ADAPTING)
        self.total_selective_rounds: int = 0
        # Full clients (STABLE/ELIGIBLE)
        self.total_full_rounds: int = 0

    def record(
        self,
        round_num: int,
        client_id: int,
        params_sent: int,
        total_params: int,
        mode: str,
    ) -> None:
        if round_num not in self._round_stats:
            self._round_stats[round_num] = {}

        bytes_sent = params_sent * self.bytes_per_float
        bytes_total = total_params * self.bytes_per_float

        self._round_stats[round_num][client_id] = {
            "params_sent": params_sent,
            "total_params": total_params,
            "bytes_sent": bytes_sent,
            "mode": mode,
        }

        self.total_params_sent += params_sent
        self.total_bytes_sent += bytes_sent
        self.total_params_baseline += total_params
        self.total_bytes_baseline += bytes_total

        if mode == "none":
            self.total_suppressed_rounds += 1
        elif mode == "selective":
            self.total_selective_rounds += 1
        else:
            self.total_full_rounds += 1

    def get_round_summary(self, round_num: int) -> Dict[str, Any]:
        if round_num not in self._round_stats:
            return {}
        stats = self._round_stats[round_num]
        total_params = sum(s["params_sent"] for s in stats.values())
        total_bytes = sum(s["bytes_sent"] for s in stats.values())
        n_suppressed = sum(1 for s in stats.values() if s["mode"] == "none")
        n_selective = sum(1 for s in stats.values() if s["mode"] == "selective")
        n_full = sum(1 for s in stats.values() if s["mode"] == "full")
        baseline_params = sum(s["total_params"] for s in stats.values())
        efficiency = 1.0 - (total_params / max(1, baseline_params))
        return {
            "round": round_num,
            "total_params_sent": total_params,
            "total_bytes_sent": total_bytes,
            "baseline_params": baseline_params,
            "communication_efficiency": efficiency,
            "n_suppressed": n_suppressed,
            "n_selective": n_selective,
            "n_full": n_full,
            "n_clients": len(stats),
        }

    def get_overall_summary(self) -> Dict[str, Any]:
        total_reduction = 1.0 - (self.total_params_sent / max(1, self.total_params_baseline))
        return {
            "total_params_sent": self.total_params_sent,
            "total_bytes_sent": self.total_bytes_sent,
            "total_params_baseline": self.total_params_baseline,
            "total_bytes_baseline": self.total_bytes_baseline,
            "bandwidth_reduction_pct": total_reduction * 100,
            "suppressed_client_rounds": self.total_suppressed_rounds,
            "selective_client_rounds": self.total_selective_rounds,
            "full_client_rounds": self.total_full_rounds,
        }

    def get_round_stats_list(self) -> List[Dict]:
        """Return per-round summary list sorted by round."""
        return [self.get_round_summary(r) for r in sorted(self._round_stats.keys())]


# ──────────────────────────────────────────────────────────────────────────────
# Communication Controller
# ──────────────────────────────────────────────────────────────────────────────
class CommunicationController:
    """
    DriftGate Core Mechanism 4:
    Controls parameter transmission based on client FSM state.

    For each client update, determines:
      1. Whether to transmit at all (TRANSIENT → skip)
      2. Which layer groups to transmit (PERSISTENT → selective)
      3. How to compress the transmitted parameters (top-k or quantize)
      4. Accounting for exact bytes/params transmitted

    This creates a coupling between drift type classification and
    communication scheduling that is the novel interaction in DriftGate.
    """

    def __init__(
        self,
        cfg: CommConfig,
        model_cfg: ModelConfig,
        total_params: int,
    ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.total_params = total_params
        self.stats = CommStats(bytes_per_float=cfg.bytes_per_float)
        self._rng = np.random.RandomState(42)

    def prepare_update(
        self,
        client_id: int,
        client_state: ClientState,
        full_params: Dict[str, np.ndarray],
        affected_layer_groups: List[str],
        layer_groups: Dict[str, List[int]],
        round_num: int,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], str, int]:
        """
        Prepare the update to be transmitted from client to server.

        Args:
            client_id: Client identifier
            client_state: Current FSM state
            full_params: Complete parameter delta from local training
            affected_layer_groups: Layer groups identified as drift-affected
            layer_groups: Mapping from group name to layer indices
            round_num: Current round

        Returns:
            (prepared_params_or_None, transmission_mode, params_sent_count)
        """
        # ── TRANSIENT: Suppress all transmission ─────────────────────────────
        if client_state == ClientState.TRANSIENT and self.cfg.transient_suppress_transmission:
            self.stats.record(round_num, client_id, 0, self.total_params, "none")
            return None, "none", 0

        # ── DRIFTING: Suppress (classification in progress) ──────────────────
        if client_state == ClientState.DRIFTING:
            self.stats.record(round_num, client_id, 0, self.total_params, "none")
            return None, "none", 0

        # ── PERSISTENT or ADAPTING: Selective layer transmission ─────────────
        if client_state in (ClientState.PERSISTENT, ClientState.ADAPTING) and \
                self.cfg.persistent_selective_transmission:

            # Determine which groups to transmit
            if client_state == ClientState.ADAPTING:
                # Adapting: head only
                groups_to_send = ["head"] if "head" in layer_groups else list(layer_groups.keys())
            else:
                # Persistent: only drift-affected groups
                groups_to_send = affected_layer_groups if affected_layer_groups else list(layer_groups.keys())

            # Extract only parameters from selected groups
            selected_params = self._extract_group_params(full_params, groups_to_send, layer_groups)

            # Apply compression to selected params
            compressed = self._compress(selected_params)
            params_sent = sum(int(np.count_nonzero(v)) for v in compressed.values())

            self.stats.record(round_num, client_id, params_sent, self.total_params, "selective")
            return compressed, "selective", params_sent

        # ── STABLE / ELIGIBLE: Full transmission with compression ─────────────
        compressed = self._compress(full_params)
        params_sent = sum(int(np.count_nonzero(v)) for v in compressed.values())

        self.stats.record(round_num, client_id, params_sent, self.total_params, "full")
        return compressed, "full", params_sent

    def _extract_group_params(
        self,
        full_params: Dict[str, np.ndarray],
        group_names: List[str],
        layer_groups: Dict[str, List[int]],
    ) -> Dict[str, np.ndarray]:
        """Extract parameters belonging to specified layer groups."""
        selected_layer_indices = set()
        for gname in group_names:
            for idx in layer_groups.get(gname, []):
                selected_layer_indices.add(idx)

        result = {}
        for param_name, param_val in full_params.items():
            # param names are like "W_0", "b_0", "W_1", "b_1"
            try:
                layer_idx = int(param_name.split("_")[1])
                if layer_idx in selected_layer_indices:
                    result[param_name] = param_val
            except (IndexError, ValueError):
                # For PyTorch-style names, use substring matching
                for idx in selected_layer_indices:
                    if f".{idx * 3}." in param_name or f".{idx * 3}." in param_name:
                        result[param_name] = param_val
                        break

        # If nothing extracted (name format mismatch), return a subset
        if not result and full_params:
            keys = list(full_params.keys())
            half = max(1, len(keys) // 2)
            result = {k: full_params[k] for k in keys[half:]}

        return result

    def _compress(
        self,
        params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply configured compression method."""
        if not params:
            return params

        method = self.cfg.compression_method
        if method == "topk":
            compressed = topk_sparsify(params, self.cfg.topk_fraction, self._rng)
            return compressed
        elif method == "quantize":
            return quantize_params(params, self.cfg.quantization_bits)
        elif method == "random_sparsification":
            compressed = topk_sparsify(
                params,
                self.cfg.topk_fraction,
                self._rng,
            )
            return compressed
        else:
            return params  # "none" — no compression

    def get_stats(self) -> CommStats:
        return self.stats
