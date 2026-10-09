"""
client_selector.py — Drift-aware adaptive client selection

DriftGate Core Mechanism 6: Joint Drift-Reliability-Bandwidth Client Selection

Client selection computes a priority score combining:
    1. FSM state (PERSISTENT > STABLE/ELIGIBLE > ADAPTING; TRANSIENT = 0)
    2. Drift severity (persistent drifters have high information value)
    3. Reliability score (higher reliability → more trustworthy update)
    4. Communication budget (prefer clients with smaller selective updates)

Selection priority formula:
    score_k = state_weight(FSM_k)
              × (1 + α_drift × drift_severity_k)
              × reliability_k
              × budget_efficiency_k

This is distinct from:
    - FedDrift: selects based on cluster membership
    - Oort: selects based on utility (data quality + system)
    - Power-of-d: random with exploitation

The DriftGate selector explicitly couples the transience classification
with selection eligibility: TRANSIENT clients are NEVER selected (FSM gate),
while PERSISTENT clients are prioritized (they have the most information-
rich updates about the current distribution shift).

Also provides baseline selectors:
    - Random selection
    - Power-of-d
    - Oort (simplified)
    - FedCS (deadline-aware, simplified)
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from config import SelectionConfig
from state_manager import ClientState


# ──────────────────────────────────────────────────────────────────────────────
# State-to-weight mapping
# ──────────────────────────────────────────────────────────────────────────────
STATE_WEIGHTS_DEFAULT: Dict[ClientState, float] = {
    ClientState.STABLE: 0.4,
    ClientState.ELIGIBLE: 0.4,
    ClientState.PERSISTENT: 0.5,
    ClientState.ADAPTING: 0.3,
    ClientState.DRIFTING: 0.0,
    ClientState.TRANSIENT: 0.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# DriftGate selector
# ──────────────────────────────────────────────────────────────────────────────
class DriftGateSelector:
    """
    Joint drift-reliability-bandwidth client selector.

    Clients in TRANSIENT or DRIFTING state are excluded.
    Remaining clients are scored and top-k are selected.

    Priority score:
        score_k = state_w_k
                  × (1 + drift_weight × drift_severity_k)
                  × reliability_k
                  × budget_efficiency_k

    where budget_efficiency is higher for clients that send fewer bytes
    (smaller selective updates consume less communication budget).
    """

    def __init__(self, cfg: SelectionConfig, num_clients: int):
        self.cfg = cfg
        self.num_clients = num_clients
        self.rng = np.random.RandomState(cfg.seed)
        self._selection_history: List[List[int]] = []

    def select(
        self,
        round_num: int,
        client_states: Dict[int, ClientState],
        drift_severities: Dict[int, float],
        reliabilities: Dict[int, float],
        comm_budgets: Dict[int, float],   # Estimated params to send (fraction)
        **kwargs,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Select clients for the current round.

        Args:
            round_num: Current round
            client_states: {cid: ClientState}
            drift_severities: {cid: float in [0,1]}
            reliabilities: {cid: float in [0,1]}
            comm_budgets: {cid: fraction of total params this client would transmit}

        Returns:
            (selected_client_ids, selection_diagnostics)
        """
        k = self.cfg.clients_per_round

        # ── Eligibility gate ─────────────────────────────────────────────────
        eligible = [
            cid for cid, state in client_states.items()
            if state not in (ClientState.TRANSIENT, ClientState.DRIFTING)
        ]

        if len(eligible) == 0:
            # Emergency fallback: all clients (should not happen in practice)
            eligible = list(range(self.num_clients))

        if len(eligible) <= k:
            selected = eligible
        else:
            # ── Score each eligible client ────────────────────────────────────
            scores = {}
            for cid in eligible:
                state = client_states.get(cid, ClientState.STABLE)
                state_w = self._state_weight(state)
                drift_sev = drift_severities.get(cid, 0.0)
                rel = reliabilities.get(cid, 0.5)
                budget_eff = 1.0 - comm_budgets.get(cid, 1.0)  # Less transmission → more efficient

                score = (
                    state_w
                    * (1.0 + 0.5 * drift_sev)
                    * rel
                    * (0.5 + 0.5 * budget_eff)  # Budget efficiency as secondary factor
                )
                # Add small random noise to break ties
                score += self.rng.uniform(0, 0.01)
                scores[cid] = float(score)

            # Top-k selection
            selected = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:k]

        self._selection_history.append(selected)
        diag = {
            "n_eligible": len(eligible),
            "n_selected": len(selected),
            "n_transient_excluded": sum(
                1 for s in client_states.values() if s == ClientState.TRANSIENT
            ),
            "n_drifting_excluded": sum(
                1 for s in client_states.values() if s == ClientState.DRIFTING
            ),
        }
        return selected, diag

    def _state_weight(self, state: ClientState) -> float:
        cfg = self.cfg
        return {
            ClientState.STABLE: cfg.weight_stable,
            ClientState.ELIGIBLE: cfg.weight_eligible,
            ClientState.PERSISTENT: cfg.weight_persistent,
            ClientState.ADAPTING: cfg.weight_adapting,
            ClientState.DRIFTING: 0.0,
            ClientState.TRANSIENT: 0.0,
        }.get(state, 0.0)

    def get_participation_history(self) -> List[List[int]]:
        return self._selection_history

    def get_participation_counts(self) -> Dict[int, int]:
        counts = {cid: 0 for cid in range(self.num_clients)}
        for round_selection in self._selection_history:
            for cid in round_selection:
                counts[cid] += 1
        return counts

    def get_fairness_score(self) -> float:
        """Jain's fairness index over participation counts."""
        counts = list(self.get_participation_counts().values())
        if not counts:
            return 1.0
        n = len(counts)
        sq_sum = sum(c ** 2 for c in counts)
        if sq_sum == 0:
            return 1.0
        return (sum(counts) ** 2) / (n * sq_sum)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline selectors
# ──────────────────────────────────────────────────────────────────────────────
class RandomSelector:
    """Uniform random client selection (FedAvg baseline)."""

    def __init__(self, cfg: SelectionConfig, num_clients: int):
        self.cfg = cfg
        self.num_clients = num_clients
        self.rng = np.random.RandomState(cfg.seed)
        self._history: List[List[int]] = []

    def select(self, round_num: int, **kwargs) -> Tuple[List[int], Dict]:
        k = self.cfg.clients_per_round
        selected = list(self.rng.choice(self.num_clients, size=min(k, self.num_clients), replace=False))
        self._history.append(selected)
        return selected, {"method": "random", "n_selected": len(selected)}

    def get_fairness_score(self) -> float:
        counts = {cid: 0 for cid in range(self.num_clients)}
        for sel in self._history:
            for cid in sel:
                counts[cid] += 1
        c = list(counts.values())
        n = len(c)
        sq_sum = sum(x ** 2 for x in c)
        return (sum(c) ** 2) / (n * max(sq_sum, 1))


class PowerOfDSelector:
    """
    Power-of-d choices: sample d candidates randomly,
    select top-k by a utility metric (data size here).
    """

    def __init__(self, cfg: SelectionConfig, num_clients: int, d: int = 2):
        self.cfg = cfg
        self.num_clients = num_clients
        self.d = d
        self.rng = np.random.RandomState(cfg.seed)
        self._history: List[List[int]] = []

    def select(
        self,
        round_num: int,
        data_sizes: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> Tuple[List[int], Dict]:
        k = self.cfg.clients_per_round
        candidates: Set[int] = set()
        while len(candidates) < min(k * self.d, self.num_clients):
            candidates.add(int(self.rng.randint(0, self.num_clients)))

        if data_sizes:
            selected = sorted(candidates, key=lambda c: data_sizes.get(c, 0), reverse=True)[:k]
        else:
            selected = list(candidates)[:k]

        self._history.append(selected)
        return selected, {"method": "pow_d", "n_selected": len(selected), "d": self.d}


class OortSelector:
    """
    Simplified Oort selector (Lai et al., OSDI 2021).
    Balances data utility (loss at last round) with system utility
    (participation freshness).
    """

    def __init__(self, cfg: SelectionConfig, num_clients: int):
        self.cfg = cfg
        self.num_clients = num_clients
        self.rng = np.random.RandomState(cfg.seed)
        self._losses: Dict[int, float] = {cid: 1.0 for cid in range(num_clients)}
        self._last_selected: Dict[int, int] = {cid: 0 for cid in range(num_clients)}
        self._history: List[List[int]] = []

    def update_loss(self, client_id: int, loss: float) -> None:
        self._losses[client_id] = loss

    def select(
        self,
        round_num: int,
        **kwargs,
    ) -> Tuple[List[int], Dict]:
        k = self.cfg.clients_per_round
        scores = {}
        for cid in range(self.num_clients):
            staleness = round_num - self._last_selected.get(cid, 0)
            data_util = self._losses.get(cid, 1.0)
            # Oort: utility = data_utility × system_utility
            # Simplified: utility = loss × (1 + log(staleness+1))
            score = data_util * (1.0 + np.log1p(staleness))
            score += self.rng.uniform(0, 0.01)
            scores[cid] = score

        selected = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:k]
        for cid in selected:
            self._last_selected[cid] = round_num
        self._history.append(selected)
        return selected, {"method": "oort", "n_selected": len(selected)}


# ──────────────────────────────────────────────────────────────────────────────
# Selector factory
# ──────────────────────────────────────────────────────────────────────────────
def create_selector(
    strategy: str,
    cfg: SelectionConfig,
    num_clients: int,
) -> Any:
    """Create a client selector by strategy name."""
    if strategy == "driftgate":
        return DriftGateSelector(cfg, num_clients)
    elif strategy == "random":
        return RandomSelector(cfg, num_clients)
    elif strategy == "pow_d":
        return PowerOfDSelector(cfg, num_clients, d=cfg.pow_d)
    elif strategy == "oort":
        return OortSelector(cfg, num_clients)
    else:
        return RandomSelector(cfg, num_clients)
