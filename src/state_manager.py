"""
state_manager.py — Client FSM (Finite State Machine) for DriftGate

DriftGate Core Mechanism 3: Stateful Client Participation Eligibility

Each client has an explicit state machine with states:
    STABLE     → Normal participation; full update transmission
    DRIFTING   → Drift detected; waiting for transience classification
    TRANSIENT  → Drift classified as transient; transmission SUPPRESSED,
                 local adaptation only (no server update)
    PERSISTENT → Drift classified as persistent; SELECTIVE LAYER transmission
                 only (drift-affected layers only)
    ADAPTING   → Post-drift recovery; head-only transmission, monitoring
                 accuracy recovery
    ELIGIBLE   → Fully re-admitted; full update transmission

State transitions:
    STABLE → DRIFTING:    when drift signal detected
    DRIFTING → TRANSIENT: when transience classifier outputs TRANSIENT
    DRIFTING → PERSISTENT: when transience classifier outputs PERSISTENT
    TRANSIENT → STABLE:   after suppression window, if no re-detection
    PERSISTENT → ADAPTING: when drift reverts (signal returns to STABLE)
    ADAPTING → ELIGIBLE:  when accuracy recovery threshold met
    ELIGIBLE → STABLE:    after confirmation round
    ANY → DRIFTING:       when drift re-detected from stable states

The FSM state is reported to the server to drive:
    - Client selection priority
    - Aggregation weight
    - Layer transmission decisions
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from config import ClientFSMConfig
from drift_detector import DriftSignal, DriftType


# ──────────────────────────────────────────────────────────────────────────────
# State definitions
# ──────────────────────────────────────────────────────────────────────────────
class ClientState(Enum):
    STABLE = "stable"
    DRIFTING = "drifting"
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    ADAPTING = "adapting"
    ELIGIBLE = "eligible"


# ──────────────────────────────────────────────────────────────────────────────
# State history entry
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StateTransition:
    from_state: ClientState
    to_state: ClientState
    round_num: int
    reason: str


# ──────────────────────────────────────────────────────────────────────────────
# Per-client FSM
# ──────────────────────────────────────────────────────────────────────────────
class ClientFSM:
    """
    Finite State Machine for a single federated client.

    The FSM determines:
        1. Whether this client should be eligible for selection
        2. What type of update to transmit (full / selective / none)
        3. What aggregation weight modifier applies

    The server reads the reported state to make aggregation and
    selection decisions.
    """

    def __init__(self, client_id: int, cfg: ClientFSMConfig):
        self.client_id = client_id
        self.cfg = cfg
        self.state: ClientState = ClientState.STABLE
        self.rounds_in_state: int = 0
        self.history: List[StateTransition] = []
        self._drift_start_round: Optional[int] = None
        self._adapt_accuracy_history: List[float] = []

    def transition(self, new_state: ClientState, round_num: int, reason: str) -> None:
        if new_state != self.state:
            self.history.append(StateTransition(
                from_state=self.state,
                to_state=new_state,
                round_num=round_num,
                reason=reason,
            ))
            self.state = new_state
            self.rounds_in_state = 0
        else:
            self.rounds_in_state += 1

    def update(
        self,
        drift_signal: DriftSignal,
        drift_type: DriftType,
        drift_confidence: float,
        local_accuracy: float,
        round_num: int,
    ) -> ClientState:
        """
        Update FSM based on drift detection results.

        Args:
            drift_signal: Raw signal from ensemble detector
            drift_type: Classified type (TRANSIENT/PERSISTENT/UNKNOWN/NONE)
            drift_confidence: Confidence score [0,1]
            local_accuracy: Current local validation accuracy
            round_num: Current federation round

        Returns:
            New FSM state
        """
        current = self.state
        cfg = self.cfg

        # ── STABLE ───────────────────────────────────────────────────────────
        if current == ClientState.STABLE:
            if drift_signal in (DriftSignal.DRIFT, DriftSignal.WARNING):
                self.transition(ClientState.DRIFTING, round_num, "drift_detected")
                self._drift_start_round = round_num
                self.rounds_in_state = 0

        # ── ELIGIBLE ─────────────────────────────────────────────────────────
        elif current == ClientState.ELIGIBLE:
            if drift_signal == DriftSignal.DRIFT:
                self.transition(ClientState.DRIFTING, round_num, "drift_re_detected")
                self._drift_start_round = round_num
            else:
                # After one stable round, fully re-admit
                self.transition(ClientState.STABLE, round_num, "confirmed_stable")

        # ── DRIFTING ─────────────────────────────────────────────────────────
        elif current == ClientState.DRIFTING:
            self.rounds_in_state += 1
            if self.rounds_in_state >= cfg.classification_wait_rounds:
                if drift_type == DriftType.TRANSIENT:
                    self.transition(ClientState.TRANSIENT, round_num, "classified_transient")
                elif drift_type in (DriftType.PERSISTENT, DriftType.RECURRING):
                    self.transition(ClientState.PERSISTENT, round_num, "classified_persistent")
                else:
                    # Timeout resolution based on confidence or persistence
                    if drift_confidence >= 0.4 or drift_signal == DriftSignal.DRIFT:
                        self.transition(ClientState.PERSISTENT, round_num, "timeout_assumed_persistent")
                    elif drift_signal == DriftSignal.STABLE:
                        self.transition(ClientState.STABLE, round_num, "self_resolved")
                    else:
                        self.transition(ClientState.TRANSIENT, round_num, "timeout_assumed_transient")

        # ── TRANSIENT ────────────────────────────────────────────────────────
        elif current == ClientState.TRANSIENT:
            self.rounds_in_state += 1
            if drift_signal == DriftSignal.DRIFT and self.rounds_in_state < cfg.transient_suppression_rounds:
                # Transient reclassified as persistent (sustained)
                self.transition(ClientState.PERSISTENT, round_num, "transient_became_persistent")
            elif self.rounds_in_state >= cfg.transient_suppression_rounds:
                # Suppression window elapsed, re-enter as stable
                self.transition(ClientState.STABLE, round_num, "transient_resolved")

        # ── PERSISTENT ───────────────────────────────────────────────────────
        elif current == ClientState.PERSISTENT:
            self.rounds_in_state += 1
            if drift_signal == DriftSignal.STABLE and self.rounds_in_state >= cfg.min_persistent_selective_rounds:
                # Drift resolved → begin recovery
                self.transition(ClientState.ADAPTING, round_num, "persistent_resolved")
                self._adapt_accuracy_history = []

        # ── ADAPTING ─────────────────────────────────────────────────────────
        elif current == ClientState.ADAPTING:
            self.rounds_in_state += 1
            self._adapt_accuracy_history.append(local_accuracy)

            if drift_signal == DriftSignal.DRIFT:
                # Relapse
                self.transition(ClientState.PERSISTENT, round_num, "adapting_relapse")
            elif local_accuracy >= cfg.recovery_accuracy_threshold:
                self.transition(ClientState.ELIGIBLE, round_num, "accuracy_recovered")
            elif self.rounds_in_state >= cfg.max_adapting_rounds:
                # Force re-admit even without full recovery
                self.transition(ClientState.ELIGIBLE, round_num, "adapting_timeout")

        return self.state

    def tick_idle(self, round_num: int) -> None:
        """Advance timers for unselected client without modifying its state abruptly."""
        self.rounds_in_state += 1
        cfg = self.cfg
        if self.state == ClientState.TRANSIENT:
            if self.rounds_in_state >= cfg.transient_suppression_rounds:
                self.transition(ClientState.STABLE, round_num, "transient_timeout_resolved")
        elif self.state == ClientState.ADAPTING:
            if self.rounds_in_state >= cfg.max_adapting_rounds:
                self.transition(ClientState.ELIGIBLE, round_num, "adapting_timeout_eligible")
        elif self.state == ClientState.ELIGIBLE:
            self.transition(ClientState.STABLE, round_num, "eligible_re_admitted")

    def should_transmit(self) -> bool:
        """Return True if client should transmit any update."""
        return self.state not in (ClientState.TRANSIENT, ClientState.DRIFTING)

    def transmission_mode(self) -> str:
        """
        Return transmission mode:
            "full":      all layers
            "selective": only drift-affected layers
            "none":      no transmission
        """
        if self.state in (ClientState.STABLE, ClientState.ELIGIBLE):
            return "full"
        elif self.state == ClientState.PERSISTENT:
            return "selective"
        elif self.state == ClientState.ADAPTING:
            return "selective"  # head-only during adaptation
        else:
            return "none"

    def get_weight_modifier(self) -> float:
        """
        Return aggregation weight modifier based on FSM state.
        Used by the episode-memory aggregation.
        """
        return {
            ClientState.STABLE: 1.0,
            ClientState.ELIGIBLE: 1.0,
            ClientState.ADAPTING: 0.5,
            ClientState.PERSISTENT: 0.7,
            ClientState.DRIFTING: 0.3,
            ClientState.TRANSIENT: 0.0,  # No transmission → 0 weight
        }.get(self.state, 0.5)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "state": self.state.value,
            "rounds_in_state": self.rounds_in_state,
            "n_transitions": len(self.history),
            "transmission_mode": self.transmission_mode(),
            "weight_modifier": self.get_weight_modifier(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Federation-wide FSM registry
# ──────────────────────────────────────────────────────────────────────────────
class FederationStateManager:
    """
    Manages FSM instances for all clients.
    Provides aggregate statistics for server-side decisions.
    """

    def __init__(self, num_clients: int, cfg: ClientFSMConfig):
        self.num_clients = num_clients
        self.cfg = cfg
        self.fsms: Dict[int, ClientFSM] = {
            cid: ClientFSM(cid, cfg) for cid in range(num_clients)
        }
        self._round_history: List[Dict] = []

    def update_client(
        self,
        client_id: int,
        drift_signal: DriftSignal,
        drift_type: DriftType,
        drift_confidence: float,
        local_accuracy: float,
        round_num: int,
    ) -> ClientState:
        """Update a single client's FSM. Returns new state."""
        return self.fsms[client_id].update(
            drift_signal=drift_signal,
            drift_type=drift_type,
            drift_confidence=drift_confidence,
            local_accuracy=local_accuracy,
            round_num=round_num,
        )

    def tick_unselected_client(self, client_id: int, round_num: int) -> None:
        """Advance timers for unselected client without injecting fake stable signals."""
        self.fsms[client_id].tick_idle(round_num)

    def get_state(self, client_id: int) -> ClientState:
        return self.fsms[client_id].state

    def get_eligible_clients(self) -> List[int]:
        """Return client IDs that can be selected for this round."""
        return [
            cid for cid, fsm in self.fsms.items()
            if fsm.state not in (ClientState.TRANSIENT, ClientState.DRIFTING)
        ]

    def get_state_counts(self) -> Dict[str, int]:
        """Return count of clients in each state."""
        counts = {s.value: 0 for s in ClientState}
        for fsm in self.fsms.values():
            counts[fsm.state.value] += 1
        return counts

    def get_client_summaries(self) -> List[Dict]:
        """Return summaries for all clients."""
        return [fsm.get_summary() for fsm in self.fsms.values()]

    def record_round(self, round_num: int) -> None:
        """Record aggregate state snapshot for this round."""
        counts = self.get_state_counts()
        counts["round"] = round_num
        self._round_history.append(counts)

    def get_round_history(self) -> List[Dict]:
        return self._round_history

    def get_transmission_modes(self) -> Dict[int, str]:
        """Return transmission mode for every client."""
        return {cid: fsm.transmission_mode() for cid, fsm in self.fsms.items()}
