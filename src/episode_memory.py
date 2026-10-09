"""
episode_memory.py — Server-Side Multi-Episode Drift Memory for DriftGate

Maintains structured historical records of concept drift episodes experienced by edge clients:
  - Start round, end round, duration
  - Peak drift severity
  - Drift classification (transient, persistent, recurring)
  - Layer divergence profile at peak
  - Post-drift recovery trajectory (accuracy recovery over subsequent rounds)
  - Recurrence status

Computes episode-memory aggregation weights:
  w_ep(k) = decay-weighted average of historical episode resolution qualities,
            penalizing recurring unmitigated oscillations.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Deque
import numpy as np

from config import AggregationConfig


@dataclass
class DriftEpisode:
    """
    Structured record of a single concept drift episode for an edge client.
    """
    client_id: int
    start_round: int
    end_round: Optional[int] = None
    peak_severity: float = 0.0
    drift_type: str = "unknown"          # "transient" | "persistent" | "recurring"
    transmission_mode: str = "full"      # "none" | "selective" | "full"
    affected_layer_groups: List[str] = field(default_factory=list)
    recovery_trajectory: List[Tuple[int, float]] = field(default_factory=list)
    resolution_quality: float = 0.0
    recurred: bool = False
    layer_divergence_at_peak: Dict[int, float] = field(default_factory=dict)

    def is_resolved(self) -> bool:
        return self.end_round is not None

    def duration(self) -> int:
        if self.end_round is None:
            return 0
        return max(1, self.end_round - self.start_round)

    def compute_resolution_quality(self) -> float:
        """
        Resolution quality is defined as the normalized recovery of validation accuracy.
        If accuracy recovered to >= 0.85, quality is high (0.8 - 1.0).
        """
        if not self.recovery_trajectory:
            return 0.5  # Neutral default
        accs = [acc for _, acc in self.recovery_trajectory]
        max_recovered = float(np.max(accs))
        return float(np.clip(max_recovered, 0.1, 1.0))


class DriftEpisodeManager:
    """
    Central server registry managing multi-episode drift history across all edge nodes.
    """

    def __init__(self, num_clients: int, cfg: AggregationConfig):
        self.num_clients = num_clients
        self.cfg = cfg
        
        # Historical closed episodes per client (bounded by episode_memory_depth)
        self._episodes: Dict[int, Deque[DriftEpisode]] = {
            cid: deque(maxlen=cfg.episode_memory_depth)
            for cid in range(num_clients)
        }
        # Currently open / active episode per client
        self._open_episodes: Dict[int, Optional[DriftEpisode]] = {
            cid: None for cid in range(num_clients)
        }

    def open_episode(
        self,
        client_id: int,
        round_num: int,
        drift_type: str,
        severity: float,
        affected_groups: List[str],
        transmission_mode: str,
        layer_divergence: Dict[int, float],
    ) -> None:
        """Register a new drift episode for an edge client."""
        if self._open_episodes[client_id] is not None:
            # Auto-close preceding open episode
            self._close_episode(client_id, round_num, 0.5)

        episode = DriftEpisode(
            client_id=client_id,
            start_round=round_num,
            drift_type=drift_type,
            peak_severity=float(severity),
            transmission_mode=transmission_mode,
            affected_layer_groups=list(affected_groups),
            layer_divergence_at_peak=dict(layer_divergence),
        )
        self._open_episodes[client_id] = episode

    def update_episode(
        self,
        client_id: int,
        severity: float,
        layer_divergence: Dict[int, float],
    ) -> None:
        """Update active episode with current severity observations."""
        ep = self._open_episodes.get(client_id)
        if ep is not None:
            ep.peak_severity = max(ep.peak_severity, float(severity))
            for k, v in layer_divergence.items():
                ep.layer_divergence_at_peak[k] = max(
                    ep.layer_divergence_at_peak.get(k, 0.0), float(v)
                )

    def record_recovery(
        self,
        client_id: int,
        round_num: int,
        accuracy: float,
    ) -> None:
        """Record accuracy checkpoint during recovery phase."""
        ep = self._open_episodes.get(client_id)
        if ep is not None:
            ep.recovery_trajectory.append((round_num, float(accuracy)))

    def close_episode(
        self,
        client_id: int,
        round_num: int,
        accuracy: float,
    ) -> None:
        """Close active episode upon drift resolution or stabilization."""
        self._close_episode(client_id, round_num, accuracy)

    def _close_episode(
        self,
        client_id: int,
        round_num: int,
        accuracy: float,
    ) -> None:
        ep = self._open_episodes.get(client_id)
        if ep is not None:
            ep.end_round = round_num
            ep.recovery_trajectory.append((round_num, float(accuracy)))
            ep.resolution_quality = ep.compute_resolution_quality()
            
            # Check for recurrence against historical episodes
            for past_ep in self._episodes[client_id]:
                if past_ep.drift_type == ep.drift_type and ep.drift_type != "transient":
                    ep.recurred = True
                    past_ep.recurred = True
                    break
                    
            self._episodes[client_id].append(ep)
            self._open_episodes[client_id] = None

    def get_episode_weight(self, client_id: int) -> float:
        """
        Compute historical episode resolution coefficient M_ep(k) in [min_weight_floor, 1.0].
        
        Formula:
          M_ep(k) = (Sum_{i=0}^{N-1} lambda^i * Q_{k,i} * (1 - rho_{k,i})) / (Sum lambda^i)
        """
        episodes = list(self._episodes[client_id])
        if not episodes:
            # Client has clean historical record: neutral high weight
            return 1.0

        decay = self.cfg.episode_recency_decay
        total_weight = 0.0
        total_decay = 0.0

        for i, ep in enumerate(reversed(episodes)):
            recency_factor = decay ** i
            recurrence_penalty = 0.35 if ep.recurred else 0.0
            quality = ep.resolution_quality if ep.is_resolved() else 0.5
            
            episode_score = quality * (1.0 - recurrence_penalty)
            total_weight += recency_factor * episode_score
            total_decay += recency_factor

        avg_weight = total_weight / max(1e-8, total_decay)
        return float(np.clip(avg_weight, self.cfg.min_weight_floor, 1.0))

    def get_all_episode_weights(self) -> Dict[int, float]:
        return {cid: self.get_episode_weight(cid) for cid in range(self.num_clients)}

    def get_open_episode(self, client_id: int) -> Optional[DriftEpisode]:
        return self._open_episodes.get(client_id)

    def get_episode_count(self, client_id: int) -> int:
        return len(self._episodes[client_id])

    def get_summary(self) -> Dict[str, Any]:
        total_closed = sum(len(eps) for eps in self._episodes.values())
        active_open = sum(1 for ep in self._open_episodes.values() if ep is not None)
        return {
            "total_closed_episodes": total_closed,
            "currently_open_episodes": active_open,
            "episode_weights": self.get_all_episode_weights(),
        }
