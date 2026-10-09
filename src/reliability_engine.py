"""
reliability_engine.py — Client and Update Reliability Estimation for DriftGate

Tracks update reliability across communication rounds based on:
  1. Gradient trajectory stability (low coefficient of variation in gradient magnitude)
  2. Validation accuracy stability across local epochs
  3. Participation history (penalizing erratic suppression or frequent drops)
  4. Directional alignment (cosine similarity of updates relative to consensus)
"""

from collections import deque
from typing import Dict, List, Optional, Deque, Any
import numpy as np


class ClientReliabilityEngine:
    """
    Tracks per-client update reliability across federation rounds.
    Used by:
      - DriftGateSelector (scoring eligible clients for participation)
      - AggregationEngine (scaling aggregation coefficients)
    """

    def __init__(self, num_clients: int, window_size: int = 10):
        self.num_clients = num_clients
        self.window_size = window_size
        
        # Per-client histories
        self._grad_norms: Dict[int, Deque[float]] = {
            cid: deque(maxlen=window_size) for cid in range(num_clients)
        }
        self._acc_histories: Dict[int, Deque[float]] = {
            cid: deque(maxlen=window_size) for cid in range(num_clients)
        }
        self._cosine_histories: Dict[int, Deque[float]] = {
            cid: deque(maxlen=window_size) for cid in range(num_clients)
        }
        self._suppressed_counts: Dict[int, int] = {cid: 0 for cid in range(num_clients)}
        self._total_rounds: Dict[int, int] = {cid: 0 for cid in range(num_clients)}

    def update(
        self,
        client_id: int,
        grad_norm: Optional[float],
        accuracy: float,
        was_suppressed: bool,
        cosine_sim: Optional[float] = None,
    ) -> None:
        """
        Record observation for a client round.
        """
        self._total_rounds[client_id] += 1
        
        if was_suppressed:
            self._suppressed_counts[client_id] += 1
            return

        if grad_norm is not None:
            self._grad_norms[client_id].append(float(grad_norm))
            
        self._acc_histories[client_id].append(float(accuracy))
        
        if cosine_sim is not None:
            self._cosine_histories[client_id].append(float(cosine_sim))

    def get_reliability(self, client_id: int) -> float:
        """
        Compute aggregate reliability score R_k in [0.0, 1.0].
        
        Components:
          (a) Gradient stability (40%): low CV indicates stable local optimization
          (b) Participation non-suppression rate (30%): reliable edge availability
          (c) Accuracy consistency (30%): low variance in local accuracy
        """
        norms = list(self._grad_norms.get(client_id, []))
        accs = list(self._acc_histories.get(client_id, []))
        total = max(1, self._total_rounds.get(client_id, 1))
        suppressed = self._suppressed_counts.get(client_id, 0)

        # 1. Gradient stability
        if len(norms) >= 2:
            mean_norm = float(np.mean(norms)) + 1e-8
            cv = float(np.std(norms) / mean_norm)
            grad_stability = float(np.clip(1.0 - cv * 0.5, 0.1, 1.0))
        else:
            grad_stability = 0.5

        # 2. Participation rate
        participation_rate = float(np.clip(1.0 - (suppressed / total), 0.1, 1.0))

        # 3. Accuracy consistency
        if len(accs) >= 2:
            acc_std = float(np.std(accs))
            acc_stability = float(np.clip(1.0 - acc_std * 3.0, 0.1, 1.0))
        else:
            acc_stability = 0.5

        score = (
            0.40 * grad_stability +
            0.30 * participation_rate +
            0.30 * acc_stability
        )
        return float(np.clip(score, 0.05, 1.0))

    def get_all_reliabilities(self) -> Dict[int, float]:
        """Return reliability scores for all clients."""
        return {cid: self.get_reliability(cid) for cid in range(self.num_clients)}

    def reset_client(self, client_id: int) -> None:
        """Clear history for a specific client."""
        self._grad_norms[client_id].clear()
        self._acc_histories[client_id].clear()
        self._cosine_histories[client_id].clear()
        self._suppressed_counts[client_id] = 0
        self._total_rounds[client_id] = 0
