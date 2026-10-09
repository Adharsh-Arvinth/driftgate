"""
metrics.py — Metrics computation, storage, and statistical analysis

Computes:
  - Per-round accuracy, F1, precision, recall
  - Communication efficiency metrics
  - Drift detection performance (FP/FN rates)
  - Convergence metrics
  - Fairness metrics
  - Statistical summaries across multiple seeds
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np


def compute_convergence_round(
    accuracy_history: List[float],
    threshold: float = 0.75,
) -> Optional[int]:
    """Return the round at which accuracy first exceeds threshold, or None."""
    for i, acc in enumerate(accuracy_history):
        if acc >= threshold:
            return i + 1
    return None


def compute_accuracy_after_drift(
    accuracy_history: List[float],
    drift_round: int,
    window: int = 5,
) -> float:
    """Mean accuracy in `window` rounds immediately after drift_round."""
    start = min(drift_round, len(accuracy_history))
    end = min(drift_round + window, len(accuracy_history))
    segment = accuracy_history[start:end]
    return float(np.mean(segment)) if segment else 0.0


def compute_recovery_rounds(
    accuracy_history: List[float],
    drift_round: int,
    pre_drift_accuracy: float,
    tolerance: float = 0.02,
) -> Optional[int]:
    """Rounds after drift to return within `tolerance` of pre-drift accuracy."""
    for i in range(drift_round, len(accuracy_history)):
        if accuracy_history[i] >= pre_drift_accuracy - tolerance:
            return i - drift_round
    return None


def compute_drift_detection_metrics(
    detected_rounds: List[int],
    true_drift_rounds: List[int],
    total_rounds: int,
    tolerance: int = 3,
) -> Dict[str, float]:
    """
    Compute FP/FN/TP for drift detection.

    Args:
        detected_rounds: Rounds where drift was flagged
        true_drift_rounds: Ground-truth rounds where drift occurred
        total_rounds: Total number of rounds
        tolerance: ±rounds window for a detection to count as TP
    """
    tp = 0
    fn = 0
    fp = 0

    matched_true = set()
    for d_round in detected_rounds:
        found = False
        for t_round in true_drift_rounds:
            if abs(d_round - t_round) <= tolerance and t_round not in matched_true:
                tp += 1
                matched_true.add(t_round)
                found = True
                break
        if not found:
            fp += 1

    fn = len(true_drift_rounds) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    fpr = fp / max(total_rounds - len(true_drift_rounds), 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,  # False positive rate
    }


def aggregate_multi_seed_results(
    results: List[Dict],
    metric_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across multiple seeds.

    Returns dict of {metric: {mean, std, min, max}}
    """
    aggregated = {}
    for key in metric_keys:
        values = []
        for r in results:
            val = r.get(key)
            if val is not None:
                values.append(float(val))
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }
    return aggregated


def extract_accuracy_curve(round_metrics: List[Dict]) -> Tuple[List[int], List[float]]:
    """Extract (rounds, accuracy) curve from round_metrics list."""
    rounds = [m["round"] for m in round_metrics]
    accs = [m["accuracy"] for m in round_metrics]
    return rounds, accs


def extract_comm_efficiency_curve(round_metrics: List[Dict]) -> Tuple[List[int], List[float]]:
    """Extract (rounds, comm_efficiency) curve."""
    rounds = []
    effs = []
    for m in round_metrics:
        comm = m.get("comm_summary", {})
        eff = comm.get("communication_efficiency", 0.0)
        rounds.append(m["round"])
        effs.append(eff)
    return rounds, effs


def extract_state_history(state_history: List[Dict]) -> Dict[str, List]:
    """Extract per-state count histories for plotting."""
    states = ["stable", "drifting", "transient", "persistent", "adapting", "eligible"]
    result = {s: [] for s in states}
    rounds = []
    for record in state_history:
        rounds.append(record.get("round", 0))
        for s in states:
            result[s].append(record.get(s, 0))
    result["round"] = rounds
    return result


class MetricsCollector:
    """Collects and stores experiment metrics across runs."""

    def __init__(self):
        self._runs: List[Dict] = []

    def add_run(self, run_result: Dict, run_name: str, seed: int) -> None:
        self._runs.append({
            "name": run_name,
            "seed": seed,
            "result": run_result,
        })

    def get_summary_table(self) -> List[Dict]:
        """Return a per-run summary with key metrics."""
        rows = []
        for run in self._runs:
            result = run["result"]
            round_metrics = result.get("round_metrics", [])
            comm_stats = result.get("comm_stats", {})
            final = result.get("final_eval", {})

            accs = [m["accuracy"] for m in round_metrics]
            f1s = [m["f1"] for m in round_metrics]

            rows.append({
                "name": run["name"],
                "seed": run["seed"],
                "final_accuracy": final.get("accuracy", 0.0),
                "final_f1": final.get("f1", 0.0),
                "peak_accuracy": max(accs) if accs else 0.0,
                "mean_accuracy": np.mean(accs) if accs else 0.0,
                "mean_f1": np.mean(f1s) if f1s else 0.0,
                "bandwidth_reduction_pct": comm_stats.get("bandwidth_reduction_pct", 0.0),
                "suppressed_rounds": comm_stats.get("suppressed_client_rounds", 0),
                "selective_rounds": comm_stats.get("selective_client_rounds", 0),
                "total_bytes_sent": comm_stats.get("total_bytes_sent", 0),
                "convergence_round": compute_convergence_round(accs, threshold=0.70),
            })
        return rows

    def get_all_runs(self) -> List[Dict]:
        return self._runs

    def to_dataframe(self):
        """Convert summary to pandas DataFrame (if available)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.get_summary_table())
        except ImportError:
            return self.get_summary_table()
