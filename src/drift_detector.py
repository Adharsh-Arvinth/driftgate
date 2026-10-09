"""
drift_detector.py — Client-side drift detection and transience classification

DriftGate Core Mechanism 1: Drift Transience Classifier

A client detects drift using statistical tests on local loss sequences,
then classifies the drift as TRANSIENT or PERSISTENT using a sliding-window
pattern analysis. This classification drives the client's FSM state, which
in turn determines whether transmission is suppressed (transient) or selective
(persistent).

Supported detectors:
  - ADWIN (sliding window with adaptive size)
  - Page-Hinkley test
  - DDM (Drift Detection Method)
  - Ensemble (majority vote)

Transience classification:
  - Monitor drift signal over `transient_window` rounds
  - If signal reverts before persistence_threshold confidence → TRANSIENT
  - If signal sustains → PERSISTENT
"""

from collections import deque
from typing import Deque, List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np

from config import DriftDetectorConfig


# ──────────────────────────────────────────────────────────────────────────────
# Drift signal enum
# ──────────────────────────────────────────────────────────────────────────────
class DriftSignal(Enum):
    STABLE = "stable"
    WARNING = "warning"
    DRIFT = "drift"


class DriftType(Enum):
    NONE = "none"
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    RECURRING = "recurring"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# ADWIN: Adaptive Windowing
# ──────────────────────────────────────────────────────────────────────────────
class ADWIN:
    """
    Lightweight ADWIN implementation for loss sequence monitoring.
    Based on Bifet & Gavalda (2007).
    Detects change by comparing two sub-windows of varying size.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window: List[float] = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self._drift_detected = False

    def update(self, value: float) -> bool:
        """Add a value, return True if drift detected."""
        self._drift_detected = False
        self.window.append(value)
        self.total += value
        self.width += 1

        # Check if a cut-point in the window causes a significant mean difference
        if self.width >= 5:
            self._drift_detected = self._detect()

        # Prune window if drift found
        if self._drift_detected:
            # Keep only the second half
            cut = self.width // 2
            removed = self.window[:cut]
            self.window = self.window[cut:]
            self.total = sum(self.window)
            self.width = len(self.window)

        return self._drift_detected

    def _detect(self) -> bool:
        n = self.width
        if n < 4:
            return False
        total = self.total
        mu_all = total / n

        # Test multiple cut points
        n0 = 0
        s0 = 0.0
        best_cut_stat = 0.0

        for i in range(1, n - 1):
            s0 += self.window[i - 1]
            n0 += 1
            n1 = n - n0
            mu0 = s0 / n0
            mu1 = (total - s0) / n1

            # Hoeffding bound
            epsilon_cut = np.sqrt(
                (1.0 / (2 * n0) + 1.0 / (2 * n1)) * np.log(4 * n / self.delta)
            )
            cut_stat = abs(mu0 - mu1)
            if cut_stat > epsilon_cut:
                best_cut_stat = max(best_cut_stat, cut_stat - epsilon_cut)

        return best_cut_stat > 0.0

    @property
    def is_change_detected(self) -> bool:
        return self._drift_detected


# ──────────────────────────────────────────────────────────────────────────────
# Page-Hinkley Test
# ──────────────────────────────────────────────────────────────────────────────
class PageHinkley:
    """
    Page-Hinkley test for monitoring monotonic increases in mean.
    Detects abrupt increases in loss (model degradation).
    """

    def __init__(self, threshold: float = 50.0, alpha: float = 0.005, delta: float = 0.005):
        self.threshold = threshold
        self.alpha = alpha       # Magnitude of acceptable change
        self.delta = delta       # Small confidence level
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self._mean = 0.0
        self._drift_detected = False

    def update(self, value: float) -> bool:
        self._drift_detected = False
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        if self._sum - self._min_sum > self.threshold:
            self._drift_detected = True
            # Reset
            self._sum = 0.0
            self._min_sum = 0.0
            self._n = 0
            self._mean = 0.0

        return self._drift_detected

    @property
    def is_change_detected(self) -> bool:
        return self._drift_detected


# ──────────────────────────────────────────────────────────────────────────────
# DDM: Drift Detection Method (Gama et al., 2004)
# ──────────────────────────────────────────────────────────────────────────────
class DDM:
    """
    Drift Detection Method monitoring error rate p with standard deviation.
    WARNING level at p + 2σ; DRIFT level at p + 3σ.
    """

    def __init__(self, min_num_instances: int = 3, warning_level: float = 2.0, out_control_level: float = 3.0):
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self._n = 0
        self._p = 1.0
        self._s = 0.0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._drift_detected = False
        self._warning_detected = False

    def update(self, error: float) -> Tuple[bool, bool]:
        """
        Args:
            error: 0 (correct) or 1 (incorrect)
        Returns:
            (drift_detected, warning_detected)
        """
        self._drift_detected = False
        self._warning_detected = False
        self._n += 1
        self._p += (error - self._p) / self._n
        self._s = np.sqrt(self._p * (1 - self._p) / self._n)

        if self._n < self.min_instances:
            return False, False

        if self._p + self._s <= self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = self._s

        if self._p + self._s > self._p_min + self.out_control_level * self._s_min:
            self._drift_detected = True
            self._n = 0
            self._p = 1.0
            self._s = 0.0
            self._p_min = float("inf")
            self._s_min = float("inf")
        elif self._p + self._s > self._p_min + self.warning_level * self._s_min:
            self._warning_detected = True

        return self._drift_detected, self._warning_detected

    @property
    def is_change_detected(self) -> bool:
        return self._drift_detected


# ──────────────────────────────────────────────────────────────────────────────
# Distribution-level drift measures
# ──────────────────────────────────────────────────────────────────────────────
def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Earth-mover distance (Wasserstein-1) for 1D distributions."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n = min(len(a_sorted), len(b_sorted))
    if n == 0:
        return 0.0
    a_sorted = a_sorted[:n]
    b_sorted = b_sorted[:n]
    return float(np.mean(np.abs(a_sorted - b_sorted)))


def label_distribution_drift(y_old: np.ndarray, y_new: np.ndarray, n_classes: int) -> float:
    """JS divergence between label distributions in two batches."""
    def label_freq(y):
        freq = np.zeros(n_classes)
        for c in range(n_classes):
            freq[c] = np.sum(y == c)
        return freq

    p = label_freq(y_old)
    q = label_freq(y_new)
    return js_divergence(p, q)


# ──────────────────────────────────────────────────────────────────────────────
# DriftGate Core: Transience Classifier
# ──────────────────────────────────────────────────────────────────────────────
class TransienceClassifier:
    """
    DriftGate Core Mechanism 1:
    Classifies detected drift as TRANSIENT or PERSISTENT.

    Algorithm:
        1. Maintain a drift-signal history of length transient_window
        2. When drift is detected, begin recording the signal
        3. After transient_window rounds:
           - If signal sustained for ≥ persistence_threshold fraction → PERSISTENT
           - Else → TRANSIENT
        4. Also check for recurring pattern: if drift occurred, reverted, and
           re-occurs within a period → mark as RECURRING

    State memory for server episode reporting is also computed here.
    """

    def __init__(
        self,
        transient_window: int = 5,
        persistence_threshold: float = 0.7,
        recurring_window: int = 20,
    ):
        self.transient_window = transient_window
        self.persistence_threshold = persistence_threshold
        self.recurring_window = recurring_window

        self._signal_history: Deque[bool] = deque(maxlen=transient_window)
        self._full_history: List[bool] = []
        self._classification_start: Optional[int] = None
        self._classified: DriftType = DriftType.NONE
        self._sustained_count: int = 0
        self._revert_count: int = 0

    def observe(self, drift_signal: bool, round_num: int) -> DriftType:
        """
        Observe a drift signal for the current round.
        Returns current classification of the ongoing drift.
        """
        self._signal_history.append(drift_signal)
        self._full_history.append(drift_signal)

        if drift_signal and self._classification_start is None:
            # Drift onset: start classification window
            self._classification_start = round_num
            self._sustained_count = 0
            self._revert_count = 0
            self._classified = DriftType.UNKNOWN

        if self._classification_start is not None:
            elapsed = round_num - self._classification_start + 1

            if drift_signal:
                self._sustained_count += 1
            else:
                self._revert_count += 1

            if elapsed >= self.transient_window:
                sustained_ratio = self._sustained_count / elapsed
                if sustained_ratio >= self.persistence_threshold:
                    self._classified = DriftType.PERSISTENT
                elif self._revert_count > 0 and sustained_ratio < self.persistence_threshold:
                    self._classified = DriftType.TRANSIENT
                    # Reset for next detection
                    self._classification_start = None
                    self._sustained_count = 0
                    self._revert_count = 0

            # Check for recurring
            if not drift_signal and self._classified == DriftType.PERSISTENT:
                # Drift reverted after persistent → could be recurring
                if len(self._full_history) >= self.recurring_window:
                    # Count how many times drift appeared in the full history
                    n_peaks = sum(
                        1 for i in range(1, len(self._full_history))
                        if self._full_history[i] and not self._full_history[i - 1]
                    )
                    if n_peaks >= 2:
                        self._classified = DriftType.RECURRING

        return self._classified

    def reset(self) -> None:
        """Reset after recovery."""
        self._classification_start = None
        self._classified = DriftType.NONE
        self._sustained_count = 0
        self._revert_count = 0

    @property
    def current_classification(self) -> DriftType:
        return self._classified

    def get_drift_confidence(self) -> float:
        """Return confidence score for current classification (0–1)."""
        if len(self._signal_history) == 0:
            return 0.0
        return float(sum(self._signal_history)) / len(self._signal_history)


# ──────────────────────────────────────────────────────────────────────────────
# Layer-level gradient divergence tracker
# ──────────────────────────────────────────────────────────────────────────────
class LayerDivergenceTracker:
    """
    DriftGate Core Mechanism 2:
    Tracks gradient norms per layer group to identify which layers
    have drifted. This drives selective layer transmission.

    Layer groups: {"base": [...], "head": [...]}
    A layer is considered "drift-affected" if its gradient norm ratio
    (current / baseline) exceeds layer_divergence_threshold.
    """

    def __init__(
        self,
        layer_groups: Dict[str, List[int]],
        divergence_threshold: float = 0.3,
        window_size: int = 5,
    ):
        self.layer_groups = layer_groups
        self.divergence_threshold = divergence_threshold
        # Per-layer rolling average gradient norms
        self._norm_history: Dict[int, Deque[float]] = {}
        self._baseline_norms: Dict[int, float] = {}
        self._window = window_size

        all_layers = [l for layers in layer_groups.values() for l in layers]
        for l in all_layers:
            self._norm_history[l] = deque(maxlen=window_size)
            self._baseline_norms[l] = None

    def update(self, layer_gradients: Dict[int, float]) -> None:
        """
        Update with current-round gradient norms per layer.

        Args:
            layer_gradients: {layer_idx: gradient_norm}
        """
        for layer_idx, norm in layer_gradients.items():
            if layer_idx not in self._norm_history:
                self._norm_history[layer_idx] = deque(maxlen=self._window)
                self._baseline_norms[layer_idx] = None
            self._norm_history[layer_idx].append(norm)
            # Establish baseline from first 5 rounds
            if self._baseline_norms[layer_idx] is None and len(self._norm_history[layer_idx]) >= 3:
                self._baseline_norms[layer_idx] = np.mean(list(self._norm_history[layer_idx]))

    def get_drift_affected_layers(self) -> List[int]:
        """
        Return list of layer indices whose gradient norm ratio exceeds threshold.
        These are the layers that should be transmitted.
        """
        affected = []
        for layer_idx, history in self._norm_history.items():
            if len(history) == 0 or self._baseline_norms.get(layer_idx) is None:
                affected.append(layer_idx)  # Unknown → include
                continue
            baseline = self._baseline_norms[layer_idx]
            current = np.mean(list(history))
            if baseline > 0:
                ratio = abs(current - baseline) / (baseline + 1e-8)
                if ratio > self.divergence_threshold:
                    affected.append(layer_idx)

        return sorted(set(affected))

    def get_drift_affected_groups(self) -> List[str]:
        """Return names of layer groups containing any drift-affected layer."""
        affected_layers = set(self.get_drift_affected_layers())
        affected_groups = []
        for group_name, layers in self.layer_groups.items():
            if any(l in affected_layers for l in layers):
                affected_groups.append(group_name)
        return affected_groups

    def get_layer_divergence_scores(self) -> Dict[int, float]:
        """Return per-layer divergence ratio."""
        scores = {}
        for layer_idx, history in self._norm_history.items():
            if len(history) == 0:
                scores[layer_idx] = 0.0
                continue
            baseline = self._baseline_norms.get(layer_idx)
            if baseline is None or baseline == 0:
                scores[layer_idx] = 0.0
                continue
            current = np.mean(list(history))
            scores[layer_idx] = float(abs(current - baseline) / (baseline + 1e-8))
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble drift detector (combines ADWIN + PH + DDM)
# ──────────────────────────────────────────────────────────────────────────────
class EnsembleDriftDetector:
    """
    Combines ADWIN, Page-Hinkley, and DDM via majority vote.
    Returns a DriftSignal and associated metrics.
    """

    def __init__(self, cfg: DriftDetectorConfig):
        self.cfg = cfg
        self.adwin = ADWIN(delta=cfg.adwin_delta)
        self.ph = PageHinkley(threshold=cfg.ph_threshold, alpha=cfg.ph_alpha)
        self.ddm = DDM()
        self._round = 0
        self._loss_history: List[float] = []

    def update(self, loss: float, error_rate: float) -> Tuple[DriftSignal, Dict[str, Any]]:
        """
        Args:
            loss: Current local training/validation loss
            error_rate: Fraction of incorrect predictions on local val set

        Returns:
            (DriftSignal, diagnostics_dict)
        """
        self._round += 1
        self._loss_history.append(loss)

        adwin_drift = self.adwin.update(loss)
        ph_drift = self.ph.update(loss)
        ddm_drift, ddm_warn = self.ddm.update(error_rate)

        # Statistical loss spike detection relative to recent moving average
        recent_window = self._loss_history[-8:-1]
        loss_spike = False
        if len(recent_window) >= 2:
            baseline_loss = float(np.mean(recent_window))
            if loss > (baseline_loss * 1.35) or (loss - baseline_loss) > 0.30:
                loss_spike = True

        n_drift = sum([adwin_drift, ph_drift, ddm_drift, loss_spike])
        n_warn = ddm_warn or loss_spike

        if n_drift >= 2 or (loss_spike and (adwin_drift or ph_drift or ddm_drift or error_rate >= 0.40)):
            signal = DriftSignal.DRIFT
        elif n_drift >= 1 or n_warn:
            signal = DriftSignal.WARNING
        else:
            signal = DriftSignal.STABLE

        diagnostics = {
            "adwin_drift": adwin_drift,
            "ph_drift": ph_drift,
            "ddm_drift": ddm_drift,
            "loss_spike": loss_spike,
            "ddm_warning": ddm_warn,
            "ensemble_score": n_drift / 4.0,
            "signal": signal.value,
            "loss": loss,
            "error_rate": error_rate,
        }
        return signal, diagnostics


# ──────────────────────────────────────────────────────────────────────────────
# Main per-client drift detector (combines all mechanisms)
# ──────────────────────────────────────────────────────────────────────────────
class ClientDriftDetector:
    """
    Full drift detector for a single client.
    Integrates:
        - EnsembleDriftDetector for raw drift signal
        - TransienceClassifier for TRANSIENT/PERSISTENT labeling
        - LayerDivergenceTracker for layer-selective transmission decisions
    """

    def __init__(
        self,
        client_id: int,
        cfg: DriftDetectorConfig,
        layer_groups: Dict[str, List[int]],
        n_classes: int = 6,
    ):
        self.client_id = client_id
        self.cfg = cfg
        self.n_classes = n_classes
        self.ensemble = EnsembleDriftDetector(cfg)
        self.transience = TransienceClassifier(
            transient_window=cfg.transient_window,
            persistence_threshold=cfg.persistence_threshold,
            recurring_window=20,
        )
        self.layer_tracker = LayerDivergenceTracker(
            layer_groups=layer_groups,
            divergence_threshold=cfg.divergence_threshold,
            window_size=cfg.window_size,
        )
        self._history: List[Dict] = []
        self._current_round: int = 0
        self._prev_label_dist: Optional[np.ndarray] = None

    def detect(
        self,
        loss: float,
        error_rate: float,
        layer_grad_norms: Dict[int, float],
        y_batch: Optional[np.ndarray] = None,
        round_num: int = 0,
    ) -> Dict[str, Any]:
        """
        Run full drift detection pipeline for the current round.

        Returns:
            result dict with:
                "drift_signal": DriftSignal enum
                "drift_type": DriftType enum
                "drift_confidence": float
                "affected_layers": List[int]
                "affected_groups": List[str]
                "label_drift": float (if y_batch provided)
                "diagnostics": dict
        """
        self._current_round = round_num

        # Step 1: Ensemble signal
        signal, diag = self.ensemble.update(loss, error_rate)

        # Step 2: Layer divergence
        self.layer_tracker.update(layer_grad_norms)
        affected_layers = self.layer_tracker.get_drift_affected_layers()
        affected_groups = self.layer_tracker.get_drift_affected_groups()

        # Step 3: Label distribution drift
        label_drift = 0.0
        if y_batch is not None:
            new_dist = np.bincount(y_batch, minlength=self.n_classes).astype(float)
            if self._prev_label_dist is not None:
                label_drift = js_divergence(self._prev_label_dist, new_dist)
            self._prev_label_dist = new_dist
            diag["label_drift_js"] = label_drift

        # Step 4: Transience classification
        is_drifting = (signal == DriftSignal.DRIFT) or (signal == DriftSignal.WARNING and diag.get("loss_spike", False))
        if label_drift > self.cfg.divergence_threshold:
            is_drifting = True  # Label shift reinforces drift signal
        drift_type = self.transience.observe(is_drifting, round_num)
        drift_confidence = self.transience.get_drift_confidence()

        if is_drifting and signal == DriftSignal.STABLE:
            signal = DriftSignal.DRIFT

        result = {
            "drift_signal": signal,
            "drift_type": drift_type,
            "drift_confidence": drift_confidence,
            "affected_layers": affected_layers,
            "affected_groups": affected_groups,
            "label_drift": label_drift,
            "layer_divergence_scores": self.layer_tracker.get_layer_divergence_scores(),
            "diagnostics": diag,
            "round": round_num,
        }
        self._history.append(result)
        return result

    def get_history(self) -> List[Dict]:
        return self._history

    def reset_transience(self) -> None:
        self.transience.reset()
