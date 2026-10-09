"""
client.py — Federated learning client implementation

Each client:
1. Receives the global model from the server
2. Detects drift in local data using the EnsembleDriftDetector + TransienceClassifier
3. Reports drift signal and transience classification to FSM
4. Trains locally for `local_epochs`
5. Computes per-layer gradient norms for layer divergence tracking
6. Returns the trained parameters + drift report to the federation manager
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import copy

from config import Config
from data import ClientDataset, FederationDataLoader
from drift_detector import ClientDriftDetector, DriftSignal, DriftType
from model import NumpyMLP, create_model

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Client result data structure
# ──────────────────────────────────────────────────────────────────────────────
class ClientResult:
    """Result from a client training round."""

    def __init__(
        self,
        client_id: int,
        trained_params: Optional[Dict[str, np.ndarray]],
        drift_signal: DriftSignal,
        drift_type: DriftType,
        drift_confidence: float,
        drift_severity: float,
        affected_layer_groups: List[str],
        layer_grad_norms: Dict[int, float],
        layer_divergence_scores: Dict[int, float],
        local_accuracy: float,
        local_loss: float,
        local_error_rate: float,
        n_samples: int,
        y_batch: Optional[np.ndarray] = None,
        label_drift: float = 0.0,
    ):
        self.client_id = client_id
        self.trained_params = trained_params
        self.drift_signal = drift_signal
        self.drift_type = drift_type
        self.drift_confidence = drift_confidence
        self.drift_severity = drift_severity
        self.affected_layer_groups = affected_layer_groups
        self.layer_grad_norms = layer_grad_norms
        self.layer_divergence_scores = layer_divergence_scores
        self.local_accuracy = local_accuracy
        self.local_loss = local_loss
        self.local_error_rate = local_error_rate
        self.n_samples = n_samples
        self.y_batch = y_batch
        self.label_drift = label_drift


# ──────────────────────────────────────────────────────────────────────────────
# Federated client
# ──────────────────────────────────────────────────────────────────────────────
class FederatedClient:
    """
    A single federated learning client.

    Maintains:
        - Local dataset
        - Local model (copy of global model)
        - Drift detector (ensemble + transience classifier + layer tracker)
        - Local training history
    """

    def __init__(
        self,
        client_id: int,
        dataset: ClientDataset,
        cfg: Config,
        n_features: int,
        n_classes: int,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.cfg = cfg
        self.n_features = n_features
        self.n_classes = n_classes

        # Drift detector
        self.drift_detector = ClientDriftDetector(
            client_id=client_id,
            cfg=cfg.detector,
            layer_groups=cfg.model.layer_groups,
            n_classes=n_classes,
        )

        # Local model (initialized to zeros; will receive global params)
        self._local_model: Optional[NumpyMLP] = None
        self._use_torch = TORCH_AVAILABLE

        # History
        self._accuracy_history: List[float] = []
        self._loss_history: List[float] = []
        self._drift_history: List[Dict] = []

    def set_model_params(self, params: Dict[str, np.ndarray]) -> None:
        """Update local model with global model parameters."""
        if self._local_model is None:
            self._local_model = NumpyMLP(
                input_dim=self.n_features,
                hidden_dims=self.cfg.model.hidden_dims,
                n_classes=self.n_classes,
            )
        self._local_model.set_parameters(params)

    def train_round(
        self,
        global_params: Dict[str, np.ndarray],
        round_num: int,
        drift_X: Optional[np.ndarray] = None,
        drift_y: Optional[np.ndarray] = None,
        drift_fraction: float = 0.0,
    ) -> ClientResult:
        """
        Execute one local training round.

        Args:
            global_params: Current global model parameters
            round_num: Current federation round
            drift_X: Optional drifted data samples
            drift_y: Optional drifted labels
            drift_fraction: Blend ratio for drift [0,1]

        Returns:
            ClientResult with trained params, drift report
        """
        cfg = self.cfg
        batch_size = cfg.training.batch_size
        local_epochs = cfg.training.local_epochs
        lr = cfg.training.local_lr

        # Receive global model
        self.set_model_params(global_params)
        model = self._local_model

        # Get training batch (with optional drift blending)
        X_batch, y_batch = self.dataset.get_batch(
            batch_size=batch_size,
            round_num=round_num,
            drift_X=drift_X,
            drift_y=drift_y,
            drift_fraction=drift_fraction,
        )

        # Pre-training accuracy (for drift detection baseline)
        pre_preds = model.predict(X_batch)
        pre_accuracy = float(np.mean(pre_preds == y_batch))
        pre_error_rate = 1.0 - pre_accuracy

        # Local training for `local_epochs` steps
        all_grad_norms: Dict[int, List[float]] = {}
        train_loss = 0.0

        for epoch in range(local_epochs):
            # Mini-batch SGD
            n = len(X_batch)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_mini = X_batch[start:end]
                y_mini = y_batch[start:end]
                loss, weight_grads, bias_grads = model.backward(X_mini, y_mini, lr=lr)
                train_loss = loss

                # Record per-layer gradient norms
                for i, wg in enumerate(weight_grads):
                    norm = float(np.linalg.norm(wg))
                    all_grad_norms.setdefault(i, []).append(norm)

        # Average gradient norms across mini-batches
        avg_grad_norms: Dict[int, float] = {
            i: float(np.mean(norms)) for i, norms in all_grad_norms.items()
        }

        # Post-training evaluation
        post_preds = model.predict(X_batch)
        post_accuracy = float(np.mean(post_preds == y_batch))
        post_error_rate = 1.0 - post_accuracy

        # Drift detection using post-training loss + error rate
        detection_result = self.drift_detector.detect(
            loss=train_loss,
            error_rate=post_error_rate,
            layer_grad_norms=avg_grad_norms,
            y_batch=y_batch,
            round_num=round_num,
        )

        # Drift severity = ensemble score × (1 + label drift contribution)
        ensemble_score = detection_result["diagnostics"].get("ensemble_score", 0.0)
        label_drift = detection_result.get("label_drift", 0.0)
        drift_severity = float(np.clip(
            ensemble_score + 0.3 * label_drift, 0.0, 1.0
        ))

        # Record histories
        self._accuracy_history.append(post_accuracy)
        self._loss_history.append(train_loss)
        self._drift_history.append(detection_result)

        return ClientResult(
            client_id=self.client_id,
            trained_params=model.get_parameters(),
            drift_signal=detection_result["drift_signal"],
            drift_type=detection_result["drift_type"],
            drift_confidence=detection_result["drift_confidence"],
            drift_severity=drift_severity,
            affected_layer_groups=detection_result["affected_groups"],
            layer_grad_norms=avg_grad_norms,
            layer_divergence_scores=detection_result.get("layer_divergence_scores", {}),
            local_accuracy=post_accuracy,
            local_loss=train_loss,
            local_error_rate=post_error_rate,
            n_samples=len(X_batch),
            y_batch=y_batch,
            label_drift=label_drift,
        )

    def get_accuracy_history(self) -> List[float]:
        return self._accuracy_history

    def get_loss_history(self) -> List[float]:
        return self._loss_history

    def get_drift_history(self) -> List[Dict]:
        return self._drift_history
