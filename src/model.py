"""
model.py — Neural network model definitions for DriftGate

Provides:
  - MLP (Multi-Layer Perceptron) for tabular/sensor data
  - Simple CNN for image data (MNIST)
  - Parameter serialization / deserialization utilities
  - Layer-group parameter extraction for selective transmission
"""

from typing import Dict, List, Optional, Tuple, OrderedDict as OrderedDictType
import numpy as np
import copy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available; using NumPy-based model.")

from config import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# NumPy-based MLP (fallback when PyTorch unavailable)
# ──────────────────────────────────────────────────────────────────────────────
class NumpyMLP:
    """
    Lightweight NumPy MLP with SGD for CPU-only environments.
    Used when PyTorch is unavailable.
    Supports softmax output, cross-entropy loss, and basic backprop.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        n_classes: int,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        rng = np.random.RandomState(seed)

        # Initialize weights using Xavier initialization
        dims = [input_dim] + hidden_dims + [n_classes]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            W = rng.randn(dims[i], dims[i + 1]) * scale
            b = np.zeros(dims[i + 1])
            self.weights.append(W.astype(np.float32))
            self.biases.append(b.astype(np.float32))

        self._layer_names = [f"layer_{i}" for i in range(len(self.weights))]
        self._activations: List[np.ndarray] = []
        self._pre_activations: List[np.ndarray] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. Returns logits (pre-softmax)."""
        self._activations = [X]
        self._pre_activations = []
        h = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            self._pre_activations.append(z)
            if i < len(self.weights) - 1:
                h = np.maximum(0, z)  # ReLU
            else:
                h = z  # Logits
            self._activations.append(h)
        return h  # (N, n_classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X)
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _cross_entropy(self, probs: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        eps = 1e-15
        return -np.mean(np.log(probs[np.arange(n), y] + eps))

    def backward(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """
        Single SGD step. Returns (loss, weight_grads, bias_grads).
        Also returns layer gradient norms for drift tracking.
        """
        n = len(y)
        logits = self.forward(X)
        probs = self._softmax(logits)
        loss = self._cross_entropy(probs, y)

        # Output gradient
        d_out = probs.copy()
        d_out[np.arange(n), y] -= 1
        d_out /= n

        weight_grads = []
        bias_grads = []

        delta = d_out
        for i in range(len(self.weights) - 1, -1, -1):
            h = self._activations[i]
            dW = h.T @ delta
            db = delta.sum(axis=0)
            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)

            if i > 0:
                delta = delta @ self.weights[i].T
                # ReLU gradient
                delta *= (self._pre_activations[i - 1] > 0).astype(np.float32)

        # Update parameters
        for i in range(len(self.weights)):
            self.weights[i] -= lr * weight_grads[i]
            self.biases[i] -= lr * bias_grads[i]

        return loss, weight_grads, bias_grads

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return named parameter dict."""
        params = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            params[f"W_{i}"] = W.copy()
            params[f"b_{i}"] = b.copy()
        return params

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        for i in range(len(self.weights)):
            if f"W_{i}" in params:
                self.weights[i] = params[f"W_{i}"].copy()
            if f"b_{i}" in params:
                self.biases[i] = params[f"b_{i}"].copy()

    def get_layer_indices(self) -> Dict[str, List[int]]:
        """
        Return layer group indices for selective transmission.
        Layer index i corresponds to (W_i, b_i).
        """
        n = len(self.weights)
        if n <= 2:
            return {"base": list(range(n // 2 or 1)), "head": list(range(n // 2, n))}
        base_end = n // 2
        return {"base": list(range(base_end)), "head": list(range(base_end, n))}

    def get_layer_grad_norms(self, weight_grads: List[np.ndarray]) -> Dict[int, float]:
        """Return L2 norm of gradient for each layer."""
        return {i: float(np.linalg.norm(g)) for i, g in enumerate(weight_grads)}

    def extract_layer_params(self, layer_indices: List[int]) -> Dict[str, np.ndarray]:
        """Extract parameters for specified layer indices only."""
        result = {}
        for i in layer_indices:
            if i < len(self.weights):
                result[f"W_{i}"] = self.weights[i].copy()
                result[f"b_{i}"] = self.biases[i].copy()
        return result

    def apply_layer_update(self, layer_params: Dict[str, np.ndarray]) -> None:
        """Apply partial parameter update (selective aggregation)."""
        for key, val in layer_params.items():
            if key.startswith("W_"):
                i = int(key[2:])
                if i < len(self.weights):
                    self.weights[i] = val.copy()
            elif key.startswith("b_"):
                i = int(key[2:])
                if i < len(self.biases):
                    self.biases[i] = val.copy()

    def count_parameters(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def clone(self) -> "NumpyMLP":
        new_model = NumpyMLP(self.input_dim, self.hidden_dims, self.n_classes)
        new_model.set_parameters(self.get_parameters())
        return new_model


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch MLP (preferred when available)
# ──────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class TorchMLP(nn.Module):
        """
        PyTorch MLP for federated learning.
        Supports named layer groups for selective transmission.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            n_classes: int,
            dropout: float = 0.2,
        ):
            super().__init__()
            layers = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, n_classes))
            self.network = nn.Sequential(*layers)
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.network(x)

        def get_parameters(self) -> Dict[str, np.ndarray]:
            return {k: v.cpu().numpy().copy() for k, v in self.state_dict().items()}

        def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
            state = {k: torch.tensor(v) for k, v in params.items()}
            self.load_state_dict(state, strict=False)

        def get_named_layer_groups(self, cfg_groups: Dict[str, List[int]]) -> Dict[str, List[str]]:
            """Map integer layer indices to actual parameter names."""
            param_names = list(self.state_dict().keys())
            result = {}
            for group_name, indices in cfg_groups.items():
                group_params = []
                for i in indices:
                    # Each linear layer has .weight and .bias
                    for suffix in [".weight", ".bias"]:
                        # Layer i corresponds to the i-th Linear in the network
                        linear_count = 0
                        for name in param_names:
                            if "weight" in name or "bias" in name:
                                layer_num = int(name.split(".")[1]) // 3  # step of 3 (Linear, ReLU, Dropout)
                                if layer_num == i and name.endswith(suffix.lstrip(".")):
                                    group_params.append(name)
                result[group_name] = group_params
            return result

        def get_layer_grad_norms(self) -> Dict[int, float]:
            """Return gradient L2 norms per linear layer."""
            norms = {}
            linear_idx = 0
            for module in self.network:
                if isinstance(module, nn.Linear):
                    if module.weight.grad is not None:
                        norms[linear_idx] = float(module.weight.grad.norm().item())
                    else:
                        norms[linear_idx] = 0.0
                    linear_idx += 1
            return norms

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────
def create_model(
    input_dim: int,
    n_classes: int,
    cfg: ModelConfig,
    use_torch: bool = True,
) -> "NumpyMLP | TorchMLP":
    """Create and return a model based on config."""
    if use_torch and TORCH_AVAILABLE:
        return TorchMLP(
            input_dim=input_dim,
            hidden_dims=cfg.hidden_dims,
            n_classes=n_classes,
            dropout=cfg.dropout,
        )
    return NumpyMLP(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        n_classes=n_classes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Parameter utilities
# ──────────────────────────────────────────────────────────────────────────────
def params_to_flat(params: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten all parameter tensors to a single 1D array."""
    return np.concatenate([v.flatten() for v in params.values()])


def flat_to_params(flat: np.ndarray, template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Unflatten a 1D array back to a parameter dict matching template shapes."""
    result = {}
    offset = 0
    for k, v in template.items():
        n = v.size
        result[k] = flat[offset:offset + n].reshape(v.shape)
        offset += n
    return result


def compute_update(
    old_params: Dict[str, np.ndarray],
    new_params: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute parameter delta (update = new - old)."""
    return {k: new_params[k] - old_params[k] for k in old_params}


def apply_update(
    params: Dict[str, np.ndarray],
    update: Dict[str, np.ndarray],
    scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Apply a parameter update: params + scale * update."""
    return {k: params[k] + scale * update.get(k, np.zeros_like(params[k])) for k in params}


def count_update_parameters(update: Dict[str, np.ndarray]) -> int:
    """Count non-zero parameters in an update."""
    return sum(int(np.count_nonzero(v)) for v in update.values())


def total_update_parameters(update: Dict[str, np.ndarray]) -> int:
    """Count total parameters in an update."""
    return sum(v.size for v in update.values())
