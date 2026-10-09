"""
data.py — Data loading, partitioning, and drift injection for DriftGate

Supports:
  - UCI HAR (Human Activity Recognition) dataset
  - Synthetic non-stationary sensor dataset
  - MNIST (secondary sanity check)

Client partitioning: Dirichlet-based non-IID split
Drift injection: Gradual, abrupt, recurring, heterogeneous, mixed scenarios
"""

import os
import random
import warnings
import urllib.request
import zipfile
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from config import DataConfig, DriftConfig, GLOBAL_SEED

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Utility: Set all random seeds for reproducibility
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic non-stationary IoT sensor dataset
# ──────────────────────────────────────────────────────────────────────────────
class SyntheticSensorDataset:
    """
    Generates synthetic multi-class sensor data with time-varying distributions.
    Each class cluster moves in feature space over time.
    """

    def __init__(self, cfg: DataConfig, drift_cfg: DriftConfig):
        self.cfg = cfg
        self.drift_cfg = drift_cfg
        set_seed(cfg.seed)
        self._base_means: Optional[np.ndarray] = None
        self._base_covs: Optional[np.ndarray] = None

    def _make_base_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create base class means and covariances."""
        n_classes = self.cfg.n_classes
        n_features = self.cfg.n_features
        rng = np.random.RandomState(self.cfg.seed)
        means = rng.randn(n_classes, n_features) * 3
        covs = []
        for _ in range(n_classes):
            A = rng.randn(n_features, n_features) * 0.3
            cov = A @ A.T + np.eye(n_features) * 0.5
            covs.append(cov)
        return means, np.array(covs)

    def generate_samples(
        self,
        n_samples: int,
        drift_fraction: float = 0.0,
        drift_direction: Optional[np.ndarray] = None,
        client_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples with optional drift applied.

        Args:
            n_samples: Total samples to generate
            drift_fraction: 0=no drift, 1=full drift
            drift_direction: Direction vector for drift (n_classes, n_features)
            client_id: Client identifier (for class subset selection)

        Returns:
            X: (n_samples, n_features)
            y: (n_samples,)
        """
        if self._base_means is None:
            self._base_means, self._base_covs = self._make_base_distribution()

        rng = np.random.RandomState(self.cfg.seed + client_id + 100)
        n_classes = self.cfg.n_classes
        n_features = self.cfg.n_features

        # Client-specific class subset (non-IID): emphasize certain classes
        class_weights = np.ones(n_classes)
        dominant_classes = rng.choice(n_classes, size=max(2, n_classes // 3), replace=False)
        class_weights[dominant_classes] = 4.0
        class_weights /= class_weights.sum()

        samples_per_class = rng.multinomial(n_samples, class_weights)

        X_list, y_list = [], []
        for c in range(n_classes):
            n_c = samples_per_class[c]
            if n_c == 0:
                continue
            mean = self._base_means[c].copy()
            cov = self._base_covs[c]

            # Apply drift shift
            if drift_fraction > 0:
                if drift_direction is None:
                    shift = rng.randn(n_features) * drift_fraction * self.drift_cfg.drift_severity * 2
                else:
                    shift = drift_direction[c] * drift_fraction * self.drift_cfg.drift_severity
                mean = mean + shift

            X_c = rng.multivariate_normal(mean, cov, size=n_c)
            y_c = np.full(n_c, c)
            X_list.append(X_c)
            y_list.append(y_c)

        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list).astype(np.int64)

        # Real concept drift: shift conditional distribution P(Y|X)
        if drift_fraction > 0.2:
            concept_shift_prob = float(np.clip(drift_fraction * self.drift_cfg.drift_severity * 0.8, 0.1, 0.85))
            shift_mask = rng.rand(len(y)) < concept_shift_prob
            class_offset = 1 + (client_id % (max(1, n_classes - 1)))
            y[shift_mask] = (y[shift_mask] + class_offset) % n_classes

        # Shuffle within this batch
        idx = rng.permutation(len(y))
        return X[idx], y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# HAR Dataset Loader
# ──────────────────────────────────────────────────────────────────────────────
class HARDataset:
    """
    Loads the UCI Human Activity Recognition (HAR) dataset.
    Downloads automatically if not present.
    URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
    """

    HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    EXPECTED_FEATURES = 561
    N_CLASSES = 6

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.har_dir = os.path.join(data_dir, "UCI HAR Dataset")
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

    def download_if_needed(self) -> bool:
        """Download and extract HAR dataset if not present. Returns True if available."""
        if os.path.isdir(self.har_dir):
            return True
        zip_path = os.path.join(self.data_dir, "har.zip")
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            print("Downloading UCI HAR dataset...")
            urllib.request.urlretrieve(self.HAR_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.data_dir)
            os.remove(zip_path)
            print("UCI HAR dataset downloaded and extracted.")
            return True
        except Exception as e:
            print(f"Could not download HAR dataset: {e}")
            print("Falling back to synthetic dataset.")
            return False

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns X_train, y_train, X_test, y_test.
        Labels are 0-indexed (original HAR uses 1–6).
        """
        if self._X_train is not None:
            return self._X_train, self._y_train, self._X_test, self._y_test

        def _read_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
            X_path = os.path.join(self.har_dir, split, f"X_{split}.txt")
            y_path = os.path.join(self.har_dir, split, f"y_{split}.txt")
            X = np.loadtxt(X_path, dtype=np.float32)
            y = np.loadtxt(y_path, dtype=np.int64) - 1  # 0-indexed
            return X, y

        self._X_train, self._y_train = _read_split("train")
        self._X_test, self._y_test = _read_split("test")

        # Normalize
        scaler = StandardScaler()
        self._X_train = scaler.fit_transform(self._X_train).astype(np.float32)
        self._X_test = scaler.transform(self._X_test).astype(np.float32)

        return self._X_train, self._y_train, self._X_test, self._y_test


# ──────────────────────────────────────────────────────────────────────────────
# Dirichlet-based non-IID client partitioning
# ──────────────────────────────────────────────────────────────────────────────
def dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = GLOBAL_SEED,
) -> List[np.ndarray]:
    """
    Partition dataset indices into num_clients non-IID subsets using
    a Dirichlet distribution over class labels.

    Args:
        y: Label array (N,)
        num_clients: Number of clients
        alpha: Dirichlet concentration; lower = more non-IID
        seed: Random seed

    Returns:
        List of index arrays, one per client
    """
    rng = np.random.RandomState(seed)
    n_classes = int(y.max()) + 1
    class_indices = [np.where(y == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(class_indices[c])

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        proportions = (proportions * len(class_indices[c])).astype(int)
        # Adjust for rounding
        proportions[-1] = len(class_indices[c]) - proportions[:-1].sum()
        proportions = np.maximum(proportions, 0)

        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_indices[i].extend(class_indices[c][start:end].tolist())
            start = end

    return [np.array(idx, dtype=np.int64) for idx in client_indices]


# ──────────────────────────────────────────────────────────────────────────────
# Client data holder
# ──────────────────────────────────────────────────────────────────────────────
class ClientDataset:
    """Holds data for a single federated client."""

    def __init__(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
        cfg: DataConfig,
    ):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.cfg = cfg
        self._rng = np.random.RandomState(cfg.seed + client_id)

    def get_batch(
        self,
        batch_size: int,
        round_num: int,
        drift_X: Optional[np.ndarray] = None,
        drift_y: Optional[np.ndarray] = None,
        drift_fraction: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch for this client. If drift data is provided and
        drift_fraction > 0, blend original and drifted data.
        """
        n = len(self.X)
        indices = self._rng.choice(n, size=min(batch_size, n), replace=False)
        X_orig = self.X[indices]
        y_orig = self.y[indices]

        if drift_X is not None and drift_fraction > 0.0 and len(drift_X) > 0:
            n_drift = min(len(drift_X), max(1, int(round(batch_size * drift_fraction))))
            n_orig = max(0, batch_size - n_drift)

            if n_orig == 0:
                drift_idx = self._rng.choice(len(drift_X), size=min(batch_size, len(drift_X)), replace=False)
                X_batch = drift_X[drift_idx]
                y_batch = drift_y[drift_idx]
            else:
                drift_idx = self._rng.choice(len(drift_X), size=n_drift, replace=False)
                orig_idx = self._rng.choice(len(X_orig), size=min(n_orig, len(X_orig)), replace=False)
                X_batch = np.vstack([X_orig[orig_idx], drift_X[drift_idx]])
                y_batch = np.concatenate([y_orig[orig_idx], drift_y[drift_idx]])
        else:
            X_batch, y_batch = X_orig, y_orig

        return X_batch.astype(np.float32), y_batch.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)


# ──────────────────────────────────────────────────────────────────────────────
# Drift engine: inject drift scenarios into client data
# ──────────────────────────────────────────────────────────────────────────────
class DriftEngine:
    """
    Manages drift injection across clients and rounds.

    Drift types:
        GRADUAL:        Slowly increasing drift_fraction over several rounds
        ABRUPT:         Instantaneous full drift at a specified round
        RECURRING:      Periodic abrupt drift that reverts and re-occurs
        HETEROGENEOUS:  Subset of clients drift; others remain stable
        MIXED:          Each client gets a randomly assigned drift type
    """

    def __init__(self, cfg: DriftConfig, num_clients: int, seed: int = GLOBAL_SEED):
        self.cfg = cfg
        self.num_clients = num_clients
        self.rng = np.random.RandomState(seed)
        self._drift_schedule = self._build_schedule()

    def _build_schedule(self) -> Dict[int, Dict[str, Any]]:
        """
        Build per-client drift schedule.

        Returns:
            dict mapping client_id → {
                "type": "gradual"|"abrupt"|"recurring"|"none",
                "start_round": int,
                "duration": int,  (0 = infinite)
                "direction": np.ndarray (n_features,)  -- not set here; set per-call
            }
        """
        schedule = {}
        scenario = self.cfg.drift_scenario

        for cid in range(self.num_clients):
            if scenario == "gradual":
                drift_type = "gradual"
                start_round = self.cfg.global_drift_round
                duration = 20
            elif scenario == "abrupt":
                drift_type = "abrupt"
                start_round = self.cfg.global_drift_round
                duration = 0
            elif scenario == "recurring":
                drift_type = "recurring"
                start_round = self.cfg.global_drift_round
                duration = 0
            elif scenario == "heterogeneous":
                drifting = self.rng.rand() < self.cfg.drifting_client_fraction
                drift_type = "abrupt" if drifting else "none"
                start_round = self.cfg.global_drift_round + self.rng.randint(0, 5)
                duration = 0
            else:  # "mixed"
                choices = ["gradual", "abrupt", "recurring", "none"]
                weights = [0.3, 0.35, 0.2, 0.15]
                drift_type = self.rng.choice(choices, p=weights)
                start_round = self.cfg.global_drift_round + self.rng.randint(-5, 10)
                duration = self.cfg.transient_drift_duration if self.rng.rand() < 0.4 else 0

            schedule[cid] = {
                "type": drift_type,
                "start_round": max(0, start_round),
                "duration": duration,
                "is_transient": duration > 0,
            }

        return schedule

    def get_drift_fraction(self, client_id: int, round_num: int) -> float:
        """
        Return drift fraction [0, 1] for a client at a given round.
        0 = no drift; 1 = full drift.
        """
        s = self._drift_schedule.get(client_id, {"type": "none"})
        dt = s["type"]
        start = s.get("start_round", 0)
        duration = s.get("duration", 0)

        if dt == "none" or round_num < start:
            return 0.0

        if dt == "abrupt":
            if duration > 0 and round_num >= start + duration:
                return 0.0  # Reverted → transient
            return 1.0

        elif dt == "gradual":
            grad_window = 15
            progress = (round_num - start) / grad_window
            if duration > 0 and round_num >= start + duration:
                return 0.0
            return float(np.clip(progress, 0.0, 1.0))

        elif dt == "recurring":
            period = self.cfg.recurring_period
            phase = (round_num - start) % period
            if phase < period // 2:
                return 1.0
            return 0.0

        return 0.0

    def is_client_drifting(self, client_id: int, round_num: int) -> bool:
        return self.get_drift_fraction(client_id, round_num) > 0.0

    def is_drift_transient(self, client_id: int) -> bool:
        return self._drift_schedule.get(client_id, {}).get("is_transient", False)

    def get_schedule(self) -> Dict:
        return self._drift_schedule


# ──────────────────────────────────────────────────────────────────────────────
# Federation data loader: create all client datasets
# ──────────────────────────────────────────────────────────────────────────────
class FederationDataLoader:
    """
    Loads and partitions data for the federated simulation.

    Provides:
        - Per-client base datasets
        - Drift engine for round-by-round drift injection
        - Global test set for evaluation
    """

    def __init__(self, data_cfg: DataConfig, drift_cfg: DriftConfig):
        self.data_cfg = data_cfg
        self.drift_cfg = drift_cfg
        set_seed(data_cfg.seed)
        self.clients: List[ClientDataset] = []
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.drift_engine: Optional[DriftEngine] = None
        self._synthetic_gen: Optional[SyntheticSensorDataset] = None
        self._setup()

    def _setup(self) -> None:
        """Load data, partition among clients, build drift engine."""
        dataset = self.data_cfg.dataset
        num_clients = self.data_cfg.num_clients
        alpha = self.data_cfg.dirichlet_alpha

        if dataset == "HAR":
            har = HARDataset(self.data_cfg.data_dir)
            available = har.download_if_needed()
            if available:
                X_train, y_train, X_test, y_test = har.load()
            else:
                print("HAR unavailable; switching to synthetic dataset.")
                dataset = "synthetic"

        if dataset == "synthetic" or self.data_cfg.dataset == "synthetic":
            self._synthetic_gen = SyntheticSensorDataset(self.data_cfg, self.drift_cfg)
            n_total = num_clients * self.data_cfg.samples_per_client_per_round * 10
            X_train, y_train = self._synthetic_gen.generate_samples(n_total)
            X_test, y_test = self._synthetic_gen.generate_samples(n_total // 5)

        if dataset == "MNIST":
            X_train, y_train, X_test, y_test = self._load_mnist()

        # Partition using Dirichlet
        client_idx_lists = dirichlet_partition(y_train, num_clients, alpha, self.data_cfg.seed)

        for cid, idx in enumerate(client_idx_lists):
            if len(idx) == 0:
                # Fallback: give this client some samples
                idx = np.arange(min(50, len(X_train)))
            X_c = X_train[idx]
            y_c = y_train[idx]
            self.clients.append(ClientDataset(cid, X_c, y_c, self.data_cfg))

        self.X_test = X_test
        self.y_test = y_test
        self.drift_engine = DriftEngine(self.drift_cfg, num_clients, self.data_cfg.seed)

        print(f"Federation setup: {num_clients} clients, "
              f"avg {np.mean([len(c) for c in self.clients]):.0f} samples/client, "
              f"test size: {len(X_test)}")

    def _load_mnist(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MNIST as flat numpy arrays."""
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
            X = mnist.data.astype(np.float32) / 255.0
            y = mnist.target.astype(np.int64)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.data_cfg.seed, stratify=y
            )
            return X_train, y_train, X_test, y_test
        except Exception as e:
            print(f"Could not load MNIST: {e}; using synthetic.")
            return self._synthetic_fallback()

    def _synthetic_fallback(self):
        gen = SyntheticSensorDataset(self.data_cfg, self.drift_cfg)
        n = self.data_cfg.num_clients * self.data_cfg.samples_per_client_per_round * 8
        return (*gen.generate_samples(n), *gen.generate_samples(n // 4))

    def get_drift_samples(
        self,
        client_id: int,
        round_num: int,
        batch_size: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Return (drift_X, drift_y, drift_fraction) for a client at given round.
        If no drift → (None, None, 0.0)
        """
        frac = self.drift_engine.get_drift_fraction(client_id, round_num)
        if frac <= 0.0:
            return None, None, 0.0

        if self._synthetic_gen is not None:
            # Generate drifted samples using a direction based on client_id seed
            drift_X, drift_y = self._synthetic_gen.generate_samples(
                batch_size,
                drift_fraction=frac,
                client_id=client_id,
            )
            return drift_X, drift_y, frac

        # For real datasets (e.g. HAR), apply feature shift and concept permutation
        if client_id < len(self.clients):
            client_data = self.clients[client_id]
            raw_X, raw_y = client_data.X, client_data.y
            rng = np.random.RandomState(round_num * 1000 + client_id)
            idx = rng.choice(len(raw_X), size=min(batch_size, len(raw_X)), replace=False)
            drift_X = raw_X[idx].copy()
            drift_y = raw_y[idx].copy()

            # Covariate feature shift
            shift = np.sin(np.arange(drift_X.shape[1]) + client_id) * self.drift_cfg.drift_severity * frac * 2.0
            drift_X += shift.astype(np.float32)
            # Concept drift (label permutation for high drift fraction)
            if frac > 0.5:
                drift_y = (drift_y + 1) % self.n_classes
            return drift_X, drift_y, frac

        return None, None, 0.0

    @property
    def n_features(self) -> int:
        if self.clients:
            return self.clients[0].X.shape[1]
        return self.data_cfg.n_features

    @property
    def n_classes(self) -> int:
        if self.y_test is not None:
            return int(self.y_test.max()) + 1
        return self.data_cfg.n_classes
