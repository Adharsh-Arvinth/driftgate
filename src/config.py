"""
config.py — Centralized configuration for DriftGate FL System

DriftGate: Drift-Transience-Aware Stateful Federated Learning with
           Layer-Selective Communication and Episode Memory Aggregation

All experiment parameters are defined here. Do NOT hardcode parameters
in other modules; always import from config.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import os


# ──────────────────────────────────────────────────────────────────────────────
# Random seed (global, used for all NumPy / torch / sklearn calls)
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED: int = 42

# ──────────────────────────────────────────────────────────────────────────────
# Dataset configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    dataset: str = "synthetic"          # "synthetic" | "HAR" | "MNIST"
    num_clients: int = 20
    # Non-IID partitioning: Dirichlet concentration parameter
    # Lower alpha → more non-IID
    dirichlet_alpha: float = 0.5
    # Time-series window for HAR
    time_steps: int = 128
    # Number of synthetic features when dataset == "synthetic"
    n_features: int = 32
    n_classes: int = 6
    # Drift window: each client sees at most this many samples per round
    samples_per_client_per_round: int = 128
    # Random seed for data generation / split
    seed: int = GLOBAL_SEED
    data_dir: str = "./data"
    train_fraction: float = 0.8


# ──────────────────────────────────────────────────────────────────────────────
# Drift generation configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DriftConfig:
    # Drift types per experiment run — can be a list: one per scenario
    drift_scenario: str = "mixed"   # "gradual" | "abrupt" | "recurring" | "heterogeneous" | "mixed"

    # Fraction of clients that drift in heterogeneous scenario
    drifting_client_fraction: float = 0.5

    # Round at which global drift starts
    global_drift_round: int = 30

    # Severity of drift (0–1); controls distribution shift magnitude
    drift_severity: float = 0.8

    # Number of rounds a transient drift lasts before reverting
    transient_drift_duration: int = 5

    # Number of rounds a persistent drift lasts (0 = infinite)
    persistent_drift_duration: int = 0

    # Recurring drift period (rounds)
    recurring_period: int = 20

    # Random seed for drift injection
    seed: int = GLOBAL_SEED + 1


# ──────────────────────────────────────────────────────────────────────────────
# Model architecture configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    model_type: str = "mlp"          # "mlp" | "cnn" | "lstm"
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    # Named layer groups for selective transmission (DriftGate mechanism)
    # Format: {"group_name": [layer_indices]}
    # Layer 0..N-1 correspond to parameter blocks
    # "base" layers are pretrained/stable; "head" layers are task-specific
    layer_groups: Dict[str, List[int]] = field(
        default_factory=lambda: {"base": [0, 1], "head": [2, 3]}
    )
    # Threshold: fraction of layer-level gradient divergence to trigger
    # selective-only transmission (DriftGate core)
    layer_divergence_threshold: float = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Drift detection configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DriftDetectorConfig:
    # Detection method: "ADWIN" | "PageHinkley" | "DDM" | "ensemble"
    method: str = "ensemble"

    # Sliding window size for statistical tests
    window_size: int = 30

    # ADWIN delta (false-alarm rate)
    adwin_delta: float = 0.002

    # PageHinkley threshold and alpha (calibrated for normalized cross-entropy loss ~0.2-2.0)
    ph_threshold: float = 2.5
    ph_alpha: float = 0.01

    # Transience classification:
    # If drift signal reverts within transient_window rounds → TRANSIENT
    transient_window: int = 4

    # Confidence threshold for classifying drift as PERSISTENT (0–1)
    persistence_threshold: float = 0.60

    # KL / JS divergence threshold
    divergence_threshold: float = 0.15

    # Wasserstein distance threshold
    wasserstein_threshold: float = 0.20


# ──────────────────────────────────────────────────────────────────────────────
# Client FSM state configuration (DriftGate core mechanism)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ClientFSMConfig:
    """
    Defines the Finite State Machine (FSM) for each client's participation
    eligibility. States:
        STABLE    → client participates normally (full update)
        DRIFTING  → drift detected, classification in progress
        TRANSIENT → drift classified as transient; LOCAL ADAPTATION ONLY,
                    communication SUPPRESSED (0 bytes uplink)
        PERSISTENT→ drift classified as persistent; SELECTIVE LAYER transmission
        ADAPTING  → post-drift recovery; gradual re-entry with head-only updates
        ELIGIBLE  → fully re-admitted after recovery

    Transitions driven by:
        - Drift detector signal
        - Transience classifier output
        - Server-side episode memory validation
        - Client accuracy recovery metric
    """
    # Rounds a DRIFTING client waits before classifying drift type
    classification_wait_rounds: int = 3

    # Rounds a TRANSIENT client is suppressed before rechecking
    transient_suppression_rounds: int = 4

    # Accuracy recovery threshold to transition from ADAPTING → ELIGIBLE
    recovery_accuracy_threshold: float = 0.70

    # Maximum rounds a client can stay ADAPTING before forced ELIGIBLE
    max_adapting_rounds: int = 8

    # Minimum rounds a PERSISTENT client must transmit selective layers
    # before allowed full update again
    min_persistent_selective_rounds: int = 2


# ──────────────────────────────────────────────────────────────────────────────
# Communication controller configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CommConfig:
    # Communication budget per round (fraction of total parameters)
    # 1.0 = all parameters; 0.5 = half
    budget_fraction: float = 0.6

    # Compression: "none" | "topk" | "random_sparsification" | "quantize"
    compression_method: str = "topk"

    # Top-k fraction for gradient sparsification
    topk_fraction: float = 0.3

    # Quantization bits (for "quantize" method)
    quantization_bits: int = 8

    # Whether TRANSIENT clients transmit ANY update
    transient_suppress_transmission: bool = True

    # Whether PERSISTENT clients send only drift-affected layers
    persistent_selective_transmission: bool = True

    # Bytes per float32 (for accounting)
    bytes_per_float: int = 4


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation / Episode Memory configuration (DriftGate core mechanism)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AggregationConfig:
    """
    Server-side episode memory: the server maintains a structured
    DriftEpisode record for each client. Each episode captures:
        - Drift start/end rounds
        - Drift severity during episode
        - Layer divergence profile during episode
        - Post-drift accuracy recovery trajectory
        - Whether drift recurred

    This episode memory modulates aggregation weights:
    Clients with histories of high-impact, well-resolved episodes
    get higher weights than clients with unresolved or recurring episodes.
    """
    # Base aggregation: "fedavg" | "fedprox" | "episode_weighted"
    base_method: str = "episode_weighted"

    # Episode memory depth (how many past drift episodes to consider)
    episode_memory_depth: int = 5

    # Weight decay factor: older episodes count less
    episode_recency_decay: float = 0.7

    # FedProx mu (for FedProx baseline)
    fedprox_mu: float = 0.01

    # Whether to use layer-level selective aggregation
    selective_layer_aggregation: bool = True

    # Minimum weight floor (prevents zero weighting)
    min_weight_floor: float = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Client selection configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SelectionConfig:
    # Number of clients to select per round
    clients_per_round: int = 10

    # Selection strategy: "driftgate" | "random" | "pow_d" | "oort" | "fedcs"
    strategy: str = "driftgate"

    # DriftGate selection: weights given to FSM state in priority score
    weight_stable: float = 0.4
    weight_persistent: float = 0.5   # Persistent drifters are prioritized
    weight_adapting: float = 0.3
    weight_transient: float = 0.0    # Transient clients NOT selected
    weight_eligible: float = 0.4

    # Power-of-d choices (for pow_d baseline)
    pow_d: int = 2

    # Oort: utilization weight
    oort_util_weight: float = 0.5

    # Random seed for selection
    seed: int = GLOBAL_SEED + 2


# ──────────────────────────────────────────────────────────────────────────────
# Training configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Number of global federation rounds
    num_rounds: int = 80

    # Local epochs per client per round
    local_epochs: int = 3

    # Local learning rate
    local_lr: float = 0.01

    # Momentum
    momentum: float = 0.9

    # Weight decay
    weight_decay: float = 1e-4

    # Batch size
    batch_size: int = 32

    # Global model optimizer: "sgd" | "adam"
    server_optimizer: str = "sgd"

    # Server learning rate (for server-side optimization)
    server_lr: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    name: str = "driftgate_full"
    # Seeds for multi-run reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    # Which baselines to compare: subset of available baselines
    baselines: List[str] = field(
        default_factory=lambda: ["fedavg", "fedprox", "feddrift", "flare_sim", "driftgate"]
    )
    # Ablation variants
    ablations: List[str] = field(
        default_factory=lambda: [
            "no_drift_detection",
            "no_transience_classifier",
            "no_selective_transmission",
            "no_episode_memory",
            "no_fsm",
            "full_driftgate"
        ]
    )
    output_dir: str = "./results"
    save_checkpoints: bool = True
    checkpoint_every: int = 10


# ──────────────────────────────────────────────────────────────────────────────
# Master configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detector: DriftDetectorConfig = field(default_factory=DriftDetectorConfig)
    fsm: ClientFSMConfig = field(default_factory=ClientFSMConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict for logging/saving."""
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        """Save config as JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Load config from dict."""
        cfg = cls()
        cfg.data = DataConfig(**d.get("data", {}))
        cfg.drift = DriftConfig(**d.get("drift", {}))
        cfg.model = ModelConfig(**d.get("model", {}))
        cfg.detector = DriftDetectorConfig(**d.get("detector", {}))
        cfg.fsm = ClientFSMConfig(**d.get("fsm", {}))
        cfg.comm = CommConfig(**d.get("comm", {}))
        cfg.aggregation = AggregationConfig(**d.get("aggregation", {}))
        cfg.selection = SelectionConfig(**d.get("selection", {}))
        cfg.training = TrainingConfig(**d.get("training", {}))
        cfg.experiment = ExperimentConfig(**d.get("experiment", {}))
        return cfg

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def get_default_config() -> Config:
    """Return the default configuration."""
    return Config()


def get_fast_config() -> Config:
    """Return a reduced configuration for quick testing."""
    cfg = Config()
    cfg.data.num_clients = 10
    cfg.data.samples_per_client_per_round = 64
    cfg.training.num_rounds = 20
    cfg.training.local_epochs = 2
    cfg.experiment.seeds = [42]
    cfg.selection.clients_per_round = 5
    return cfg
