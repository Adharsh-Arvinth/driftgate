"""
experiments.py — Experiment runner: baselines, ablations, multi-seed

Runs:
  A: FedAvg baseline
  B: FedProx baseline
  C: Oort baseline
  D: DriftGate without drift detection (random selection + FedAvg)
  E: DriftGate without transience classifier (no suppression)
  F: DriftGate without selective transmission (full updates always)
  G: DriftGate without episode memory (uniform aggregation)
  H: DriftGate without FSM (no state gating)
  I: Full DriftGate

Saves results to JSON. Supports multi-seed execution.
"""

import os
import json
import copy
import time
from typing import Dict, List, Optional, Any
import numpy as np

from config import Config, get_default_config, get_fast_config
from federation import FederationRunner, BaselineRunner
from metrics import MetricsCollector


# ──────────────────────────────────────────────────────────────────────────────
# Ablation configurations
# ──────────────────────────────────────────────────────────────────────────────
def get_ablation_configs(base_cfg: Config) -> Dict[str, Config]:
    """
    Generate ablation configurations by disabling specific DriftGate mechanisms.

    Returns dict: {ablation_name: config}
    """
    ablations = {}

    # ── A: FedAvg baseline ────────────────────────────────────────────────────
    cfg_a = copy.deepcopy(base_cfg)
    cfg_a.selection.strategy = "random"
    cfg_a.aggregation.base_method = "fedavg"
    cfg_a.comm.compression_method = "none"
    cfg_a.comm.transient_suppress_transmission = False
    cfg_a.comm.persistent_selective_transmission = False
    cfg_a.aggregation.selective_layer_aggregation = False
    ablations["A_fedavg"] = cfg_a

    # ── B: FedProx baseline ───────────────────────────────────────────────────
    cfg_b = copy.deepcopy(base_cfg)
    cfg_b.selection.strategy = "random"
    cfg_b.aggregation.base_method = "fedprox"
    cfg_b.comm.compression_method = "none"
    cfg_b.comm.transient_suppress_transmission = False
    cfg_b.comm.persistent_selective_transmission = False
    cfg_b.aggregation.selective_layer_aggregation = False
    ablations["B_fedprox"] = cfg_b

    # ── C: Oort (client selection baseline) ──────────────────────────────────
    cfg_c = copy.deepcopy(base_cfg)
    cfg_c.selection.strategy = "oort"
    cfg_c.aggregation.base_method = "fedavg"
    cfg_c.comm.compression_method = "none"
    cfg_c.comm.transient_suppress_transmission = False
    cfg_c.comm.persistent_selective_transmission = False
    cfg_c.aggregation.selective_layer_aggregation = False
    ablations["C_oort"] = cfg_c

    # ── D: DriftGate − Drift Detection (no drift awareness at all) ────────────
    cfg_d = copy.deepcopy(base_cfg)
    cfg_d.selection.strategy = "random"
    cfg_d.aggregation.base_method = "fedavg"
    cfg_d.comm.transient_suppress_transmission = False
    cfg_d.comm.persistent_selective_transmission = False
    cfg_d.aggregation.selective_layer_aggregation = False
    ablations["D_no_drift_detection"] = cfg_d

    # ── E: DriftGate − Transience Classifier (no suppression; all clients send) ─
    cfg_e = copy.deepcopy(base_cfg)
    cfg_e.comm.transient_suppress_transmission = False
    # Transience classifier is disabled by forcing all clients to stay in
    # PERSISTENT mode when drift is detected (never TRANSIENT suppression)
    cfg_e.fsm.transient_suppression_rounds = 0  # Immediately exit transient
    ablations["E_no_transience_classifier"] = cfg_e

    # ── F: DriftGate − Selective Transmission (full updates always) ───────────
    cfg_f = copy.deepcopy(base_cfg)
    cfg_f.comm.persistent_selective_transmission = False
    cfg_f.aggregation.selective_layer_aggregation = False
    ablations["F_no_selective_transmission"] = cfg_f

    # ── G: DriftGate − Episode Memory (uniform aggregation) ──────────────────
    cfg_g = copy.deepcopy(base_cfg)
    cfg_g.aggregation.base_method = "fedavg"  # No episode weighting
    ablations["G_no_episode_memory"] = cfg_g

    # ── H: DriftGate − FSM (no state gating; all clients always eligible) ────
    cfg_h = copy.deepcopy(base_cfg)
    # Disable state gating: clients are always treated as STABLE
    cfg_h.fsm.classification_wait_rounds = 9999  # Never classify
    cfg_h.comm.transient_suppress_transmission = False
    cfg_h.comm.persistent_selective_transmission = False
    ablations["H_no_fsm"] = cfg_h

    # ── I: Full DriftGate ─────────────────────────────────────────────────────
    ablations["I_full_driftgate"] = copy.deepcopy(base_cfg)

    return ablations


# ──────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ──────────────────────────────────────────────────────────────────────────────
def run_single_experiment(
    cfg: Config,
    name: str,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one experiment with one seed. Returns result dict."""
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.experiment.seeds = [seed]

    if verbose:
        print(f"\n>>> Running: {name} | seed={seed}")

    runner = FederationRunner(cfg_copy, seed=seed)
    result = runner.run()
    result["experiment_name"] = name
    result["seed"] = seed
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed experiment runner
# ──────────────────────────────────────────────────────────────────────────────
def run_multi_seed_experiment(
    cfg: Config,
    name: str,
    seeds: List[int],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run multiple seeds for one configuration."""
    results = []
    for seed in seeds:
        result = run_single_experiment(cfg, name, seed, verbose=verbose)
        results.append(result)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Full ablation study
# ──────────────────────────────────────────────────────────────────────────────
def run_ablation_study(
    base_cfg: Config,
    seeds: Optional[List[int]] = None,
    output_dir: str = "./results",
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Run all ablation configurations.

    Returns dict: {ablation_name: [results_per_seed]}
    """
    if seeds is None:
        seeds = base_cfg.experiment.seeds

    os.makedirs(output_dir, exist_ok=True)
    ablation_cfgs = get_ablation_configs(base_cfg)
    all_results: Dict[str, List[Dict]] = {}

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {len(ablation_cfgs)} variants × {len(seeds)} seeds")
    print(f"{'='*60}")

    for abl_name, abl_cfg in ablation_cfgs.items():
        t_start = time.time()
        seed_results = run_multi_seed_experiment(abl_cfg, abl_name, seeds, verbose=verbose)
        all_results[abl_name] = seed_results
        t_elapsed = time.time() - t_start

        # Save intermediate
        out_path = os.path.join(output_dir, f"{abl_name}.json")
        save_results(seed_results, out_path)

        if verbose:
            final_accs = [r["final_eval"]["accuracy"] for r in seed_results]
            comm_bws = [r["comm_stats"].get("bandwidth_reduction_pct", 0) for r in seed_results]
            print(f"  {abl_name}: acc={np.mean(final_accs):.3f}±{np.std(final_accs):.3f} "
                  f"bw_reduction={np.mean(comm_bws):.1f}% [{t_elapsed:.1f}s]")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Results I/O
# ──────────────────────────────────────────────────────────────────────────────
def save_results(results: Any, path: str) -> None:
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved: {path}")


def load_results(path: str) -> Any:
    """Load experiment results from JSON."""
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Drift scenario experiments
# ──────────────────────────────────────────────────────────────────────────────
def run_drift_scenarios(
    base_cfg: Config,
    seed: int = 42,
    output_dir: str = "./results",
) -> Dict[str, Dict]:
    """Run DriftGate under all drift scenario types."""
    scenarios = ["gradual", "abrupt", "recurring", "heterogeneous", "mixed"]
    results = {}

    for scenario in scenarios:
        cfg_s = copy.deepcopy(base_cfg)
        cfg_s.drift.drift_scenario = scenario
        result = run_single_experiment(cfg_s, f"driftgate_{scenario}", seed)
        results[scenario] = result

        out_path = os.path.join(output_dir, f"drift_scenario_{scenario}.json")
        save_results(result, out_path)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
def run_quick_sanity_check() -> Dict:
    """Run a very fast test to verify the system works end-to-end."""
    cfg = get_fast_config()
    cfg.training.num_rounds = 5
    cfg.data.num_clients = 5
    cfg.selection.clients_per_round = 3
    cfg.experiment.seeds = [42]
    result = run_single_experiment(cfg, "sanity_check", seed=42, verbose=True)
    print("Sanity check passed!")
    return result
