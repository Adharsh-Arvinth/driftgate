"""
visualization.py — Plotting and visualization for DriftGate experiments

Produces:
  1. Accuracy curves across rounds (multi-method comparison)
  2. Communication efficiency plot
  3. Client state distribution over rounds (stacked area)
  4. Ablation study bar chart
  5. Drift severity heatmap
  6. Episode memory weight evolution
  7. Bandwidth reduction comparison
  8. Convergence comparison
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available; visualization disabled.")


# ──────────────────────────────────────────────────────────────────────────────
# Style configuration
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "A_fedavg": "#6c757d",
    "B_fedprox": "#adb5bd",
    "C_oort": "#f8961e",
    "D_no_drift_detection": "#f4a261",
    "E_no_transience_classifier": "#90be6d",
    "F_no_selective_transmission": "#43aa8b",
    "G_no_episode_memory": "#4895ef",
    "H_no_fsm": "#7209b7",
    "I_full_driftgate": "#f72585",
    "driftgate": "#f72585",
    "fedavg": "#6c757d",
    "fedprox": "#adb5bd",
    "oort": "#f8961e",
}

STATE_COLORS = {
    "stable": "#2d6a4f",
    "eligible": "#52b788",
    "adapting": "#f8961e",
    "persistent": "#f3722c",
    "drifting": "#ae2012",
    "transient": "#7209b7",
}

FIGURE_SIZE = (12, 7)
DPI = 150


def _set_style():
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": DPI,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 1. Accuracy curves
# ──────────────────────────────────────────────────────────────────────────────
def plot_accuracy_curves(
    results: Dict[str, List[Dict]],
    drift_round: Optional[int] = None,
    title: str = "Global Model Accuracy over Communication Rounds",
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Plot accuracy curves for multiple methods/ablations.

    Args:
        results: {method_name: [per_round_metrics_list_per_seed]}
        drift_round: If provided, mark with vertical line
        output_path: Save path
        show: Whether to call plt.show()
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    _set_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for method_name, seed_results in results.items():
        color = PALETTE.get(method_name, "#333333")
        # Each seed_result can be a list of round_metrics or a single dict
        all_curves = []
        for result in seed_results:
            if isinstance(result, dict):
                round_metrics = result.get("round_metrics", [])
                if round_metrics:
                    accs = [m["accuracy"] for m in round_metrics]
                    all_curves.append(accs)
            elif isinstance(result, list):
                all_curves.append([m["accuracy"] for m in result])

        if not all_curves:
            continue

        min_len = min(len(c) for c in all_curves)
        all_curves = [c[:min_len] for c in all_curves]
        arr = np.array(all_curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        rounds = list(range(1, min_len + 1))

        label = method_name.replace("_", " ").replace("A ", "A: ").replace("B ", "B: ")
        ax.plot(rounds, mean, color=color, linewidth=2, label=label)
        ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)

    if drift_round is not None:
        ax.axvline(x=drift_round, color="red", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"Drift onset (r={drift_round})")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path
    if show:
        plt.show()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. Communication efficiency
# ──────────────────────────────────────────────────────────────────────────────
def plot_communication_efficiency(
    results: Dict[str, List[Dict]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """Bandwidth reduction comparison bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    method_names, mean_reductions, std_reductions = [], [], []
    for method_name, seed_results in results.items():
        reductions = []
        for result in seed_results:
            if isinstance(result, dict):
                comm = result.get("comm_stats", {})
                reductions.append(comm.get("bandwidth_reduction_pct", 0.0))
        if reductions:
            method_names.append(method_name.replace("_", "\n"))
            mean_reductions.append(np.mean(reductions))
            std_reductions.append(np.std(reductions))

    colors = [PALETTE.get(m.replace("\n", "_"), "#4895ef") for m in method_names]
    x = np.arange(len(method_names))
    bars = ax.bar(x, mean_reductions, color=colors, alpha=0.85, edgecolor="white")
    ax.errorbar(x, mean_reductions, yerr=std_reductions, fmt="none",
                color="black", capsize=5, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=9)
    ax.set_ylabel("Bandwidth Reduction (%)")
    ax.set_title("Communication Efficiency: Bandwidth Reduction vs Baseline")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path
    if show:
        plt.show()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 3. Client state distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_client_state_distribution(
    state_history: List[Dict],
    drift_round: Optional[int] = None,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """Stacked area chart of client FSM state distribution over rounds."""
    if not MATPLOTLIB_AVAILABLE or not state_history:
        return None

    _set_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    states = ["stable", "eligible", "adapting", "persistent", "drifting", "transient"]
    rounds = [r.get("round", i + 1) for i, r in enumerate(state_history)]

    state_data = {s: [r.get(s, 0) for r in state_history] for s in states}

    # Stacked area
    bottom = np.zeros(len(rounds))
    for state in states:
        vals = np.array(state_data[state], dtype=float)
        ax.fill_between(rounds, bottom, bottom + vals,
                        label=state.capitalize(),
                        color=STATE_COLORS.get(state, "#999"),
                        alpha=0.8)
        bottom += vals

    if drift_round is not None:
        ax.axvline(x=drift_round, color="red", linestyle="--",
                   linewidth=1.5, alpha=0.8, label=f"Drift onset (r={drift_round})")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Number of Clients")
    ax.set_title("Client FSM State Distribution over Rounds (DriftGate)")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path
    if show:
        plt.show()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ablation study summary
# ──────────────────────────────────────────────────────────────────────────────
def plot_ablation_summary(
    ablation_results: Dict[str, List[Dict]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """2-panel: accuracy + bandwidth reduction for all ablations."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    _set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names, accs, bws = [], [], []
    acc_stds, bw_stds = [], []

    for name, seed_results in ablation_results.items():
        run_accs = []
        run_bws = []
        for r in seed_results:
            if isinstance(r, dict):
                run_accs.append(r.get("final_eval", {}).get("accuracy", 0))
                run_bws.append(r.get("comm_stats", {}).get("bandwidth_reduction_pct", 0))
        if run_accs:
            names.append(name.split("_", 1)[-1].replace("_", " ") if "_" in name else name)
            accs.append(np.mean(run_accs))
            acc_stds.append(np.std(run_accs))
            bws.append(np.mean(run_bws))
            bw_stds.append(np.std(run_bws))

    x = np.arange(len(names))
    colors = [PALETTE.get(list(ablation_results.keys())[i], "#4895ef") for i in range(len(names))]

    ax1.bar(x, accs, color=colors, alpha=0.85, edgecolor="white")
    ax1.errorbar(x, accs, yerr=acc_stds, fmt="none", color="black", capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Final Accuracy")
    ax1.set_title("Ablation: Final Accuracy")
    ax1.set_ylim(0, 1.0)

    ax2.bar(x, bws, color=colors, alpha=0.85, edgecolor="white")
    ax2.errorbar(x, bws, yerr=bw_stds, fmt="none", color="black", capsize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Bandwidth Reduction (%)")
    ax2.set_title("Ablation: Communication Efficiency")

    plt.suptitle("DriftGate Ablation Study", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path
    if show:
        plt.show()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 5. Drift detection heatmap
# ──────────────────────────────────────────────────────────────────────────────
def plot_drift_severity_heatmap(
    client_results_per_round: List[Dict],
    num_clients: int,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """Client × Round heatmap of drift severity."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    _set_style()
    num_rounds = len(client_results_per_round)
    matrix = np.zeros((num_clients, num_rounds))

    for r_idx, round_data in enumerate(client_results_per_round):
        severities = round_data.get("drift_severities", {})
        for cid, sev in severities.items():
            if cid < num_clients:
                matrix[cid, r_idx] = sev

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Drift Severity")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Client ID")
    ax.set_title("Client Drift Severity Heatmap (DriftGate)")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path
    if show:
        plt.show()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 6. Full experiment report figure
# ──────────────────────────────────────────────────────────────────────────────
def generate_full_report(
    ablation_results: Dict[str, List[Dict]],
    main_result: Dict,
    output_dir: str = "./results",
    drift_round: int = 30,
) -> List[str]:
    """Generate all figures for the experiment report."""
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # Accuracy curves
    p = plot_accuracy_curves(
        ablation_results,
        drift_round=drift_round,
        output_path=os.path.join(output_dir, "accuracy_curves.png"),
    )
    if p:
        generated.append(p)

    # Communication efficiency
    p = plot_communication_efficiency(
        ablation_results,
        output_path=os.path.join(output_dir, "comm_efficiency.png"),
    )
    if p:
        generated.append(p)

    # Ablation summary
    p = plot_ablation_summary(
        ablation_results,
        output_path=os.path.join(output_dir, "ablation_summary.png"),
    )
    if p:
        generated.append(p)

    # State distribution (for DriftGate full run)
    if main_result and "state_history" in main_result:
        p = plot_client_state_distribution(
            main_result["state_history"],
            drift_round=drift_round,
            output_path=os.path.join(output_dir, "state_distribution.png"),
        )
        if p:
            generated.append(p)

    print(f"Generated {len(generated)} figures in {output_dir}")
    return generated
