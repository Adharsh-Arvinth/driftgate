"""
run_benchmarks.py — Script to execute full benchmark suites, ablations,
and generate experiment artifacts and graphs.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from config import Config, get_default_config
from experiments import run_ablation_study, run_drift_scenarios, save_results
from visualization import generate_full_report


def main():
    print("=" * 70)
    print("DRIFTGATE FEDERATED LEARNING: BENCHMARK SUITE EXECUTION")
    print("=" * 70)

    cfg = get_default_config()
    cfg.training.num_rounds = 40
    cfg.data.num_clients = 10
    cfg.selection.clients_per_round = 5
    cfg.drift.global_drift_round = 15
    cfg.experiment.seeds = [42, 123]

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(results_dir, exist_ok=True)

    print("\n[Phase 1] Executing Ablation Study (9 variants x 2 seeds)...")
    ablation_results = run_ablation_study(
        base_cfg=cfg,
        seeds=cfg.experiment.seeds,
        output_dir=results_dir,
        verbose=True
    )

    print("\n[Phase 2] Executing Multi-Scenario Drift Evaluations...")
    scenario_results = run_drift_scenarios(
        base_cfg=cfg,
        seed=42,
        output_dir=results_dir
    )

    print("\n[Phase 3] Generating Scientific Visualization Figures...")
    main_full_res = ablation_results.get("I_full_driftgate", [{}])[0]
    generated_figs = generate_full_report(
        ablation_results=ablation_results,
        main_result=main_full_res,
        output_dir=results_dir,
        drift_round=cfg.drift.global_drift_round
    )
    print(f"Generated {len(generated_figs)} plot assets in {results_dir}")

    # Summary table generation
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    summary = {}
    for name, seed_runs in ablation_results.items():
        accs = [r["final_eval"]["accuracy"] for r in seed_runs]
        f1s = [r["final_eval"]["f1"] for r in seed_runs]
        bw_reds = [r["comm_stats"]["bandwidth_reduction_pct"] for r in seed_runs]
        summary[name] = {
            "accuracy_mean": float(sum(accs) / len(accs)),
            "f1_mean": float(sum(f1s) / len(f1s)),
            "bw_reduction_pct_mean": float(sum(bw_reds) / len(bw_reds)),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBenchmark run completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
