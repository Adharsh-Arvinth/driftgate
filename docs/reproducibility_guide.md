# DriftGate: Reproducibility & Execution Guide

This guide provides end-to-end instructions for reproducing all benchmark figures, running unit and integration tests, launching the interactive Streamlit dashboard, and running the Google Colab notebook.

---

## 1. System Requirements & Environment Setup

### 1.1 Prerequisites
- Python 3.10+ (tested on Python 3.13.0, 64-bit Windows & Linux)
- Recommended RAM: $\ge 8$ GB
- GPU: Optional (fully optimized for standard multi-core CPU execution via vectorized NumPy and PyTorch)

### 1.2 Installation
```bash
# Clone or navigate to the project directory
cd /path/to/project

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Running Test Suites

DriftGate includes 30 verified automated tests covering drift detection, FSM transitions, client selection, communication control, episode-memory aggregation, and end-to-end integration:

```bash
# Run the entire test suite
python -m unittest discover tests

# Or run individual modules:
python tests/test_drift.py
python tests/test_fsm.py
python tests/test_selection.py
python tests/test_transmission.py
python tests/test_aggregation.py
python tests/test_integration.py
```

---

## 3. Running Benchmarks & Generating Results

The benchmark suite evaluates:
- **9 Ablation Variants (A through I)**: FedAvg, FedProx, Oort, No-Drift-Det, No-Transience, No-Selective, No-Episode-Memory, No-FSM, Full DriftGate
- **5 Drift Scenarios**: Abrupt, Gradual, Recurring, Heterogeneous, Mixed
- **Multi-Seed Statistical Replication**: Computing mean and variance across random seeds

```bash
# Execute the full benchmark suite
python run_benchmarks.py
```

All results and plots will be saved to `results/`:
- `results/benchmark_summary.json`
- `results/accuracy_curves.png`
- `results/comm_efficiency.png`
- `results/ablation_summary.png`
- `results/state_distribution.png`

---

## 4. Launching the Interactive Frontend Dashboard

DriftGate features a comprehensive, professional Streamlit web interface with 8 navigation sections:
- Executive Summary & KPI cards
- Live Interactive Simulation Runner
- Accuracy vs. Communication Tradeoff curves
- 6-State FSM Transition Visualizer
- Layer Divergence & Sparsification inspector
- Server Episode-Memory Ledger view
- Patent Claim & Prior-Art Explorer
- Section 3(k) CRI Compliance view

```bash
# Launch the Streamlit dashboard
streamlit run frontend/app.py
```

---

## 5. Google Colab Notebook

To run in the cloud with zero local setup:
1. Open Google Colab: [https://colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/driftgate_colab.ipynb`
3. Run all cells sequentially. The notebook clones the repo or runs standalone with inline synthetic or UCI HAR data generation.
