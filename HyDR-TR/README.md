# HyDR-TR: Hybrid Deep Reinforcement Learning for Multi-Agent TSP

Official implementation of **HyDR-TR**, a hybrid deep reinforcement learning framework for solving multi-agent Traveling Salesman Problems (TSP).

---

## 🧭 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Generation](#data-generation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproduction Quick-Start](#reproduction-quick-start)
- [Project Structure](#project-structure)
- [License](#license)

---

## 📘 Overview

**HyDR-TR (Hybrid Deep Reinforcement Learning with Tree-based Refinement)**  
combines neural policy learning with combinatorial optimization for improved generalization and scalability on large TSP instances.

The model integrates:
- **Graph neural networks (GNNs)** for route construction.  
- **Reinforcement learning (RL)** for adaptive policy updates.  
- **Tree-based search refinement** for enhanced solution quality.

---

## ⚙️ Installation

Install required dependencies:

```bash
pip install torch torch-geometric numpy PyYAML tqdm scikit-learn matplotlib tensorboard ortools
```

Make sure the GLKH solver has execution permissions:

```bash
chmod +x GLKH
```

---

## 🧩 Data Generation

Generate training and validation datasets:

```bash
# Generate validation data for 51-node TSP with 512 instances
python3 data_generator.py
```

---

## 🏋️ Training

Train the HyDR-TR model:

```bash
python3 train_HyDR_TR.py
```

Train the GIN-GLKH baseline model:

```bash
python3 train_GIN_GLKH.py
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir log/
```

---

## 📊 Evaluation & Analysis

Perform benchmark evaluations and analyses:

```bash
python3 analyze_TSPLIB.py          # TSPLIB benchmark evaluation
python3 analyze_random_dataset.py  # Random dataset analysis
python3 analyze_sensitivity.py     # Sensitivity analysis
python3 analyze_AA_GLKH_correlation.py  # AA-GLKH correlation study
```

---

## 🚀 Reproduction Quick-Start

To support end-to-end verification, this section provides a concise guide for reproducing the **TSPLIB benchmark results**.

After cloning the repository, you can reproduce all benchmark experiments with a single command.

```bash
# Default
python3 analyze_subset_TSPLIB.py

# Run specific instances with 10 runs
python3 analyze_subset_TSPLIB.py     --instances berlin52,gr96 --n_runs 10

# Run all instances with custom output directory
python3 analyze_subset_TSPLIB.py     --instances all --output TSPLIB_custom_output

# Run with custom seeds
python3 analyze_subset_TSPLIB.py     --instances berlin52 --n_runs 5     --seeds 42,123,456,789,999

# Run with seed range
python3 analyze_subset_TSPLIB.py     --instances berlin52 --n_runs 10 --seeds 100-109

# Get help
python3 analyze_subset_TSPLIB.py --help
```

Each command executes the trained models and **reproduces the reported TSPLIB results deterministically**.  
Unless otherwise specified, the default configuration runs **30 repetitions** using predefined random seeds for reproducibility.

---

## 🗂️ Project Structure

```
HyDR-TR/
├── train_HyDR_TR.py              # HyDR-TR training script
├── policy_HyDR_TR.py             # HyDR-TR policy network
├── validation_HyDR_TR.py         # HyDR-TR validation
├── train_GIN_GLKH.py             # GIN-GLKH baseline training
├── policy_GIN_GLKH.py            # GIN-GLKH policy
├── validation_GIN_GLKH.py        # GIN-GLKH validation
├── data_generator.py             # Dataset generation
├── analyze_*.py                  # Analysis scripts
├── GLKH                          # GLKH solver binary
├── TSPLIB/                       # Benchmark datasets
├── saved_model/                  # Model checkpoints
└── utils/                        # Utility functions
```

---

## 📜 License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.