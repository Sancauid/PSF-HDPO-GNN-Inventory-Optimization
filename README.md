# PSF-HDPO-GNN-Inventory-Optimization

This repository contains a professional-grade, clean-room implementation of the **Hindsight Differentiable Policy Optimization (HDPO)** framework for inventory management. The primary focus is on Graph Neural Network (GNN) architectures, built from the ground up using PyTorch and PyTorch Geometric.

This document serves as a comprehensive guide to the project's architecture, data flow, and operational procedures, designed to be understood by both human collaborators and Large Language Models (LLMs).

## 1. Core Concepts & Objective

The goal of this project is to replicate and extend the research from the paper "Deep Reinforcement Learning for Inventory Networks" (Alvo et al.). We use a **differentiable simulator** to train neural network policies via **pathwise gradients**.

- **HDPO:** Unlike traditional RL methods that rely on high-variance estimators (like REINFORCE), HDPO treats the entire simulation as a differentiable computation graph. This allows for stable, low-variance gradient updates by backpropagating directly through the known system dynamics.
- **Optimization Objective:** The framework is designed to **minimize a cost function**. The `reward` variable returned by the simulator is semantic `cost` (`holding_cost + stockout_cost` or `holding_cost - revenue`). The optimizer's goal is to find policy parameters θ that minimize the expected total cost over a simulation episode.

## 2. Project Architecture

The codebase is organized into a modular `src` package to ensure clarity, maintainability, and reusability. **All project-internal imports are absolute, starting from `hdpo_gnn`.**

```
/
├── configs/                  # Experiment configurations (YAML)
├── src/
│  └── hdpo_gnn/             # The core Python package
│     ├── data/              # Data loading and generation
│     ├── engine/            # The core simulation logic
│     ├── models/            # Policy network architectures
│     └── training/          # The training orchestration logic
├── tests/                    # Unit tests for all components
└── train.py                  # Main entry point for experiments
```

### 2.1. `configs/`
Contains all `.yml` configuration files. Experiments are defined by combining a `settings` file (problem definition) with a `hyperparams` file (model/optimizer definition).

### 2.2. `src/hdpo_gnn/` - The Core Package

- **`data/datasets.py`:**
  - `create_synthetic_data_dict()`: Generates a dictionary of raw PyTorch tensors representing a batch of simulation scenarios (demands, initial inventories, costs). This is the source of truth for our synthetic experiments.
  - `create_pyg_dataset()`: Converts the synthetic data dictionary into a list of `torch_geometric.data.Data` graphs for PyG-based models (fully connected store graph per sample, with node features and per-sample attributes such as demands and costs).

- **`engine/simulator.py`:**
  - `DifferentiableSimulator`: A class-based, differentiable simulator built from first principles in PyTorch. It contains two main methods:
    - `reset()`: Initializes the simulator state for a batch of scenarios.
    - `step()`: Advances the simulation by one time step, applying actions, satisfying demand, and calculating the cost for that period.

- **`models/`:**
  - `vanilla.py`: Defines `VanillaPolicy`, a standard Multi-Layer Perceptron (MLP) that takes the entire flattened state as input and outputs per-store and warehouse logits.
  - `gnn.py`: Defines `GNNPolicy`, a Graph Neural Network (GCN-based) that operates on graph-structured data and produces per-node outputs.

- **`training/trainer.py`:**
  - `Trainer`: The main orchestrator class. It encapsulates the entire training process.
    - `__init__()`: Receives all necessary components (model, optimizer, data, simulator, config).
    - `train()`: Contains the main epoch loop.
    - `_train_epoch()`: Contains the batch loop (`for batch in ...`).
    - `_train_batch()`: Contains the logic for a single batch, including the episodic simulation loop (`for t in range(periods):`), action generation, feasibility enforcement (FELs), cost calculation, backpropagation, and optimizer step.

### 2.3. `train.py` - The Entry Point

A lightweight script responsible for:
1. Parsing command-line arguments (config paths, model choice, epochs).
2. Loading and merging configurations.
3. Instantiating all components (`DataLoader`, `model`, `optimizer`, `Simulator`, `Trainer`).
4. Calling `trainer.train()` to start the process.

## 3. Data Flow & Execution Logic

The end-to-end data flow for a single training step is as follows:

1. `train.py` is executed.
2. `load_configs` reads and merges the specified `.yml` files.
3. `create_synthetic_data_dict` generates a dictionary of tensors for the entire dataset.
4. A `torch.utils.data.DataLoader` (or iterable) is created to serve batches of this data.
5. The `Trainer` is initialized.
6. `trainer.train()` begins. In each batch:
   a. The `Trainer` receives a `batch` of data (e.g., 256 samples).
   b. `simulator.reset()` is called with this batch data to initialize the simulation state.
   c. A loop runs for `t` in `periods`:
      i. The current `observation` from the simulator is fed to the `model`.
      ii. The model's raw output (`logits`) is passed through a **Feasibility Enforcement Layer (FEL)** to produce valid `actions`.
      iii. The `actions` are passed to `simulator.step()`.
      iv. The `simulator` returns the `cost` for that step, which is accumulated.
   d. After the episode, a final `loss` is calculated from the total accumulated cost, normalized correctly.
   e. `loss.backward()` computes gradients through the entire unrolled simulation.
   f. `optimizer.step()` updates the model's weights.

## 4. How to Use

### 4.1. Setup
Clone the repository and install the package in editable mode.

```bash
git clone https://github.com/Sancauid/PSF-HDPO-GNN-Inventory-Optimization.git
cd PSF-HDPO-GNN-Inventory-Optimization
conda activate pyg_env
pip install -e .
pip install pytest
```

### 4.2. Running Tests
Verify the integrity of all components by running the unit tests.

```bash
pytest
```

### 4.3. Running Training
Launch a training run from the root directory.

```bash
# Example: Train the Vanilla MLP model for 20 epochs
python train.py configs/settings/base_setting.yml configs/hyperparams/vanilla.yml --model vanilla --epochs 20
```

## 5. Development & Quality Assurance

Use the included helper script to run formatting and tests locally before committing changes.

```bash
./run_quality_checks.sh
```

This script is intended to enforce code style and verify unit tests pass, helping maintain a consistently healthy codebase.