# PSF-HDPO-GNN-Inventory-Optimization

This repository contains a clean-room, professional-grade implementation of the Hindsight Differentiable Policy Optimization (HDPO) framework for inventory management, with a focus on Graph Neural Network (GNN) architectures using PyTorch Geometric.

This project is being built from the ground up, following modern software engineering best practices, including a modular `src` layout, comprehensive documentation, and a full suite of unit tests.

## 1. Project Goal

The primary objective is to build a robust, maintainable, and high-performance framework to replicate and extend the research from the paper "Deep Reinforcement Learning for Inventory Networks: Toward Reliable Policy Optimization" by Alvo et al.

## 2. Current Status & Architecture

The foundational architecture of the project is now complete. The codebase is organized into a modular `src` package, with clear separation of concerns.

*   **`configs/`**: Contains all experiment configuration files, separating `settings` from `hyperparams`.
*   **`src/hdpo_gnn/`**: The core Python package.
    *   `utils/config_loader.py`: A robust, tested utility for loading and merging configurations.
    *   `engine/simulator.py`: Fully implemented and tested differentiable simulator (PyTorch), powering the HDPO engine.
    *   `data/datasets.py`: Synthetic training data generation logic (config-driven tensors for inventories, demands, costs, and lead times).
*   **`tests/`**: A suite of unit tests to ensure code correctness and prevent regressions.
*   **`train.py`**: The main entry point and orchestrator for all training runs.

**Current milestone: The full, end-to-end simulation engine is functional. The system can now load configurations, generate synthetic data, and run a complete, multi-period simulation episode. All core components are unit-tested.**

## 3. How to Run

### 3.1. Setup

1.  **Clone and Install:**
    ```bash
    git clone https://github.com/Sancauid/PSF-HDPO-GNN-Inventory-Optimization.git
    cd PSF-HDPO-GNN-Inventory-Optimization
    conda activate pyg_env
    pip install -e .
    pip install pytest # For running tests
    ```

2.  **Run Tests (Verify Installation):**
    ```bash
    pytest
    ```
    *(All tests should pass.)*

### 3.2. Running the Main Script (Test Simulation)

The main entry point now loads configurations, generates synthetic data, and runs a test simulation episode while logging per-step costs.

```bash
python train.py configs/settings/base_setting.yml configs/hyperparams/vanilla_mlp.yml
```


