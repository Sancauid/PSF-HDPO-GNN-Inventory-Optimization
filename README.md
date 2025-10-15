# PSF-HDPO-GNN-Inventory-Optimization

This repository contains a professional-grade, clean-room implementation of the **Hindsight Differentiable Policy Optimization (HDPO)** framework for inventory management. The primary focus is on Graph Neural Network (GNN) architectures, built from the ground up using PyTorch and PyTorch Geometric.

This document serves as a comprehensive guide to the project's architecture, data flow, and operational procedures, designed to be understood by both human collaborators and Large Language Models (LLMs).

## 1. Core Concepts & Objective

The goal of this project is to replicate and extend the research from the paper "Deep Reinforcement Learning for Inventory Networks" (Alvo et al.). We use a **differentiable simulator** to train neural network policies via **pathwise gradients**.

- **HDPO:** Unlike traditional RL methods that rely on high-variance estimators (like REINFORCE), HDPO treats the entire simulation as a differentiable computation graph. This allows for stable, low-variance gradient updates by backpropagating directly through the known system dynamics.
- **Optimization Objective:** The framework is designed to **minimize a cost function**. The `reward` variable returned by the simulator is semantic `cost` (`holding_cost + stockout_cost` or `holding_cost - revenue`). The optimizer's goal is to find policy parameters θ that minimize the expected total cost over a simulation episode.

## Current milestone

**Project Complete: The full, end-to-end HDPO training framework is implemented, numerically validated, and ready for experimentation.**

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
├── tests/                    # A comprehensive suite of unit and integration tests, including a numerical gradient check (`gradcheck`), which validates the end-to-end differentiability of the entire simulation pipeline.
└── train.py                  # Main entry point for experiments
```

### 2.1. `configs/`
Contains all `.yml` configuration files. Experiments are defined by combining a `settings` file (problem definition) with a `hyperparams` file (model/optimizer definition).

### 2.2. `src/hdpo_gnn/` - The Core Package

- **`data/datasets.py`:**
  - `create_synthetic_data_dict()`: Generates a dictionary of raw PyTorch tensors representing a batch of simulation scenarios (demands, initial inventories, costs). This is the source of truth for our synthetic experiments.
  - `create_pyg_dataset()`: Converts the synthetic data dictionary into a list of `torch_geometric.data.Data` graphs for PyG-based models (fully connected store graph per sample, with node features and per-sample attributes such as demands and costs).

- **`engine/functional.py`:**
  - `transition_step(state, action, demand_t, cost_params, lead_times)`: Pure functional simulator physics. Takes the current state and returns `(next_state, cost)` without mutating global state, preserving the computation graph.
  - Episode orchestration is performed from `training/engine.py` (see below).

- **`models/`:**
  - `vanilla.py`: Defines `VanillaPolicy`, a standard Multi-Layer Perceptron (MLP) that takes the entire flattened state as input and outputs per-store and warehouse logits.
  - `gnn.py`: Defines `GNNPolicy`, a Graph Neural Network (GCN-based) that operates on graph-structured data and produces per-node outputs.

- **`training/engine.py`:**
  - `prepare_batch_for_simulation(...)`: Prepares dense tensors (vanilla or PyG) for the episode.
  - `run_simulation_episode(...)`: Executes a multi-period episode by calling `transition_step` at each step; applies `store_mask` to actions; accumulates costs and computes `cost_to_report` considering `ignore_periods`.
- **`training/trainer.py`:**
  - `Trainer`: Training orchestrator.
    - `__init__()`: Receives `model`, `optimizer`, `scheduler`, `train_loader`, `configs` (no longer needs a stateful simulator object).
    - `train()`: Epoch loop.
    - `_train_epoch()`: Batch loop.
    - `_train_batch()`: Orchestrates calls to `prepare_batch_for_simulation`, `run_simulation_episode`, loss calculation, and gradient step.

### 2.3. `train.py` - The Entry Point

A lightweight script responsible for:
1. Parsing command-line arguments (config paths, model choice, epochs).
2. Loading and merging configurations.
3. Instantiating all components (`DataLoader`, `model`, `optimizer`, `Trainer`).
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
   b. Initial state and tensors are prepared with `prepare_batch_for_simulation`.
   c. A loop runs over `t` in `periods` calling `run_simulation_episode`, which internally invokes `transition_step(state, action, demand_t, ...)` and accumulates cost per sample at each step.
   d. After the episode, the `loss` is calculated from the accumulated cost, normalized correctly (and `cost_to_report` considering `ignore_periods`).
   e. `loss.backward()` propagates gradients through the entire episode.
   f. `optimizer.step()` updates the model weights.

### 3.1. End-to-end computation graph (visual flow)

The complete gradient flow from training entry point to backpropagation:

```
train.py
   ↓
Trainer.train()
   ↓
Trainer._train_batch()
   ↓
prepare_batch_for_simulation  ← prepares tensors/graphs for episode
   ↓
run_simulation_episode
   ↓
   ├─→ [for t in periods]:
   │      ↓
   │   model.forward(state_t) → logits  [GRADIENTS FLOW HERE]
   │      ↓
   │   sigmoid/softplus → actions_t     [GRADIENTS FLOW HERE]
   │      ↓
   │   transition_step(state_t, actions_t, demand_t, ...)  [GRADIENTS FLOW HERE]
   │      ↓
   │   (next_state, step_cost)
   │      ↓
   │   total_episode_cost += step_cost  [GRADIENTS ACCUMULATE]
   │
   ↓
calculate_losses(total_episode_cost, ...) → loss_for_backward
   ↓
loss.backward()  ← BACKPROP THROUGH ENTIRE MULTI-PERIOD EPISODE
   ↓
optimizer.step()  ← UPDATE POLICY PARAMETERS
```

**Key differentiability guarantees:**
- The sales function uses a smooth `softplus`-based approximation instead of hard `min(demand, inventory)`.
- The warehouse-to-store shipment feasibility cap uses a smooth sigmoid-like function: `ratio / (1 + ratio)`.
- All operations in `transition_step` are continuous and differentiable, preserving the computation graph across all periods.

### 3.2. Tensor and I/O contracts (detailed shapes and semantics)

**Dimension notation:**
- `B`: batch size
- `S`: number of stores
- `W`: number of warehouses
- `T`: simulation periods
- `N`: nodes per graph (GNN; typically `N = S`)
- `Ls`, `Lw`: lead times for stores and warehouses

| Component | Tensor/Field | Shape | Semantics |
|-----------|--------------|-------|-----------|
| **State (at reset)** | `inventories["stores"]` | `[B, S]` | Initial inventory at each store |
| | `inventories["warehouses"]` | `[B, W]` | Initial inventory at each warehouse |
| | `pipeline_stores` (optional) | `[B, S, Ls]` | In-transit shipments to stores (if Ls > 0) |
| | `pipeline_warehouses` (optional) | `[B, W, Lw]` | In-transit shipments to warehouses (if Lw > 0) |
| **Demand** | `demands` | `[T, B, S]` | Per-store demand over time |
| **Cost parameters** | `holding_store` | scalar or `[S]` | Holding cost per unit per store; broadcast internally |
| | `underage_store` | scalar or `[S]` | Stockout/underage cost per unit per store |
| | `holding_warehouse` | scalar or `[W]` | Holding cost per unit per warehouse |
| **Lead times** | `lead_times["stores"]` | int | Ls: delay for store shipments |
| | `lead_times["warehouses"]` | int | Lw: delay for warehouse shipments |
| **Store mask** | `store_mask` | `[S]` (bool) | Indicates active stores; broadcast-multiplied with store actions |
| **Actions (per step)** | `stores` | `[B, S]` | Shipment quantities to stores |
| | `warehouses` | `[B, W]` | Replenishment to warehouses (vanilla: may be `[B,1]` then expanded) |
| **Episode outputs** | `total_episode_cost` | `[B]` | Sum of per-step costs over all T periods |
| | `cost_to_report` | `[B]` | Sum of costs over last `max(T - ignore_periods, 1)` periods |

**Key notes:**
- `prepare_batch_for_simulation` reshapes PyG batches from `[B*N, F]` node features to dense `[B, N]` inventories.
- `run_simulation_episode` applies `store_mask` by element-wise multiplication: `actions_stores * store_mask.view(1, -1)`.

### 3.3. Episode mechanics and operation ordering (per period)

**For each period `t` in `[0, T)`:**

1. **Model input construction:**
   - `vanilla`: Concatenate store and warehouse inventories → `[B, S+W]`
   - `gnn`: Reshape `inventory_stores` to nodes `[B*N, F]` and pass with `edge_index`

2. **Action projection:**
   - `vanilla`:
     - Apply `sigmoid` to store and warehouse logits
     - Multiply store actions by `store_mask` (element-wise broadcast)
     - Expand warehouse actions from `[B, 1]` → `[B, W]` if needed
   - `gnn`:
     - Apply `softplus` to channel 0 of each node for store actions
     - Warehouse actions fixed to zeros (current design)

3. **Physics (`transition_step`):**
   - **Warehouse replenishment:** If `Lw > 0`, advance pipeline and add arrivals; else immediate
   - **Feasibility cap:** Compute smooth feasibility factor: `total_wh_inv / (total_ship_req + ε)` → `ratio / (1 + ratio)`
   - **Store shipments:** Multiply requested shipments by feasibility factor; if `Ls > 0`, queue in pipeline
   - **Sales:** Compute `sales = demand - softplus(demand - inventory) / β` (smooth approximation)
   - **Update inventories:** `inventory_stores -= sales`, `inventory_warehouses -= shipped_total`
   - **Compute costs:** `holding_store * end_inv + underage_store * underage + holding_warehouse * wh_inv`

4. **Cost accumulation:**
   - Add `step_cost` to `total_episode_cost` for all periods
   - For `cost_to_report`, sum only the last `effective_periods = max(T - ignore_periods, 1)` steps

**Key parameters:**
- `ignore_periods`: Warm-up periods excluded from reported cost (but gradients still flow through them)
- `lead_times`: Deterministic integer delays; pipeline queues managed as `[B, S/W, L]` tensors
- Smooth operations ensure no gradient breaks across the entire episode

### 3.4. Differentiability guarantees and numerical validation (gradcheck)

**Validation methodology:**
- **Tool:** `torch.autograd.gradcheck` validates the complete path: model → episode → loss
- **Test function:** `test_gradient_flow` in `tests/training/test_end_to_end.py`
- **Configuration:**
  - Double precision (`torch.double`) for numerical stability
  - Finite-difference epsilon: `eps=1e-6`
  - Tolerance: `atol=1e-4`
  - Seed: `torch.manual_seed(0)` for reproducibility
- **Test design:**
  - Wraps a `VanillaPolicy` to inject a single trainable weight matrix
  - Freezes all other parameters to isolate gradient flow through the specific weight
  - Uses a functional simulator with smooth quadratic cost `(stores² + warehouses²).sum()` to avoid non-differentiable kinks
  - Runs a 2-period episode with `B=1`, `S=2`, `W=1`

**Result:** The entire pipeline is end-to-end differentiable under the following assumptions:
- All activation functions are smooth (`sigmoid`, `softplus`)
- The feasibility cap uses `ratio / (1 + ratio)` (smooth sigmoid-like)
- Sales use `softplus`-based approximation instead of hard `min()`
- No discrete operations or hard clamps interrupt the computation graph

### 3.5. Configuration schema (settings + hyperparams)

**Structure:** Experiments combine two YAML files:
1. `configs/settings/*.yml` — problem definition
2. `configs/hyperparams/*.yml` — model/optimizer definition

| File | Field | Type | Description | Default/Example |
|------|-------|------|-------------|-----------------|
| **settings** | `problem_params.n_stores` | int | Number of stores | 3 |
| | `problem_params.n_warehouses` | int | Number of warehouses | 1 |
| | `problem_params.periods` | int | Simulation periods per episode | 50 |
| | `problem_params.ignore_periods` | int | Warm-up periods (excluded from reported cost) | 30 |
| | `data_params.n_samples` | int | Total synthetic samples | 1024 |
| | `data_params.batch_size` | int | Batch size for training | 256 |
| **hyperparams** | `optimizer_params.learning_rate` | float | Optimizer learning rate | 0.0003 |
| | `model_params.architecture` | str | Model type: `vanilla` or `gnn` | `vanilla` |
| | `model_params.layers` | list[int] | Hidden layer widths (vanilla only) | `[128, 128]` |
| | `model_params.hidden_channels` | int | GNN hidden channels (gnn only) | 128 |
| | `model_params.num_layers` | int | GNN message-passing layers (gnn only) | 2 |

**Example minimal config:**

```yaml
# configs/settings/base_setting.yml
problem_params:
  n_stores: 3
  n_warehouses: 1
  periods: 50
  ignore_periods: 30
data_params:
  n_samples: 1024
  batch_size: 256
```

```yaml
# configs/hyperparams/vanilla_mlp.yml
optimizer_params:
  learning_rate: 0.0003
model_params:
  architecture: vanilla
  layers:
    - 128
    - 128
```

### 3.6. Policy interface contract (what `Trainer` expects from `model.forward`)

**`VanillaPolicy`:**
- **Input:** `x: [B, S+W]` — concatenated store and warehouse inventories
- **Output:** `{"stores": [B, S], "warehouses": [B, 1 or W]}` — raw logits
- **Post-processing (by orchestrator):**
  - Apply `sigmoid` to both stores and warehouses
  - Multiply store actions by `store_mask` (broadcast)
  - Expand warehouse actions from `[B, 1]` → `[B, W]` if necessary

**`GNNPolicy`:**
- **Input:** `x: [B*N, F]` node features, `edge_index: [2, E]` graph connectivity
- **Output:** `{"stores": [B*N, C]}` — per-node embeddings (typically C=1)
- **Post-processing (by orchestrator):**
  - Reshape to `[B, N, C]` using `to_dense_batch`
  - Extract channel 0: `actions_stores = node_out[:, :, 0]`
  - Apply `softplus` activation
  - Multiply by `store_mask` (broadcast)
  - Warehouse actions are set to zeros (current design limitation)

**How `prepare_batch_for_simulation` adapts data:**
- **Vanilla:** Directly uses `batch["inventories"]` and `batch["demands"]` as-is
- **GNN:** Calls `to_dense_batch(batch.x, batch.batch)` to convert PyG node features to dense `[B, N]` inventories; reshapes `batch.demands` from `[B*N, T]` or `[B, N, T]` to `[T, B, N]`

### 3.7. Test coverage map (1:1 validation checklist)

| Test File | Test Function | What It Validates |
|-----------|---------------|-------------------|
| **`tests/data/test_datasets.py`** | `test_create_synthetic_data_dict` | Dataset tensor shapes and value ranges |
| | `test_create_pyg_dataset` | PyG graph structure, node features, per-graph attributes |
| **`tests/training/test_engine.py`** | `test_calculate_losses` | Loss normalization formulas and edge cases (zero stores, zero periods) |
| | `test_prepare_vanilla_batch` | Vanilla batch preparation and store mask creation |
| | `test_prepare_gnn_batch` | PyG batch → dense tensor reshaping and demand alignment |
| | `test_run_simulation_episode_logic` | Multi-period cost accumulation, `ignore_periods` logic, mask application |
| | `test_perform_gradient_step` | Gradient clipping and optimizer weight updates |
| **`tests/training/test_trainer.py`** | `test_trainer_integration` | Full epoch/batch loop, scheduler step, optimizer integration |
| **`tests/training/test_end_to_end.py`** | `test_simulation_logic_with_mocks` | Model call count per period, store mask application to actions, cost totals |
| | `test_gradient_flow` | **Numerical gradient check (`gradcheck`)** — end-to-end differentiability validation |

**Critical validation:** `test_gradient_flow` uses `torch.autograd.gradcheck` to confirm that gradients propagate correctly through the entire multi-period simulation, ensuring no detached tensors or non-differentiable operations.

### 3.8. Reproducibility

To ensure deterministic training runs, set seeds before starting training:

```python
import torch, random, numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)  # Optional: enforces deterministic ops
```

**Note:** The gradient check test (`test_gradient_flow`) already sets `torch.manual_seed(0)` for reproducible validation.

### 3.9. Performance notes

- **Gradient clipping:** Applied during training with `grad_clip_norm` (default: 1.0) to prevent exploding gradients
- **Batch sizing:**
  - `vanilla`: Default batch size is `data_params.n_samples` (full dataset)
  - `gnn`: Processes the entire list of graphs in a single batch by default
- **Memory considerations:** Adjust `batch_size` and `n_samples` in settings to balance memory usage and gradient stability
- **Computational complexity per step:**
  - Vanilla: O(B × (S + W) × hidden_width)
  - GNN: O(B × N × E × hidden_channels) where E is edges per graph
- **Known bottlenecks:** Multi-period simulation unrolls T steps; larger T increases memory and compute linearly

### 3.10. Extensibility (adding new policies, dynamics, or costs)

**Adding a new policy:**
1. Implement `forward(...)` following the contract in section 3.6
2. Add a new architecture string to `model_params.architecture` in configs
3. Update `train.py` to instantiate your model class
4. Ensure output shapes match `{"stores": [...], "warehouses": [...]}`

**Modifying simulator dynamics:**
1. Edit `engine/functional.py::transition_step`
2. Maintain smooth, differentiable operations (no hard clamps, discrete switches, or non-differentiable `min`/`max`)
3. Add new fields to `current_state` dict as needed (e.g., `backorders`, `transit_costs`)
4. Propagate new state fields through the episode loop in `training/engine.py::run_simulation_episode`

**Adding new cost terms:**
1. Extend the cost calculation block in `transition_step` (after line 85–98 in `functional.py`)
2. Preserve continuity and differentiability (use smooth approximations)
3. Update `cost_params` dict to include new parameters
4. Pass new parameters through `prepare_batch_for_simulation` and config files

### 3.11. Known limitations (current design)

- **GNN warehouse actions:** In the `gnn` architecture, warehouse actions are fixed to zero; only store-level policies are learned
- **Deterministic lead times:** `lead_times` are fixed integers; stochastic or variable lead times are not modeled
- **No explicit backorders:** Sales use a smooth `softplus`-based approximation; explicit backorder tracking is not implemented
- **Fully connected graphs:** The current GNN setup uses fully connected store graphs; sparse or hierarchical topologies require manual edge_index construction


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