# PSF-HDPO-GNN-Inventory-Optimization

This repository contains a professional-grade implementation of the **Hindsight Differentiable Policy Optimization (HDPO)** framework for inventory management using Graph Neural Networks (GNNs). The system models inventory networks as generic graphs, enabling flexible representation of arbitrary supply chain topologies.

## 1. Core Concepts & Objective

The goal of this project is to implement and extend the research from "Deep Reinforcement Learning for Inventory Networks" (Alvo et al.) using a **differentiable simulator** to train neural network policies via **pathwise gradients**.

### Key Design Principles

- **Generic Graph Topology:** The system models inventory networks as arbitrary directed graphs defined in YAML configuration files. Nodes represent locations (warehouses, stores, distribution centers) with static features, and edges represent possible shipment routes with lead times and costs.

- **Edge GNN Architecture:** The policy is a Graph Neural Network that predicts **edge flows** (shipment quantities) rather than node-level values. This naturally aligns with the physics of inventory networks where decisions are about moving goods between locations.

- **State-Dependent Learning:** At each time step, the GNN receives **dynamic node features** constructed by concatenating static node properties with current simulation state (on-hand inventory, outstanding orders, and current demand). This enables the model to learn true, state-dependent policies that respond to real-time conditions.

- **HDPO Framework:** Unlike traditional RL methods that rely on high-variance estimators (like REINFORCE), HDPO treats the entire simulation as a differentiable computation graph. This allows for stable, low-variance gradient updates by backpropagating directly through the known system dynamics.

- **Optimization Objective:** The framework minimizes a cost function comprising holding costs (for excess inventory), underage costs (for unmet demand), and procurement costs (for external supplier orders). The optimizer finds policy parameters θ that minimize expected total cost over simulation episodes.

- **Feasibility Enforcement Layer (FEL):** The system includes a differentiable feasibility enforcement mechanism that ensures all actions respect inventory constraints while maintaining gradient flow.

- **Differentiable Pipeline:** In-transit inventory is managed through a differentiable pipeline system that tracks orders across multiple lead times, enabling accurate modeling of supply chain delays.

## Current Status

**Project Complete:** The full, end-to-end HDPO training framework with generic graph-based topology is implemented, including FEL, differentiable pipeline, and procurement costs. The system supports hyperparameter sweeps and is ready for experimentation.

## 2. Project Architecture

The codebase is organized into a modular `src` package to ensure clarity, maintainability, and reusability.

```
/
├── configs/
│   └── experiments/                   # Unified experiment configurations
│       ├── S4_OWMS_synthetic.yml      # Example: One Warehouse, Multiple Stores
│       └── ...
├── src/
│   └── hdpo_gnn/                      # The core Python package
│       ├── data/
│       │   └── dataset_manager.py     # Data generation, splitting, and loading
│       ├── engine/
│       │   ├── functional.py          # Core differentiable simulation physics
│       │   ├── fel.py                 # Feasibility Enforcement Layer
│       │   └── pipeline.py            # DifferentiablePipeline for in-transit inventory
│       ├── models/
│       │   ├── factory.py            # Model factory for dynamic model creation
│       │   └── paper_gnn.py           # Paper's GNN policy network
│       ├── training/
│       │   ├── simulation_engine.py  # Simulation orchestration and episode execution
│       │   └── trainer.py             # Training loop coordinator
│       └── utils/
│           └── environment_builder.py # Environment construction from config
├── run_experiment.py                  # Manager script for hyperparameter sweeps
└── train.py                           # Worker script for single training runs
```

### 2.1. `configs/experiments/` - Unified Experiment Configurations

Experiments are now defined in a single unified YAML file that combines:
- **Environment definition:** Graph topology, demand generation, cost parameters, lead times
- **Feature engineering:** Dynamic and static features to include
- **Data handling:** Split policies, scenario counts, episode lengths
- **Model architecture:** GNN parameters, FEL type
- **Training configuration:** Optimizer settings, epochs, early stopping

### 2.2. `src/hdpo_gnn/` - The Core Package

- **`utils/environment_builder.py`:**
  - `Environment`: Builds the supply chain environment from configuration
  - Creates graph topology, node features, edge parameters (lead times, costs)
  - Generates procurement costs and manages edge parameters

- **`data/dataset_manager.py`:**
  - `DatasetManager`: Orchestrates data generation, splitting, and loading
  - `PyGInventoryDataset`: PyTorch Geometric dataset wrapper
  - Generates synthetic scenarios and manages train/dev/test splits

- **`engine/functional.py`:**
  - `transition_step(...)`: Pure functional simulator physics
  - Implements per-node feasibility capping, flow aggregation using `torch_scatter`
  - Smooth sales calculation, cost computation (holding, underage, procurement)
  - Fully differentiable with no gradient breaks

- **`engine/fel.py`:**
  - `apply_fel(...)`: Feasibility Enforcement Layer dispatcher
  - `g1a_full_allocation_proportional(...)`: Full Allocation Proportional FEL for transshipment
  - `g1_proportional_allocation(...)`: Standard proportional allocation
  - Ensures actions respect inventory constraints while maintaining differentiability

- **`engine/pipeline.py`:**
  - `DifferentiablePipeline`: Manages in-transit inventory queues
  - Tracks orders across multiple lead times with device-aware buffer management
  - Methods: `get_arrivals()`, `place_orders()`, `advance_step()`

- **`models/paper_gnn.py`:**
  - `PaperGNNPolicy`: Faithful implementation of the paper's GNN architecture
  - Message-passing layers with configurable width and depth
  - Predicts edge flows from node features

- **`models/factory.py`:**
  - `ModelFactory`: Dynamically creates models based on configuration
  - Calculates input feature dimensions from feature specifications
  - Handles model architecture selection and hyperparameter configuration

- **`training/simulation_engine.py`:**
  - `run_simulation_episode(...)`: **Critical function** that orchestrates multi-period simulation
  - Integrates FEL, pipeline, and feature construction
  - At each time step: constructs dynamic features, calls GNN, applies FEL, steps physics, updates pipeline
  - Returns total episode cost and reported costs (after warmup)

- **`training/trainer.py`:**
  - `Trainer`: Training orchestrator that manages epoch loop, batch processing, validation, and early stopping

### 2.3. `run_experiment.py` - Hyperparameter Sweep Manager

The manager script orchestrates hyperparameter sweeps:
- Reads experiment configuration
- Extracts hyperparameter lists (e.g., learning rates)
- Launches `train.py` as subprocess workers for each hyperparameter combination
- Captures and displays output from each training run
- Provides summary of all runs

### 2.4. `train.py` - Training Worker Script

A worker script for single training runs:
- Accepts learning rate as command-line argument (`--lr`)
- Loads unified experiment configuration
- Initializes environment, dataset manager, model, optimizer, and trainer
- Executes full training process
- Can be run standalone or orchestrated by `run_experiment.py`

## 3. Data Flow & Execution Logic

The end-to-end data flow for a single training step:

1. **Environment Construction:** `Environment` builds graph topology from config, creating node features, edge indices, and edge parameters (lead times, costs).

2. **Data Generation:** `DatasetManager` generates synthetic scenarios with random initial inventories `[B, N]` and demands `[T, B, N]`, masks demands to demand-facing nodes, and creates PyG datasets.

3. **Training Loop:** The `Trainer` receives mini-batches from the DataLoader and processes each batch:
   
   a. `run_simulation_episode` orchestrates the multi-period simulation:
      - **For each time step t:**
        - **Dynamic Feature Construction:** Concatenates static node features with current inventory, outstanding orders (from pipeline), demand, and static parameters (costs, lead times)
        - **GNN Forward Pass:** Model receives `dynamic_x` and predicts raw edge logits
        - **FEL Application:** `apply_fel` enforces feasibility constraints on raw logits
        - **Physics Step:** `transition_step` computes feasible flows, updates inventories, calculates sales, computes costs (holding, underage, procurement)
        - **Pipeline Update:** Places orders into pipeline queues, advances time step, retrieves arrivals
        - **State Update:** Current inventories are updated with arrivals
   
   b. Loss is calculated from accumulated episode costs
   
   c. `loss.backward()` propagates gradients through the entire multi-period episode
   
   d. `optimizer.step()` updates the GNN parameters

### 3.1. End-to-end Computation Graph (Visual Flow)

```
run_experiment.py
   ↓
train.py (worker)
   ↓
Trainer.train()
   ↓
Trainer._run_epoch()
   ↓
run_simulation_episode
   ↓
   ├─→ [for t in episode_length]:
   │      ↓
   │   ┌─────────────────────────────────────────────────┐
   │   │ DYNAMIC FEATURE CONSTRUCTION                    │
   │   │ dynamic_x = [static_features, inventory_t,      │
   │   │              outstanding_orders_t, demand_t,   │
   │   │              holding_cost, underage_cost,       │
   │   │              lead_time]                         │
   │   └─────────────────────────────────────────────────┘
   │      ↓
   │   model.forward(dynamic_x, edge_index) → raw_logits [GRADIENTS FLOW]
   │      ↓
   │   fel.apply_fel(raw_logits, inventory, ...) → feasible_flows [GRADIENTS FLOW]
   │      ↓
   │   pipeline.get_arrivals() → arrivals [GRADIENTS FLOW]
   │      ↓
   │   transition_step(inventory + arrivals, flows, demand, ...) [GRADIENTS FLOW]
   │      ↓
   │   (next_inventory, step_cost) [GRADIENTS ACCUMULATE]
   │      ↓
   │   pipeline.place_orders(orders, lead_times) [GRADIENTS FLOW]
   │      ↓
   │   pipeline.advance_step() [GRADIENTS FLOW]
   │      ↓
   │   total_episode_cost += step_cost
   │
   ↓
loss = total_episode_cost.mean()
   ↓
loss.backward()  ← BACKPROP THROUGH ENTIRE MULTI-PERIOD EPISODE
   ↓
optimizer.step()  ← UPDATE GNN PARAMETERS
```

**Key differentiability guarantees:**
- Sales calculation uses smooth `softplus`-based approximation: `sales = inventory - softplus(inventory - demand) / β`
- Feasibility enforcement uses differentiable capping: `capped_factor = inventory / (requested_flow + inventory + ε)`
- FEL maintains gradient flow through proportional allocation
- Pipeline operations use device-aware buffers that preserve gradients
- All operations in `transition_step` use `torch.scatter_add` to preserve gradients

### 3.2. Tensor Shapes and Semantics

**Dimension notation:**
- `B`: batch size (number of samples)
- `N`: number of nodes in the graph (total locations)
- `E`: number of edges in the graph (shipment routes)
- `T`: simulation periods (time horizon)
- `L`: maximum lead time (for outstanding orders feature)

| Component | Tensor/Field | Shape | Semantics |
|-----------|--------------|-------|-----------|
| **Graph Topology** | `static_node_features` | `[N, F_static]` | Static node properties (node type encoding) |
| | `edge_index` | `[2, E]` | Graph connectivity in COO format |
| | `edge_lead_times` | `[E]` | Lead time for each edge |
| **Initial State** | `initial_inventory` | `[B, N]` | Starting inventory at each node |
| **Demand** | `demands` | `[B, T, N]` | Per-node demand over time |
| **Cost Parameters** | `holding_costs` | `[B, N]` | Holding cost per unit per period per node |
| | `underage_costs` | `[B, N]` | Underage/stockout cost per unit per period per node |
| | `procurement_costs` | `[E]` | Procurement cost per unit per edge (supplier edges) |
| **Dynamic Features (per step)** | `dynamic_x` | `[B*N, F_total]` | Concatenation of static features, inventory, outstanding orders, demand, static parameters |
| **Pipeline State** | `pipeline.queues[lt]` | `[B, N, lt]` | In-transit inventory for lead time `lt` |
| **Actions (per step)** | `feasible_edge_flows` | `[B, E]` | Feasibility-capped shipment quantities |
| **Episode Outputs** | `total_episode_cost` | `[B]` | Sum of costs over all T periods |
| | `reported_costs` | `[B]` | Sum of costs over last `max(T - warmup_periods, 1)` periods |

## 4. Configuration Schema

### 4.1. Unified Experiment Configuration

Example: `configs/experiments/S4_OWMS_synthetic.yml`

```yaml
# 1. METADATA & EXPERIMENT SETUP
experiment:
  name: "S4_OWMS_Backlogged_Transshipment_50S_128Scenarios"
  setting_id: "S4"
  seed: 42
  device: "cuda"

# 2. ENVIRONMENT DEFINITION
environment:
  topology:
    generator: "from_rules"
    rules:
      node_counts: { warehouse: 1, store: 50 }
      edge_policy: "warehouse_to_all_stores"
  demand:
    source: "synthetic"
    synthetic_data_config:
      distribution: "normal"
      mean_range: [2.5, 7.5]
      cv_range: [0.25, 0.5]
      correlation: 0.5
  parameters:
    nodes:
      stores:
        holding_cost: { sampling_method: "fixed", value: 1.0 }
        underage_cost: { sampling_method: "fixed", value: 9.0 }
      warehouses:
        holding_cost: { sampling_method: "fixed", value: 0.0 }
    edges:
      supplier_to_warehouse:
        lead_time: { sampling_method: "fixed", value: 3 }
        procurement_cost: { sampling_method: "fixed", value: 0.0 }
      warehouse_to_store:
        lead_time: { sampling_method: "fixed", value: 4 }
  dynamics:
    unmet_demand_assumption: "backlogged"
    is_transshipment: true
    objective: "minimize_cost"

# 3. FEATURE ENGINEERING
features:
  dynamic: ["inventory_on_hand", "outstanding_orders"]
  static: ["holding_cost", "underage_cost", "lead_time"]
  demand_history: { past_periods: 0 }
  time_features: null

# 4. DATA HANDLING
data:
  split_policy: "random"
  scenarios: { train: 128, dev: 128, test: 8192 }
  episode_length: { train: 50, dev: 100, test: 5000 }
  warmup_periods: { train: 30, dev: 60, test: 3000 }
  batch_size: 128

# 5. MODEL ARCHITECTURE
model:
  architecture_class: "gnn"
  gnn_params:
    fel_type: "g1a"  # Full Allocation Proportional for transshipment
    message_passing_layers: 2
    module_layers: 2
    module_width: 32
    weight_sharing: true

# 6. TRAINING PROCESS (HDPO)
training:
  optimizer:
    name: "adam"
    lr: 0.01  # Can be a list for hyperparameter sweeps: [0.0001, 0.001, 0.01]
    betas: [0.9, 0.999]
  epochs: 100
  early_stopping_patience: 20
  softplus_bias: 5.0
  gradient_clip_norm: 1.0
```

**Key Configuration Sections:**

1. **`experiment`**: Metadata, random seed, compute device
2. **`environment`**: Graph topology generation, demand source, cost parameters, lead times, dynamics
3. **`features`**: Dynamic features (inventory, outstanding orders), static features (costs, lead times), optional demand history
4. **`data`**: Split policy, scenario counts, episode lengths, warmup periods, batch size
5. **`model`**: Architecture class, GNN hyperparameters (layers, width), FEL type
6. **`training`**: Optimizer settings (learning rate can be a list for sweeps), epochs, early stopping, softplus bias, gradient clipping

## 5. How to Use

### 5.1. Setup

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd gnn_from_scratch

# Activate your conda environment with PyTorch and PyTorch Geometric
conda activate pyg_env

# Install the package in editable mode
pip install -e .
```

### 5.2. Running a Single Training Run

Run a single training session with a specific learning rate:

```bash
python train.py configs/experiments/S4_OWMS_synthetic.yml --lr 0.001
```

Add `-v` or `--verbose` for DEBUG-level logging:

```bash
python train.py configs/experiments/S4_OWMS_synthetic.yml --lr 0.001 -v
```

### 5.3. Running Hyperparameter Sweeps

The recommended way to run experiments is through the manager script, which supports hyperparameter sweeps:

```bash
python run_experiment.py configs/experiments/S4_OWMS_synthetic.yml -v
```

**What happens:**
1. `run_experiment.py` loads the experiment configuration
2. Extracts the learning rate list from `training.optimizer.lr` (if it's a list)
3. For each learning rate:
   - Launches `train.py` as a subprocess with `--lr <learning_rate>`
   - Captures and displays all output in real-time
   - Waits for completion before proceeding to next learning rate
4. Provides a summary of all runs at the end

**To specify learning rates in the config:**

```yaml
training:
  optimizer:
    lr: [0.0001, 0.001, 0.01]  # List of learning rates to sweep
```

**Training Output:**
- Real-time progress bars for each training run
- Epoch-level loss reporting (training and validation)
- Automatic early stopping based on validation loss
- Gradient clipping for stability
- Summary of all hyperparameter combinations at the end

### 5.4. Hyperparameter Sweep Configuration

To run a hyperparameter sweep, modify the learning rate in your experiment config to be a list:

```yaml
training:
  optimizer:
    lr: [0.0001, 0.001, 0.01]  # Will run 3 separate training sessions
```

Then run:

```bash
python run_experiment.py configs/experiments/S4_OWMS_synthetic.yml
```

Each learning rate will trigger a complete training run from start to finish.

## 6. Training & Hyperparameters

### 6.1. Sensitivity and Tuning Guidance

Training differentiable physics models like this is **highly sensitive** to problem difficulty and hyperparameters. The two most critical parameters are:

1. **`episode_length.train` (simulation horizon):** Longer horizons create deeper computation graphs, making optimization harder
2. **`training.optimizer.lr` (learning rate):** Controls the magnitude of parameter updates

### 6.2. Recommended Starting Point

**Users should start with a short `episode_length.train` (e.g., 50) and a conservative `learning_rate` (e.g., 0.001) to establish a stable learning baseline before attempting to solve problems with longer time horizons.**

Example progression:
1. Start: `episode_length.train: 50`, `lr: 0.001`
2. Once stable: Increase to `episode_length.train: 100`, keep same learning rate
3. Advanced: `episode_length.train: 500`, reduce to `lr: 0.0001`

### 6.3. Key Hyperparameters

| Parameter | Location | Typical Range | Notes |
|-----------|----------|---------------|-------|
| `episode_length.train` | `data` section | 50-500 | Start small (50-100) for initial experiments |
| `training.optimizer.lr` | `training` section | 0.0001-0.01 | Lower is safer; use 0.001 as default |
| `data.batch_size` | `data` section | 64-256 | Larger batches = more stable gradients |
| `data.scenarios.train` | `data` section | 128-8192 | More samples = better generalization |
| `training.gradient_clip_norm` | `training` section | 0.5-2.0 | Prevents exploding gradients |
| `model.gnn_params.module_width` | `model` section | 32-128 | GNN capacity; 32-64 is a good default |
| `model.gnn_params.message_passing_layers` | `model` section | 2-4 | More layers = more expressive, but slower |
| `training.softplus_bias` | `training` section | 3.0-10.0 | Controls softplus curvature; 5.0 is default |

### 6.4. Feature Engineering

The system supports flexible feature engineering through the `features` section:

- **Dynamic features:**
  - `inventory_on_hand`: Current on-hand inventory at each node
  - `outstanding_orders`: In-transit inventory broken down by lead time (shape `[B, N, L]`)

- **Static features:**
  - `holding_cost`: Per-node holding cost
  - `underage_cost`: Per-node underage cost
  - `lead_time`: Maximum incoming lead time for each node

- **Optional features:**
  - `demand_history`: Past demand values (if `past_periods > 0`)
  - `time_features`: Time-based features (if not null)

### 6.5. FEL Types

The Feasibility Enforcement Layer type is specified in `model.gnn_params.fel_type`:

- **`g1a`**: Full Allocation Proportional - Ensures transshipment nodes allocate ALL available inventory
- **`g1b`**: DC-Bid Proportional - (Not fully implemented, falls back to `g1`)
- **`g1`** (default): Standard Proportional Allocation - Scales down outgoing orders if sum exceeds available inventory

### 6.6. Debugging Training Issues

If loss is not decreasing:
1. **Reduce `episode_length.train`** to simplify the problem
2. **Lower `training.optimizer.lr`** to prevent instability
3. **Check gradient norms** (should be non-zero but not exploding)
4. **Verify data generation** (check that demands and inventories are reasonable)

If loss diverges (NaN or explodes):
1. **Reduce `training.optimizer.lr`** immediately
2. **Increase `training.gradient_clip_norm`** for stronger clipping
3. **Check for numerical issues** in cost parameters (very large underage costs can cause instability)
4. **Reduce `training.softplus_bias`** if softplus is causing issues

## 7. Key Components

### 7.1. Feasibility Enforcement Layer (FEL)

The FEL ensures that all actions respect inventory constraints:
- Takes raw edge logits from the GNN
- Applies softplus to make them non-negative
- Scales down outgoing flows if they exceed available inventory
- For transshipment nodes (with `fel_type: "g1a"`), ensures full allocation of available inventory
- Maintains differentiability throughout

### 7.2. Differentiable Pipeline

The pipeline manages in-transit inventory:
- Tracks orders across multiple lead times using separate queues
- Each queue is a registered buffer: `[B, N, lead_time]`
- `get_arrivals()`: Returns inventory arriving at current time step
- `place_orders()`: Places new orders into appropriate lead-time queues
- `advance_step()`: Shifts queues forward in time
- All operations are device-aware and preserve gradients

### 7.3. Cost Function

The cost function includes three components:
- **Holding cost:** `holding_cost[i] * max(0, inventory[i])` for each node
- **Underage cost:** `underage_cost[i] * max(0, demand[i] - sales[i])` for each node
- **Procurement cost:** `procurement_cost[e] * flow[e]` for supplier edges

All costs are computed per time step and accumulated over the episode.

## 8. Extensibility

### 8.1. Adding New Graph Topologies

Modify the `environment.topology` section in your experiment config. The system supports:
- Rule-based generation (`generator: "from_rules"`)
- Custom node counts and edge policies
- Per-node-type parameter specification

### 8.2. Modifying Simulation Physics

Edit `src/hdpo_gnn/engine/functional.py::transition_step`. Maintain smooth, differentiable operations (no hard clamps, discrete switches, or non-differentiable `min`/`max`). Use `torch.nn.functional.softplus` for smooth approximations.

### 8.3. Adding New Cost Terms

Extend the cost calculation in `transition_step` (after inventory updates). Add new parameters to `environment.parameters` in the experiment config and propagate them through the environment builder.

### 8.4. Adding New Features

Extend the feature construction in `simulation_engine.py`:
1. Add feature name to `features.dynamic` or `features.static` in config
2. Update `ModelFactory._calculate_node_feature_size()` to account for new feature
3. Add feature construction logic in `run_simulation_episode()`

## 9. Known Limitations

- **Deterministic Dynamics:** Lead times and demands are deterministic (no stochasticity in the simulator)
- **Single-Period Decisions:** The GNN makes decisions at each time step independently (no explicit planning horizon)
- **Fixed Topology:** Graph structure is fixed during training (no dynamic graph changes)

## 10. References

- Alvo, M., et al. "Deep Reinforcement Learning for Inventory Networks"
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Hindsight Differentiable Policy Optimization (HDPO) framework

---

**Project Status:** Production-ready. The system is fully implemented with FEL, differentiable pipeline, procurement costs, and hyperparameter sweep support. Ready for research and experimentation on inventory optimization problems with arbitrary graph topologies.
