# PSF-HDPO-GNN-Inventory-Optimization

This repository contains a professional-grade implementation of the **Hindsight Differentiable Policy Optimization (HDPO)** framework for inventory management using Graph Neural Networks (GNNs). The system models inventory networks as generic graphs, enabling flexible representation of arbitrary supply chain topologies.

## 1. Core Concepts & Objective

The goal of this project is to implement and extend the research from "Deep Reinforcement Learning for Inventory Networks" (Alvo et al.) using a **differentiable simulator** to train neural network policies via **pathwise gradients**.

### Key Design Principles

- **Generic Graph Topology:** The system models inventory networks as arbitrary directed graphs defined in YAML configuration files. Nodes represent locations (warehouses, stores, distribution centers) with static features (e.g., `has_external_supply`, `is_demand_facing`), and edges represent possible shipment routes.

- **Edge GNN Architecture:** The policy is a Graph Neural Network that predicts **edge flows** (shipment quantities) rather than node-level values. This naturally aligns with the physics of inventory networks where decisions are about moving goods between locations.

- **State-Dependent Learning:** At each time step, the GNN receives **dynamic node features** constructed by concatenating static node properties with current simulation state (on-hand inventory and current demand). This enables the model to learn true, state-dependent policies that respond to real-time conditions.

- **HDPO Framework:** Unlike traditional RL methods that rely on high-variance estimators (like REINFORCE), HDPO treats the entire simulation as a differentiable computation graph. This allows for stable, low-variance gradient updates by backpropagating directly through the known system dynamics.

- **Optimization Objective:** The framework minimizes a cost function comprising holding costs (for excess inventory) and underage costs (for unmet demand). The optimizer finds policy parameters θ that minimize expected total cost over simulation episodes.

## Current Milestone

**Project Complete:** The full, end-to-end HDPO training framework with generic graph-based topology is implemented, numerically validated via `torch.autograd.gradcheck`, and ready for experimentation.

## 2. Project Architecture

The codebase is organized into a modular `src` package to ensure clarity, maintainability, and reusability.

```
/
├── configs/
│   ├── settings/                    # Problem definitions (graph topology, simulation params)
│   │   ├── graph_simple.yml         # Example: 1 warehouse → 3 stores
│   │   └── graph_topology_example.yml
│   └── hyperparams/                 # Model and optimizer configurations
│       ├── gnn_basic.yml
│       └── gnn_conservative.yml
├── src/
│   └── hdpo_gnn/                    # The core Python package
│       ├── data/
│       │   └── datasets.py          # Synthetic data generation and PyG dataset creation
│       ├── engine/
│       │   └── functional.py        # Core differentiable simulation physics
│       ├── models/
│       │   └── gnn.py               # Edge GNN policy network
│       ├── training/
│       │   ├── engine.py            # Simulation orchestration and loss calculation
│       │   └── trainer.py           # Training loop coordinator
│       └── utils/
│           ├── config_loader.py     # OmegaConf-based configuration loading
│           └── graph_parser.py      # YAML graph topology parser
├── tests/                           # Comprehensive test suite
│   ├── data/
│   │   └── test_datasets.py
│   ├── models/
│   │   └── test_gnn.py
│   ├── training/
│   │   ├── test_engine.py
│   │   └── test_end_to_end.py       # Includes numerical gradient check
│   └── utils/
│       └── test_graph_parser.py
└── train.py                         # Main entry point for experiments
```

### 2.1. `configs/` - Experiment Definitions

Experiments are defined by combining two YAML files:
1. **Settings file** (`configs/settings/*.yml`): Defines the graph topology, simulation parameters, and data generation settings
2. **Hyperparameters file** (`configs/hyperparams/*.yml`): Defines model architecture and optimizer settings

### 2.2. `src/hdpo_gnn/` - The Core Package

- **`utils/graph_parser.py`:**
  - `parse_graph_topology(config)`: Parses YAML graph definition into PyTorch tensors (`node_features` and `edge_index`)
  - `validate_graph_topology(config)`: Validates graph structure and feature consistency
  - `get_node_type_counts(config)`: Extracts node type statistics for analysis

- **`data/datasets.py`:**
  - `create_synthetic_data_dict(config)`: Generates a dictionary of PyTorch tensors representing simulation scenarios. Uses the graph parser to extract topology, then generates random initial inventories and demands. Demands are masked to only apply to demand-facing nodes.
  - `create_pyg_dataset(data, config)`: Converts the synthetic data dictionary into a list of `torch_geometric.data.Data` graphs for the GNN model.

- **`engine/functional.py`:**
  - `transition_step(current_inventories, edge_flows, node_features, edge_index, demand_t, cost_params)`: Pure functional simulator physics. Implements per-node feasibility capping, flow aggregation using `torch_scatter`, smooth sales calculation, and cost computation. Fully differentiable with no gradient breaks.

- **`models/gnn.py`:**
  - `GNNPolicy`: Edge GNN that uses GCN layers to create node embeddings, then concatenates source and destination node embeddings for each edge and passes them through an MLP to predict edge flows.

- **`training/engine.py`:**
  - `prepare_batch_for_simulation(batch, device)`: Prepares PyG Batch objects for simulation, extracting dynamic data and static graph structure.
  - `run_simulation_episode(model, pyg_batch, data_for_reset, periods, ignore_periods)`: **Critical function** that orchestrates the multi-period simulation. At each time step, it constructs dynamic node features by concatenating static features with current inventory and demand, calls the GNN model, and steps the simulation forward using `transition_step`.
  - `calculate_losses(...)`: Computes training loss from episode costs.
  - `perform_gradient_step(...)`: Executes gradient clipping and optimizer update.

- **`training/trainer.py`:**
  - `Trainer`: Training orchestrator that manages the epoch loop, batch processing, and scheduler updates.

### 2.3. `train.py` - The Entry Point

A lightweight script responsible for:
1. Parsing command-line arguments (config paths, epochs)
2. Loading and merging configurations using OmegaConf
3. Generating synthetic data and creating PyG dataset
4. Instantiating the GNN model, optimizer, and trainer
5. Launching the training process

## 3. Data Flow & Execution Logic

The end-to-end data flow for a single training step:

1. **Graph Definition:** User defines the supply chain topology in a YAML file with nodes (features: `has_external_supply`, `is_demand_facing`) and edges (shipment routes).

2. **Data Generation:** `create_synthetic_data_dict` parses the graph topology, generates random initial inventories `[B, N]` and demands `[T, B, N]`, and masks demands to only apply to demand-facing nodes.

3. **PyG Dataset Creation:** `create_pyg_dataset` creates one `torch_geometric.data.Data` object per sample, attaching static node features, edge connectivity, initial inventories, and per-sample demands.

4. **Training Loop:** The `Trainer` receives mini-batches from the DataLoader and processes each batch:
   
   a. `prepare_batch_for_simulation` extracts the batch structure
   
   b. `run_simulation_episode` orchestrates the multi-period simulation:
      - **For each time step t:**
        - **Dynamic Feature Construction (KEY TO LEARNING):** Concatenates static node features with current inventory and current demand to create `dynamic_x = [static_features, inventory_t, demand_t]`
        - **GNN Forward Pass:** Model receives `dynamic_x` and predicts edge flows
        - **Physics Step:** `transition_step` computes feasibility-capped flows, updates inventories, calculates sales, and computes costs
        - **State Update:** Current inventories are updated for next time step
   
   c. Loss is calculated from accumulated episode costs
   
   d. `loss.backward()` propagates gradients through the entire multi-period episode
   
   e. `optimizer.step()` updates the GNN parameters

### 3.1. End-to-end Computation Graph (Visual Flow)

```
train.py
   ↓
Trainer.train()
   ↓
Trainer._train_batch()
   ↓
prepare_batch_for_simulation  ← extracts batch structure
   ↓
run_simulation_episode
   ↓
   ├─→ [for t in periods]:
   │      ↓
   │   ┌─────────────────────────────────────────────────┐
   │   │ DYNAMIC FEATURE CONSTRUCTION (KEY TO LEARNING)  │
   │   │ dynamic_x = [static_features, inventory_t,      │
   │   │              demand_t]                           │
   │   └─────────────────────────────────────────────────┘
   │      ↓
   │   model.forward(dynamic_x, edge_index) → edge_flows  [GRADIENTS FLOW]
   │      ↓
   │   softplus(edge_flows) → feasible_flows              [GRADIENTS FLOW]
   │      ↓
   │   transition_step(inventory_t, flows, demand_t, ...) [GRADIENTS FLOW]
   │      ↓
   │   (next_inventory, step_cost)
   │      ↓
   │   total_episode_cost += step_cost                    [GRADIENTS ACCUMULATE]
   │
   ↓
calculate_losses(total_episode_cost, ...) → loss_for_backward
   ↓
loss.backward()  ← BACKPROP THROUGH ENTIRE MULTI-PERIOD EPISODE
   ↓
optimizer.step()  ← UPDATE GNN PARAMETERS
```

**Key differentiability guarantees:**
- Sales calculation uses smooth `softplus`-based approximation: `sales = inventory - softplus(inventory - demand) / β`
- Feasibility capping uses numerically stable formula: `capped_factor = inventory / (requested_flow + inventory + ε)`
- All operations in `transition_step` use `torch.scatter_add` (functional form) to preserve gradients
- Dynamic feature construction at each time step enables state-dependent learning

### 3.2. Tensor Shapes and Semantics

**Dimension notation:**
- `B`: batch size (number of samples)
- `N`: number of nodes in the graph (total locations)
- `E`: number of edges in the graph (shipment routes)
- `T`: simulation periods (time horizon)
- `F`: number of static node features (typically 2: `has_external_supply`, `is_demand_facing`)

| Component | Tensor/Field | Shape | Semantics |
|-----------|--------------|-------|-----------|
| **Graph Topology** | `node_features` | `[N, F]` | Static node properties (same for all samples) |
| | `edge_index` | `[2, E]` | Graph connectivity in COO format |
| **Initial State** | `inventories` | `[B, N]` | Starting inventory at each node |
| **Demand** | `demands` | `[T, B, N]` | Per-node demand over time (masked for non-demand-facing nodes) |
| **Cost Parameters** | `holding_store` | scalar | Holding cost per unit per period |
| | `underage_store` | scalar | Underage/stockout cost per unit per period |
| | `holding_warehouse` | scalar | Warehouse holding cost per unit per period |
| **Dynamic Features (per step)** | `dynamic_x` | `[B*N, F+2]` | Concatenation of static features, current inventory, current demand |
| **Actions (per step)** | `edge_flows` | `[B, E]` | Predicted shipment quantities on each edge |
| **Episode Outputs** | `total_episode_cost` | `[B]` | Sum of costs over all T periods |
| | `cost_to_report` | `[B]` | Sum of costs over last `max(T - ignore_periods, 1)` periods |

**Key notes:**
- PyG batches graphs by stacking: `[B*N, F]` for node features, `[B*E, 1]` for edge predictions
- `run_simulation_episode` reshapes these back to standard batch format `[B, N]` and `[B, E]`
- Dynamic feature construction happens **at every time step** using current simulation state

## 4. Configuration Schema

### 4.1. Settings File (Graph Topology and Simulation Parameters)

Example: `configs/settings/graph_simple.yml`

```yaml
# Problem definition: graph topology and simulation parameters
problem_params:
  # Simulation horizon (number of time periods)
  periods: 10
  
  # Warm-up periods excluded from reported cost (but gradients still flow)
  ignore_periods: 0
  
  # Graph topology definition
  graph:
    nodes:
      # Node 0: Warehouse (has external supply, not demand-facing)
      - id: 0
        type: 'warehouse'
        features:
          has_external_supply: 1
          is_demand_facing: 0
      
      # Nodes 1-3: Stores (no external supply, demand-facing)
      - id: 1
        type: 'store'
        features:
          has_external_supply: 0
          is_demand_facing: 1
      - id: 2
        type: 'store'
        features:
          has_external_supply: 0
          is_demand_facing: 1
      - id: 3
        type: 'store'
        features:
          has_external_supply: 0
          is_demand_facing: 1
    
    # Edges define possible shipment routes (source → destination)
    edges:
      - [0, 1]  # Warehouse → Store 1
      - [0, 2]  # Warehouse → Store 2
      - [0, 3]  # Warehouse → Store 3

# Data generation parameters
data_params:
  n_samples: 1000        # Number of training samples
  batch_size: 256        # Mini-batch size for SGD

# Cost parameters
cost_params:
  holding_store: 1.0     # Cost per unit of inventory held per period
  underage_store: 9.0    # Cost per unit of unmet demand per period
  holding_warehouse: 0.5 # Warehouse holding cost per unit per period
```

**Node Features Explained:**
- `has_external_supply`: 1 if the node can receive unlimited external supply (e.g., warehouse), 0 otherwise
- `is_demand_facing`: 1 if the node experiences customer demand, 0 otherwise (used to mask demand tensor)

**Edges:** Define the directed graph structure. Each edge `[source, dest]` represents a possible shipment route.

### 4.2. Hyperparameters File (Model and Optimizer Settings)

Example: `configs/hyperparams/gnn_conservative.yml`

```yaml
# Optimizer configuration
optimizer_params:
  learning_rate: 0.00003  # Conservative learning rate for stable training

# Model architecture
model_params:
  hidden_channels: 64     # GNN hidden layer width
  num_layers: 2           # Number of GCN message-passing layers

# Training configuration
training_params:
  grad_clip_norm: 1.0     # Gradient clipping threshold
```

## 5. How to Use

### 5.1. Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Sancauid/PSF-HDPO-GNN-Inventory-Optimization.git
cd PSF-HDPO-GNN-Inventory-Optimization

# Activate your conda environment with PyTorch and PyTorch Geometric
conda activate pyg_env

# Install the package in editable mode
pip install -e .

# Install testing dependencies
pip install pytest
```

### 5.2. Running Tests

Verify the integrity of all components by running the comprehensive test suite:

```bash
pytest tests/ -v
```

The test suite includes:
- Unit tests for all core components
- Integration tests for the training pipeline
- **Numerical gradient check** (`torch.autograd.gradcheck`) that validates end-to-end differentiability

### 5.3. Running Training

Launch a training run from the root directory:

```bash
# Example: Train on the simple graph topology for 100 epochs
python train.py configs/settings/graph_simple.yml configs/hyperparams/gnn_conservative.yml --epochs 100
```

**Training Output:**
- Epoch-level loss reporting
- Automatic learning rate scheduling (ReduceLROnPlateau)
- Progress bars for batch processing

## 6. Training & Hyperparameters

### 6.1. Sensitivity and Tuning Guidance

Training differentiable physics models like this is **highly sensitive** to problem difficulty and hyperparameters. The two most critical parameters are:

1. **`periods` (simulation horizon):** Longer horizons create deeper computation graphs, making optimization harder
2. **`learning_rate`:** Controls the magnitude of parameter updates

### 6.2. Recommended Starting Point

**Users should start with a short `periods` (e.g., 10-20) and a conservative `learning_rate` (e.g., 1e-4 to 3e-5) to establish a stable learning baseline before attempting to solve problems with longer time horizons.**

Example progression:
1. Start: `periods: 10`, `learning_rate: 0.00003` (3e-5)
2. Once stable: Increase to `periods: 20`, keep same learning rate
3. Advanced: `periods: 50`, reduce to `learning_rate: 0.00001` (1e-5)

### 6.3. Key Hyperparameters

| Parameter | Location | Typical Range | Notes |
|-----------|----------|---------------|-------|
| `periods` | settings file | 10-50 | Start small (10-20) for initial experiments |
| `learning_rate` | hyperparams file | 1e-5 to 1e-4 | Lower is safer; use 3e-5 as default |
| `batch_size` | settings file | 128-512 | Larger batches = more stable gradients |
| `n_samples` | settings file | 1000-5000 | More samples = better generalization |
| `grad_clip_norm` | hyperparams file | 0.5-2.0 | Prevents exploding gradients |
| `hidden_channels` | hyperparams file | 32-128 | GNN capacity; 64 is a good default |

### 6.4. Debugging Training Issues

If loss is not decreasing:
1. **Reduce `periods`** to simplify the problem
2. **Lower `learning_rate`** to prevent instability
3. **Check gradient norms** (should be non-zero but not exploding)
4. **Verify data generation** (run tests to ensure demands and inventories are reasonable)

If loss diverges (NaN or explodes):
1. **Reduce `learning_rate`** immediately
2. **Increase `grad_clip_norm`** for stronger clipping
3. **Check for numerical issues** in cost parameters (very large underage costs can cause instability)

## 7. Test Coverage

The project is rigorously tested with a comprehensive suite that validates:

- **Data Generation:** Tensor shapes, value ranges, demand masking
- **Graph Parsing:** YAML topology parsing, node feature extraction, edge index construction
- **GNN Model:** Forward pass shapes, edge flow predictions
- **Simulation Physics:** Feasibility capping, flow aggregation, sales calculation, cost computation
- **Training Pipeline:** Batch preparation, episode orchestration, loss calculation, gradient flow

**Critical Validation:** The test suite includes a full numerical gradient check using `torch.autograd.gradcheck` (in `tests/training/test_end_to_end.py`) that verifies the end-to-end differentiability of the entire simulation pipeline. This confirms that gradients propagate correctly through:
- Dynamic feature construction
- GNN forward pass
- Multi-period simulation loop
- Smooth feasibility capping and sales calculations
- Cost accumulation

All tests pass, confirming the system is mathematically sound and ready for experimentation.

## 8. Extensibility

### 8.1. Adding New Graph Topologies

Create a new settings file in `configs/settings/` with your desired node and edge structure. The system supports arbitrary directed graphs.

### 8.2. Modifying Simulation Physics

Edit `src/hdpo_gnn/engine/functional.py::transition_step`. Maintain smooth, differentiable operations (no hard clamps, discrete switches, or non-differentiable `min`/`max`). Use `torch.nn.functional.softplus` for smooth approximations.

### 8.3. Adding New Cost Terms

Extend the cost calculation in `transition_step` (after inventory updates). Add new parameters to `cost_params` in the settings file and propagate them through the data pipeline.

## 9. Known Limitations

- **Deterministic Dynamics:** Lead times and demands are deterministic (no stochasticity in the simulator)
- **Single-Period Decisions:** The GNN makes decisions at each time step independently (no explicit planning horizon)
- **Holding Cost Simplification:** All nodes use the same holding cost structure (can be extended to per-node costs)

## 10. References

- Alvo, M., et al. "Deep Reinforcement Learning for Inventory Networks"
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Hindsight Differentiable Policy Optimization (HDPO) framework

---

**Project Status:** Production-ready. The system is fully implemented, tested, and validated. Ready for research and experimentation on inventory optimization problems with arbitrary graph topologies.
