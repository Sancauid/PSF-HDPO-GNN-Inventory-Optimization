from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch

from src.hdpo_gnn.engine.functional import transition_step


def prepare_batch_for_simulation(
    batch: Batch, device: torch.device
) -> Tuple[Dict[str, Any], Batch]:
    """
    Prepares a PyG Batch for simulation, providing both dynamic batched data
    and static, un-batched graph structure.
    """
    batch = batch.to(device)
    B = batch.num_graphs
    N = batch.num_nodes // B
    E = batch.edge_index.shape[1] // B

    # Dynamic, per-sample data
    inventories = batch.initial_inventory.view(B, N)
    
    # --- ROBUST DEMAND RESHAPING ---
    if batch.demands.dim() == 3 and batch.demands.shape[0] == B:
        # Shape is already [B, T, N], just permute it
        demands = batch.demands.permute(1, 0, 2)
    elif batch.demands.dim() == 2:
        # Shape is [B*T, N], need to infer T and reshape
        T = batch.demands.shape[0] // B
        demands = batch.demands.view(B, T, N).permute(1, 0, 2)
    else:
        raise ValueError(f"Unsupported demands shape in PyG batch: {batch.demands.shape}")

    # Static, single-graph data
    static_node_features = batch.x[:N, :]
    static_edge_index = batch.edge_index[:, :E]

    data_for_reset = {
        "inventories": inventories,
        "demands": demands,
        "node_features": static_node_features,
        "edge_index": static_edge_index,
        "cost_params": {
            "holding_store": batch.holding_store[0],
            "underage_store": batch.underage_store[0],
            "holding_warehouse": batch.holding_warehouse[0],
        }
    }
    
    return data_for_reset, batch


def run_simulation_episode(
    model: nn.Module,
    pyg_batch: Batch,
    data_for_reset: Dict[str, Any],
    periods: int,
    ignore_periods: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = pyg_batch.num_graphs
    E_per_graph = data_for_reset["edge_index"].shape[1]
    current_inventories = data_for_reset["inventories"].clone()
    step_costs: list[torch.Tensor] = []
    
    # Static features from the batch, shape [B*N, F_static]
    static_node_features = pyg_batch.x

    for t in range(periods):
        # Get current demand for this time step, shape [B, N]
        demand_t = data_for_reset["demands"][t]

        # Flatten dynamic state to match PyG's batching format [B*N, 1]
        # The '-1' tells PyTorch to automatically infer the total number of elements (B*N)
        inv_flat = current_inventories.reshape(-1, 1)
        demand_flat = demand_t.reshape(-1, 1)

        # Construct the full dynamic feature matrix for the model
        # Shape: [B*N, F_static + 2]
        dynamic_x = torch.cat([static_node_features, inv_flat, demand_flat], dim=1)

        # Call the model with the state-dependent features
        outputs = model(dynamic_x, pyg_batch.edge_index)
        edge_flows = torch.nn.functional.softplus(outputs["flows"]).view(B, E_per_graph)

        # Step the simulation forward
        next_inventories, step_cost = transition_step(
            current_inventories=current_inventories,
            edge_flows=edge_flows,
            node_features=data_for_reset["node_features"],
            edge_index=data_for_reset["edge_index"],
            demand_t=demand_t,
            cost_params=data_for_reset["cost_params"],
        )

        current_inventories = next_inventories
        step_costs.append(step_cost)

    total_episode_cost = torch.stack(step_costs).sum(dim=0)
    effective_periods = max(periods - ignore_periods, 1)
    cost_to_report = torch.stack(step_costs[-effective_periods:]).sum(dim=0)
    return total_episode_cost, cost_to_report


def calculate_losses(
    total_episode_cost: torch.Tensor,
    cost_to_report: torch.Tensor,
    num_nodes: int,
    periods: int,
    ignore_periods: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute losses from episode costs.
    The loss for backpropagation is the simple mean of the total episode cost.
    """
    loss_for_backward = total_episode_cost.mean()
    loss_to_report = cost_to_report.mean()
    
    return loss_for_backward, loss_to_report


def perform_gradient_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> None:
    """Run a standard gradient step with clipping."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip_norm > 0:
        clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()