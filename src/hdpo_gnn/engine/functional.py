from typing import Any, Dict, Tuple

import torch
from torch_scatter import scatter


def transition_step(
    current_inventories: torch.Tensor,
    edge_flows: torch.Tensor,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    demand_t: torch.Tensor,
    cost_params: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Graph-based inventory transition function with edge flows.
    
    This function computes the next inventory state and associated costs for a
    generic graph topology. Flows are requested along edges, subject to per-node
    feasibility constraints. The function is fully differentiable for end-to-end
    training of graph neural network policies.
    
    Args:
        current_inventories: Tensor of shape [B, N] containing current on-hand
            inventory at each node for each batch sample.
        edge_flows: Tensor of shape [B, E] containing requested flow quantities
            along each edge (model's predicted actions).
        node_features: Tensor of shape [N, F] containing static node features.
            Feature at index 1 is 'is_demand_facing' (1 for demand-facing nodes,
            0 otherwise).
        edge_index: Tensor of shape [2, E] in COO format defining graph connectivity.
            edge_index[0] are source nodes, edge_index[1] are destination nodes.
        demand_t: Tensor of shape [B, N] containing external demand at each node
            for the current time step.
        cost_params: Dictionary containing cost scalars:
            - 'holding_store': Cost per unit of inventory held
            - 'underage_store': Cost per unit of unsatisfied demand
    
    Returns:
        next_inventories: Tensor of shape [B, N] with updated inventory levels.
        step_cost: Tensor of shape [B] with total cost for each batch sample.
    
    Implementation Details:
        1. Per-Node Feasibility: Each node can only send out flow up to its current
           inventory. Uses a numerically stable capping function: 
           capped = inventory / (requested_outflow + inventory + epsilon)
        2. Flow Aggregation: Uses torch_scatter to aggregate flows at source and
           destination nodes, computing inflows and outflows separately.
        3. Order of Operations: CRITICAL - shipments leave BEFORE sales occur:
           inventory_after_shipments = current_inventory - outflows
           sales = min(demand, inventory_after_shipments) at demand-facing nodes
           next_inventory = inventory_after_shipments + inflows - sales
        4. Cost Calculation: Holding costs on remaining inventory and underage costs
           on unsatisfied demand, both masked appropriately by node features.
    """
    B, N = current_inventories.shape
    E = edge_index.shape[1]
    device = current_inventories.device
    dtype = current_inventories.dtype
    
    # Extract source and destination nodes from edge connectivity
    source = edge_index[0]  # [E]
    dest = edge_index[1]    # [E]
    
    # Ensure edge_flows are non-negative
    edge_flows = torch.clamp(edge_flows, min=0.0)  # [B, E]
    
    # Step 1: Per-Node Feasibility Capping
    # Calculate total outgoing flow requested from each node
    # We need to expand source indices for batch dimension
    source_expanded = source.unsqueeze(0).expand(B, -1)  # [B, E]
    
    # Aggregate outgoing flows per node using functional scatter_add (gradient-friendly)
    total_outgoing_flow = torch.zeros(B, N, device=device, dtype=dtype)
    total_outgoing_flow = total_outgoing_flow.scatter_add(1, source_expanded, edge_flows)  # [B, N]
    
    # Smooth, differentiable feasibility factor
    # Single-step formula for numerical stability and better gradients:
    # capped_factor = inventory / (total_outgoing_flow + inventory + epsilon)
    # This directly calculates the feasible proportion and has much better gradient behavior
    capped_factor = current_inventories / (total_outgoing_flow + current_inventories + 1e-8)  # [B, N]
    
    # Apply capping to edge flows
    # Each edge's flow is scaled by its source node's capping factor
    source_capping = capped_factor.gather(1, source_expanded)  # [B, E]
    feasible_edge_flows = edge_flows * source_capping  # [B, E]
    
    # Step 2: Calculate Net Inventory Change
    # Aggregate inflows and outflows using functional scatter_add (gradient-friendly)
    dest_expanded = dest.unsqueeze(0).expand(B, -1)  # [B, E]
    
    total_in_flow = torch.zeros(B, N, device=device, dtype=dtype)
    total_in_flow = total_in_flow.scatter_add(1, dest_expanded, feasible_edge_flows)  # [B, N]
    
    total_out_flow = torch.zeros(B, N, device=device, dtype=dtype)
    total_out_flow = total_out_flow.scatter_add(1, source_expanded, feasible_edge_flows)  # [B, N]
    
    # Step 3: Calculate intermediate inventory after shipments leave
    # CRITICAL: Sales must be calculated AFTER shipments have left the node
    inventory_after_shipments = current_inventories - total_out_flow  # [B, N]
    
    # Step 4: Calculate Sales (only at demand-facing nodes)
    # Smooth approximation of sales = min(demand, inventory_after_shipments)
    # Use softplus for differentiability: sales = inventory_after_shipments - softplus(inventory_after_shipments - demand) / beta
    beta = 10.0
    
    # Calculate potential sales for all nodes
    potential_sales = inventory_after_shipments - torch.nn.functional.softplus(inventory_after_shipments - demand_t, beta=beta) / beta  # [B, N]
    
    # Apply hard mask to get final sales (only at demand-facing nodes)
    is_demand_facing = node_features[:, 1]  # [N]
    sales = potential_sales * is_demand_facing.unsqueeze(0)  # [B, N]
    
    # Step 5: Update Inventories
    # Final inventory = inventory_after_shipments + inflows - sales
    next_inventories = inventory_after_shipments + total_in_flow - sales  # [B, N]
    
    # Step 6: Calculate Costs
    # Holding cost: cost per unit of inventory held
    holding_cost_per_node = next_inventories * cost_params["holding_store"]
    holding_cost = holding_cost_per_node.sum(dim=1)  # [B]
    
    # Underage cost: cost per unit of unsatisfied demand
    # Only at demand-facing nodes (hard mask ensures correct behavior)
    underage = demand_t - sales  # [B, N]
    underage_cost_per_node = underage * cost_params["underage_store"]
    underage_cost = underage_cost_per_node.sum(dim=1)  # [B]
    
    # Total step cost
    step_cost = holding_cost + underage_cost  # [B]
    
    return next_inventories, step_cost
