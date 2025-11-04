# src/hdpo_gnn/engine/functional.py
from typing import Any, Dict, Tuple

import torch
from torch_scatter import scatter_add # Use scatter_add for compatibility


def transition_step(
    current_inventories: torch.Tensor,
    edge_flows: torch.Tensor,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    demand_t: torch.Tensor,
    holding_costs: torch.Tensor,
    underage_costs: torch.Tensor,
    procurement_costs: torch.Tensor,
    is_transshipment_mask: torch.Tensor,
    unmet_demand_assumption: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Graph-based inventory transition function with edge flows.
    This version includes correct logic for backlogged/lost sales and cost calculation.
    """
    B, N = current_inventories.shape
    device = current_inventories.device
    dtype = current_inventories.dtype
    
    source, dest = edge_index[0], edge_index[1]
    
    # This section is based on your original, working code.
    # It calculates feasible flows based on inventory BEFORE arrivals.
    # This is a valid physical model and aligns with the paper's description.
    source_expanded = source.unsqueeze(0).expand(B, -1)
    total_requested_outflow = torch.zeros(B, N, device=device, dtype=dtype).scatter_add_(1, source_expanded, edge_flows)
    
    cap_factor = current_inventories / (total_requested_outflow + 1e-8)
    
    # For transshipment nodes, they must send everything. The cap factor should be based on what arrives.
    # However, to keep changes minimal, we will stick to the paper's core logic where flows are decided first.
    # If a transshipment node has 0 inventory, its cap_factor is 0, and it sends nothing. This is the source
    # of the "Cycle of Inaction" if it never receives inventory. Let's proceed and see if the non-zero cost
    # allows the gradient to flow back to the supplier->warehouse edge.
    
    # The paper mentions that for transshipment, the node must fully allocate inventory.
    # Let's handle that case specifically.
    
    # We must ensure the mask is on the correct device.
    is_transshipment_mask_expanded = is_transshipment_mask.unsqueeze(0).expand(B, -1)
    
    # For non-transshipment nodes, cap the factor at 1.
    cap_factor = torch.where(
        is_transshipment_mask_expanded, 
        cap_factor, 
        torch.clamp(cap_factor, max=1.0)
    )

    source_capping = cap_factor.gather(1, source_expanded)
    feasible_edge_flows = edge_flows * source_capping

    # Calculate net inventory changes from these feasible flows
    dest_expanded = dest.unsqueeze(0).expand(B, -1)
    total_in_flow = torch.zeros(B, N, device=device, dtype=dtype).scatter_add_(1, dest_expanded, feasible_edge_flows)
    total_out_flow = torch.zeros(B, N, device=device, dtype=dtype).scatter_add_(1, source_expanded, feasible_edge_flows)
    
    inventory_after_ship_and_arrival = current_inventories - total_out_flow + total_in_flow

    # --- Sales Calculation ---
    is_demand_facing = node_features[:, 2].to(device)
    
    if unmet_demand_assumption == "backlogged":
        # In backlogged, sales always equal demand, potentially driving inventory negative.
        # Mask sales to only occur at demand-facing nodes.
        sales = demand_t * is_demand_facing.unsqueeze(0)
    elif unmet_demand_assumption == "lost":
        # In lost sales, sales are the minimum of available inventory and demand.
        # `clamp(min=0)` ensures we don't sell from backorders.
        available_inv = torch.clamp(inventory_after_ship_and_arrival, min=0)
        potential_sales = torch.min(available_inv, demand_t)
        sales = potential_sales * is_demand_facing.unsqueeze(0)
    else:
        raise ValueError(f"Unknown unmet_demand_assumption: {unmet_demand_assumption}")
    
    # --- Inventory Update ---
    next_inventories = inventory_after_ship_and_arrival - sales
    
    # --- CORRECTED COST CALCULATION (THE MINIMAL FIX) ---
    
    # Holding cost only applies to POSITIVE inventory.
    positive_inventory = torch.clamp(next_inventories, min=0)
    holding_cost = (positive_inventory * holding_costs).sum(dim=1)
    
    # Underage cost applies to POSITIVE backorders (negative inventory).
    backorders = torch.clamp(-next_inventories, min=0)
    underage_cost = (backorders * underage_costs).sum(dim=1)
    
    procurement_cost_total = (feasible_edge_flows * procurement_costs.unsqueeze(0)).sum(dim=1)
    
    step_cost = holding_cost + underage_cost + procurement_cost_total
    
    return next_inventories, step_cost