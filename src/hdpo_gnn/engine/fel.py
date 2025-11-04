# src/hdpo_gnn/engine/fel.py

import logging
import torch

log = logging.getLogger(__name__)

def apply_fel(
    raw_edge_logits: torch.Tensor,
    current_inventories: torch.Tensor,
    env: 'Environment',
    config: 'DictConfig'
) -> torch.Tensor:
    """
    Dispatcher for Feasibility Enforcement Layers.
    Takes raw model logits and returns feasible, non-negative flows.
    """
    fel_type = config.model.gnn_params.fel_type

    num_samples = current_inventories.shape[0]
    num_edges = env.num_edges
    
    # Step 1: Convert raw logits to non-negative "tentative orders"
    tentative_orders = torch.nn.functional.softplus(raw_edge_logits + config.training.softplus_bias)
    tentative_orders = tentative_orders.view(num_samples, num_edges)
    
    # The paper's FELs are all variants of proportional allocation.
    # We can handle them all in one unified function.
    return _proportional_allocation(tentative_orders, current_inventories, env)

def _proportional_allocation(
    tentative_orders: torch.Tensor,
    current_inventories: torch.Tensor,
    env: 'Environment'
) -> torch.Tensor:
    """
    A single, unified function for proportional allocation that correctly handles
    standard nodes, transshipment nodes, and the infinite-supply source node.
    
    This correctly implements both g1a and the standard proportional allocation.
    """
    B, E = tentative_orders.shape
    N = current_inventories.shape[1]
    device = tentative_orders.device
    source_nodes = env.edge_index[0].to(device)
    
    feasible_flows = tentative_orders.clone()
    
    # Get a list of all unique physical supplier nodes in the graph
    # (i.e., any node that is the source of an edge, excluding the infinite supplier node 0)
    physical_supplier_ids = torch.unique(source_nodes[source_nodes > 0]).tolist()
    
    for node_id in physical_supplier_ids:
        # Find all edges originating from this physical node
        is_outgoing_edge_mask = (source_nodes == node_id)
        
        # Get the sum of tentative orders for these edges
        requested_outflow = tentative_orders[:, is_outgoing_edge_mask].sum(dim=1) # Shape [B]
        inventory_at_node = current_inventories[:, node_id] # Shape [B]
        
        # Check if this node is a transshipment node
        is_transshipment = env.config.dynamics.is_transshipment and node_id in env.warehouse_ids
        # Calculate scaling factor based on node type
        if is_transshipment:
            # Full Allocation (g1a): Must allocate everything. Scaling factor is inventory / requested.
            # (See Appendix C.4, Equation 28)
            scaling_factor = inventory_at_node / (requested_outflow + 1e-8)
        else:
            # Standard Proportional Allocation: Cannot ship more than available.
            # Scaling factor is min(1, inventory / requested).
            # (See Appendix A.5, Equation 14)
            scaling_factor = torch.min(
                torch.ones_like(inventory_at_node), 
                inventory_at_node / (requested_outflow + 1e-8)
            )
        # Apply the calculated scaling factor to the flows for this node's outgoing edges
        # We need to expand the scaling_factor to match the number of outgoing edges
        feasible_flows[:, is_outgoing_edge_mask] = tentative_orders[:, is_outgoing_edge_mask] * scaling_factor.unsqueeze(-1)
        
    return feasible_flows