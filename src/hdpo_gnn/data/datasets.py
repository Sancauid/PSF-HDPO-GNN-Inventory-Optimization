from typing import Any, Dict, List

import torch
from torch import Tensor
from torch_geometric.data import Data
from omegaconf import DictConfig

from src.hdpo_gnn.utils.graph_parser import parse_graph_topology


def create_synthetic_data_dict(config: DictConfig) -> Dict[str, Any]:
    """
    Create a synthetic dataset dictionary for inventory RL experiments using graph topology.

    This function generates batched initial inventories for all nodes in the graph,
    a time-series of stochastic non-negative demands (masked to only affect demand-facing nodes),
    cost parameters, and fixed lead times. All numerical arrays are returned as PyTorch tensors 
    to enable GPU execution and differentiation where applicable.

    Args:
      config: OmegaConf DictConfig containing:
        - problem_params.graph: Graph topology with nodes and edges
        - problem_params.periods: Number of time periods T
        - data_params.n_samples: Batch size B (number of scenarios)

    Returns:
      A dictionary with the following keys:
        - inventories: tensor of shape [B, N] where N is total number of nodes
        - demands: tensor of shape [T, B, N] (non-negative, masked for demand-facing nodes)
        - node_features: tensor of shape [N, 2] with features [has_external_supply, is_demand_facing]
        - edge_index: tensor of shape [2, E] in COO format
        - cost_params: dict with tensors 'holding_store', 'underage_store', and 'holding_warehouse'
        - lead_times: dict with ints {'stores': 2, 'warehouses': 3}
    """
    # Parse graph topology
    node_features, edge_index = parse_graph_topology(config)
    N = node_features.shape[0]  # Total number of nodes
    
    # Extract parameters
    num_periods: int = int(config.problem_params.periods)
    batch_size: int = int(config.data_params.n_samples)

    device = torch.device("cpu")
    dtype = torch.float32

    # Generate unified inventories for all nodes
    inventory_dist = torch.distributions.Exponential(
        rate=torch.tensor(0.15, dtype=dtype, device=device)  # mean = 1/0.15 ≈ 6.67
    )
    inventories = inventory_dist.rsample((batch_size, N))

    # Generate demands for all nodes
    normal = torch.distributions.Normal(
        loc=torch.tensor(5.0, dtype=dtype, device=device),
        scale=torch.tensor(2.0, dtype=dtype, device=device),
    )
    raw_demands = normal.rsample((num_periods, batch_size, N)).clamp(min=0.0)
    
    # Create demand mask from node features (is_demand_facing is the second feature)
    demand_mask = node_features[:, 1].unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    demands = raw_demands * demand_mask  # [T, B, N]

    cost_params = {
        "holding_store": torch.tensor(1.0, device=device, dtype=dtype),
        "underage_store": torch.tensor(9.0, device=device, dtype=dtype),
        "holding_warehouse": torch.tensor(0.5, device=device, dtype=dtype),
    }

    lead_times = {"stores": 2, "warehouses": 3}

    return {
        "inventories": inventories,
        "demands": demands,
        "node_features": node_features,
        "edge_index": edge_index,
        "cost_params": cost_params,
        "lead_times": lead_times,
    }


def create_pyg_dataset(data: Dict[str, Any], config: DictConfig) -> List[Data]:
    """
    Convert synthetic tensors into a list of PyG `Data` graphs for GNN models.

    Each sample becomes a graph with N nodes using the graph topology defined in the config.
    All graphs share the same node features and edge connectivity, but have different
    initial inventories and demand time-series per sample.

    Args:
        data: Output of `create_synthetic_data_dict`, containing:
          - 'inventories': tensor of shape [B, N] - initial inventories per sample
          - 'demands': tensor of shape [T, B, N] - demand time-series per sample
          - 'node_features': tensor of shape [N, F] - static node features
          - 'edge_index': tensor of shape [2, E] - graph connectivity
          - 'cost_params': dict with cost parameters
          - 'lead_times': dict with lead time parameters
        config: DictConfig containing graph topology and other parameters

    Returns:
        A list of `torch_geometric.data.Data` objects, one per sample. Each Data object contains:
          - x: Node features tensor [N, F] (same for all graphs)
          - edge_index: Graph connectivity [2, E] (same for all graphs)
          - initial_inventory: Initial inventories for this sample [N]
          - demands: Demand time-series for this sample [T, N]
          - cost_params and lead_times as scalar attributes
    """
    # Extract dimensions from data
    inventories = data["inventories"]  # [B, N]
    demands = data["demands"]  # [T, B, N]
    node_features = data["node_features"]  # [N, F]
    edge_index = data["edge_index"]  # [2, E]
    cost_params = data["cost_params"]
    lead_times = data["lead_times"]
    
    B, N = inventories.shape
    T = demands.shape[0]

    data_list: List[Data] = []
    for i in range(B):
        # Create PyG Data object
        g = Data(x=node_features, edge_index=edge_index)
        
        # Attach sample-specific data
        g.initial_inventory = inventories[i]  # [N]
        g.demands = demands[:, i, :]  # [T, N]
        
        # Attach cost parameters and lead times
        g.holding_store = cost_params["holding_store"].detach().clone()
        g.underage_store = cost_params["underage_store"].detach().clone()
        g.holding_warehouse = cost_params["holding_warehouse"].detach().clone()
        g.lead_time_stores = torch.tensor(
            int(lead_times.get("stores", 0)), dtype=torch.long
        )
        g.lead_time_warehouses = torch.tensor(
            int(lead_times.get("warehouses", 0)), dtype=torch.long
        )

        data_list.append(g)

    return data_list
