from typing import Any, Dict, List

import torch
from torch import Tensor
from torch_geometric.data import Data


def create_synthetic_data_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a synthetic dataset dictionary for inventory RL experiments.

    This function generates batched initial inventories for stores and warehouses,
    a time-series of stochastic non-negative demands, cost parameters, and fixed
    lead times. All numerical arrays are returned as PyTorch tensors to enable
    GPU execution and differentiation where applicable.

    Args:
      config: Configuration mapping. Expected structure:
        - problem_params: dict with keys
            * n_stores (int): number of store nodes S
            * n_warehouses (int): number of warehouse nodes W
            * periods (int): number of time periods T
        - data_params: dict with keys
            * n_samples (int): batch size B (number of scenarios)

    Returns:
      A dictionary with the following keys:
        - inventories: dict with
            * 'stores': tensor of shape [B, S]
            * 'warehouses': tensor of shape [B, W]
        - demands: tensor of shape [T, B, S] (non-negative)
        - cost_params: dict with tensors 'holding_store', 'underage_store',
          and 'holding_warehouse' (scalars as 0-D tensors)
        - lead_times: dict with ints {'stores': 2, 'warehouses': 3}
    """
    problem_params = config.get("problem_params", {})
    data_params = config.get("data_params", {})

    num_stores: int = int(problem_params.get("n_stores", 1))
    num_warehouses: int = int(problem_params.get("n_warehouses", 1))
    num_periods: int = int(problem_params.get("periods", 1))
    batch_size: int = int(data_params.get("n_samples", 1))

    device = torch.device("cpu")
    dtype = torch.float32

    # Initialize with realistic starting inventories
    # Stores start with some inventory to meet initial demand
    store_inv_dist = torch.distributions.Exponential(
        rate=torch.tensor(0.2, dtype=dtype, device=device)  # mean = 5
    )
    warehouse_inv_dist = torch.distributions.Exponential(
        rate=torch.tensor(0.1, dtype=dtype, device=device)  # mean = 10
    )

    inventories = {
        "stores": store_inv_dist.rsample((batch_size, num_stores)),
        "warehouses": warehouse_inv_dist.rsample((batch_size, num_warehouses)),
    }

    normal = torch.distributions.Normal(
        loc=torch.tensor(5.0, dtype=dtype, device=device),
        scale=torch.tensor(2.0, dtype=dtype, device=device),
    )
    demands = normal.rsample((num_periods, batch_size, num_stores)).clamp(min=0.0)

    cost_params = {
        "holding_store": torch.tensor(1.0, device=device, dtype=dtype),
        "underage_store": torch.tensor(9.0, device=device, dtype=dtype),
        "holding_warehouse": torch.tensor(0.5, device=device, dtype=dtype),
    }

    lead_times = {"stores": 2, "warehouses": 3}

    return {
        "inventories": inventories,
        "demands": demands,
        "cost_params": cost_params,
        "lead_times": lead_times,
    }


def create_pyg_dataset(data: Dict[str, Any], config: Dict[str, Any]) -> List[Data]:
    """
    Convert synthetic tensors into a list of PyG `Data` graphs for GNN models.

    Each sample `i` becomes a graph with `n_stores` nodes and a fully-connected
    directed edge set (excluding self-loops). Node features contain the initial
    store inventories for that sample. Additional tensors relevant for simulation
    (e.g., demands, cost parameters, lead times) are attached as attributes.

    Args:
        data: Output of `create_synthetic_data_dict`, including keys
          'inventories', 'demands', 'cost_params', 'lead_times'.
        config: Config mapping with 'data_params' and 'problem_params'.

    Returns:
        A list of `torch_geometric.data.Data` objects, one per sample.
    """
    problem_params = config.get("problem_params", {})
    data_params = config.get("data_params", {})

    n_samples: int = int(data_params.get("n_samples", data["demands"].shape[1]))
    n_stores: int = int(
        problem_params.get("n_stores", data["inventories"]["stores"].shape[1])
    )

    stores_inv: Tensor = data["inventories"]["stores"]  # [B, S]
    demands: Tensor = data["demands"]  # [T, B, S]
    cost_params: Dict[str, Tensor] = data["cost_params"]
    lead_times: Dict[str, int] = data["lead_times"]

    idx = torch.arange(n_stores, dtype=torch.long)
    src = idx.repeat_interleave(n_stores)
    dst = idx.repeat(n_stores)
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)  # [2, E]

    data_list: List[Data] = []
    for i in range(n_samples):
        x_i = stores_inv[i].unsqueeze(-1)  # [S, 1]
        d_i = demands[:, i, :]  # [T, S]

        g = Data(x=x_i, edge_index=edge_index)
        g.demands = d_i

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
