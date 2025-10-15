from typing import Any, Dict
import torch


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

  inventories = {
    "stores": torch.zeros(batch_size, num_stores, device=device, dtype=dtype),
    "warehouses": torch.zeros(batch_size, num_warehouses, device=device, dtype=dtype),
  }

  normal = torch.distributions.Normal(loc=torch.tensor(5.0, dtype=dtype, device=device), scale=torch.tensor(2.0, dtype=dtype, device=device))
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


