"""
Differentiable inventory simulator compatible with batch optimization.
"""
from typing import Any, Dict, Optional, Tuple
import torch


class DifferentiableSimulator:
    """
    Differentiable inventory system simulator with store and warehouse nodes.

    The simulator maintains batched state and supports shipments with lead times
    via FIFO pipelines. Transitions and costs are implemented with PyTorch
    tensors to enable end-to-end differentiation.
    """

    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> None:
        """
        Initializes an empty simulator. Call reset to provide data and shapes.

        Args:
            device: Torch device used for internal tensors.
            dtype: Torch dtype used for internal tensors.
        """
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.t: int = 0
        self.batch_size: int = 0
        self.num_stores: int = 0
        self.num_warehouses: int = 0
        self.lead_time_stores: int = 0
        self.lead_time_warehouses: int = 0
        self.inventory_stores: Optional[torch.Tensor] = None
        self.inventory_warehouses: Optional[torch.Tensor] = None
        self.pipeline_stores: Optional[torch.Tensor] = None
        self.pipeline_warehouses: Optional[torch.Tensor] = None
        self.demands: Optional[torch.Tensor] = None
        self.holding_store_rate: Optional[torch.Tensor] = None
        self.underage_store_rate: Optional[torch.Tensor] = None
        self.holding_warehouse_rate: Optional[torch.Tensor] = None

    def reset(
        self,
        inventories: Dict[str, torch.Tensor],
        demands: torch.Tensor,
        cost_params: Dict[str, torch.Tensor | float],
        lead_times: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Resets internal state for a batch of scenarios.

        Args:
            inventories: Mapping with keys 'stores' [B, S] and 'warehouses' [B, W].
            demands: Tensor [T, B, S] of exogenous demands per period.
            cost_params: Mapping with keys 'holding_store', 'underage_store', 'holding_warehouse'.
                Each value can be a scalar or a 1D tensor of shape [S] or [W].
            lead_times: Mapping with keys 'stores' (int Ls) and 'warehouses' (int Lw).

        Returns:
            Observation dict containing current state tensors.
        """
        inv_s = inventories["stores"].to(device=self.device, dtype=self.dtype)
        inv_w = inventories["warehouses"].to(device=self.device, dtype=self.dtype)
        T, B, S = demands.shape
        W = inv_w.shape[1]
        self.t = 0
        self.batch_size = B
        self.num_stores = S
        self.num_warehouses = W
        self.lead_time_stores = int(lead_times.get("stores", 0))
        self.lead_time_warehouses = int(lead_times.get("warehouses", 0))
        self.inventory_stores = inv_s.clone()
        self.inventory_warehouses = inv_w.clone()
        if self.lead_time_stores > 0:
            self.pipeline_stores = torch.zeros(B, S, self.lead_time_stores, device=self.device, dtype=self.dtype)
        else:
            self.pipeline_stores = None
        if self.lead_time_warehouses > 0:
            self.pipeline_warehouses = torch.zeros(B, W, self.lead_time_warehouses, device=self.device, dtype=self.dtype)
        else:
            self.pipeline_warehouses = None
        self.demands = demands.to(device=self.device, dtype=self.dtype)

        hs = cost_params["holding_store"]
        us = cost_params["underage_store"]
        hw = cost_params["holding_warehouse"]
        self.holding_store_rate = (hs if isinstance(hs, torch.Tensor) else torch.tensor(hs, device=self.device, dtype=self.dtype)).reshape(-1)
        self.underage_store_rate = (us if isinstance(us, torch.Tensor) else torch.tensor(us, device=self.device, dtype=self.dtype)).reshape(-1)
        self.holding_warehouse_rate = (hw if isinstance(hw, torch.Tensor) else torch.tensor(hw, device=self.device, dtype=self.dtype)).reshape(-1)
        if self.holding_store_rate.numel() == 1:
            self.holding_store_rate = self.holding_store_rate.expand(self.num_stores)
        if self.underage_store_rate.numel() == 1:
            self.underage_store_rate = self.underage_store_rate.expand(self.num_stores)
        if self.holding_warehouse_rate.numel() == 1:
            self.holding_warehouse_rate = self.holding_warehouse_rate.expand(self.num_warehouses)

        return {
            "inventory_stores": self.inventory_stores,
            "inventory_warehouses": self.inventory_warehouses,
            "pipeline_stores": self.pipeline_stores if self.pipeline_stores is not None else torch.zeros(0, device=self.device, dtype=self.dtype),
            "pipeline_warehouses": self.pipeline_warehouses if self.pipeline_warehouses is not None else torch.zeros(0, device=self.device, dtype=self.dtype),
            "time_index": torch.tensor(self.t, device=self.device),
        }

    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Advances the simulator by one period applying shipments and demands.

        Args:
            actions: Mapping with keys 'stores' [B, S] and 'warehouses' [B, W].

        Returns:
            A tuple (observation, cost_per_sample). Observation contains updated
            state tensors. The cost tensor has shape [B].
        """
        assert self.inventory_stores is not None and self.inventory_warehouses is not None
        assert self.demands is not None and self.holding_store_rate is not None and self.underage_store_rate is not None and self.holding_warehouse_rate is not None

        B = self.batch_size
        S = self.num_stores
        W = self.num_warehouses

        a_s = actions["stores"].to(device=self.device, dtype=self.dtype).reshape(B, S)
        a_w = actions["warehouses"].to(device=self.device, dtype=self.dtype).reshape(B, W)
        a_s = torch.clamp(a_s, min=0.0)
        a_w = torch.clamp(a_w, min=0.0)

        if self.lead_time_warehouses > 0 and self.pipeline_warehouses is not None:
            arrivals_w = self.pipeline_warehouses[:, :, 0]
            self.pipeline_warehouses = torch.cat([self.pipeline_warehouses[:, :, 1:], a_w.unsqueeze(-1)], dim=2)
        else:
            arrivals_w = a_w
        self.inventory_warehouses = self.inventory_warehouses + arrivals_w

        total_ship_req = a_s.sum(dim=1, keepdim=True)
        total_wh_inv = self.inventory_warehouses.sum(dim=1, keepdim=True)
        feasible_factor = torch.where(total_ship_req > 0, torch.clamp(total_wh_inv / total_ship_req, max=1.0), torch.ones_like(total_ship_req))
        a_s_feasible = a_s * feasible_factor

        shipped_total = a_s_feasible.sum(dim=1, keepdim=True)
        inv_share = self.inventory_warehouses / (total_wh_inv + 1e-8)
        deduct_w = inv_share * shipped_total
        self.inventory_warehouses = self.inventory_warehouses - deduct_w

        if self.lead_time_stores > 0 and self.pipeline_stores is not None:
            arrivals_s = self.pipeline_stores[:, :, 0]
            self.pipeline_stores = torch.cat([self.pipeline_stores[:, :, 1:], a_s_feasible.unsqueeze(-1)], dim=2)
        else:
            arrivals_s = a_s_feasible
        self.inventory_stores = self.inventory_stores + arrivals_s

        if self.t < self.demands.shape[0]:
            demand_t = self.demands[self.t]
        else:
            demand_t = torch.zeros(B, S, device=self.device, dtype=self.dtype)

        sales = torch.minimum(self.inventory_stores, demand_t)
        end_inv_s = self.inventory_stores - sales
        underage = demand_t - sales

        holding_store_cost = (end_inv_s * self.holding_store_rate.view(1, S)).sum(dim=1)
        underage_store_cost = (underage * self.underage_store_rate.view(1, S)).sum(dim=1)
        holding_warehouse_cost = (self.inventory_warehouses * self.holding_warehouse_rate.view(1, W)).sum(dim=1)
        cost = holding_store_cost + underage_store_cost + holding_warehouse_cost

        self.inventory_stores = end_inv_s
        self.t += 1

        obs = {
            "inventory_stores": self.inventory_stores,
            "inventory_warehouses": self.inventory_warehouses,
            "pipeline_stores": self.pipeline_stores if self.pipeline_stores is not None else torch.zeros(0, device=self.device, dtype=self.dtype),
            "pipeline_warehouses": self.pipeline_warehouses if self.pipeline_warehouses is not None else torch.zeros(0, device=self.device, dtype=self.dtype),
            "time_index": torch.tensor(self.t, device=self.device),
        }
        return obs, cost


