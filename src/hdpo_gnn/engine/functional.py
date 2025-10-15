from typing import Any, Dict, Optional, Tuple

import torch


def transition_step(
    current_state: Dict[str, torch.Tensor],
    action: Dict[str, torch.Tensor],
    demand_t: torch.Tensor,
    cost_params: Dict[str, torch.Tensor],
    lead_times: Dict[str, int],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Pure functional transition: computes next state and per-sample cost.

    Args:
      current_state: Mapping with keys 'inventory_stores' [B,S], 'inventory_warehouses' [B,W],
        and optionally 'pipeline_stores' [B,S,Ls], 'pipeline_warehouses' [B,W,Lw].
      action: Mapping with keys 'stores' [B,S] and 'warehouses' [B,W]. Non-negative.
      demand_t: Tensor [B,S] for current period demand.
      cost_params: Tensors for costs: 'holding_store', 'underage_store', 'holding_warehouse'.
      lead_times: Dict with ints 'stores' and 'warehouses'.

    Returns:
      next_state: Same structure as current_state, with updated inventories/pipelines.
      cost: Per-sample cost tensor [B].
    """
    inv_s = current_state["inventory_stores"]
    inv_w = current_state["inventory_warehouses"]
    B, S = inv_s.shape
    W = inv_w.shape[1]

    Ls = int(lead_times.get("stores", 0))
    Lw = int(lead_times.get("warehouses", 0))
    pipe_s = current_state.get("pipeline_stores", None)
    pipe_w = current_state.get("pipeline_warehouses", None)

    a_s = torch.clamp(action["stores"], min=0.0).reshape(B, S)
    a_w = torch.clamp(action["warehouses"], min=0.0).reshape(B, W)

    if Lw > 0:
        arrivals_w = pipe_w[:, :, 0] if pipe_w is not None else torch.zeros_like(a_w)
        new_pipe_w = (
            torch.cat([pipe_w[:, :, 1:], a_w.unsqueeze(-1)], dim=2)
            if pipe_w is not None
            else a_w.unsqueeze(-1)
        )
    else:
        arrivals_w = a_w
        new_pipe_w = pipe_w
    inv_w_next = inv_w + arrivals_w

    total_ship_req = a_s.sum(dim=1, keepdim=True)
    total_wh_inv = inv_w_next.sum(dim=1, keepdim=True)
    # Smooth, always-differentiable cap in (0,1): ratio / (1 + ratio)
    feasible_ratio = total_wh_inv / (total_ship_req + 1e-8)
    feasible_factor = feasible_ratio / (1.0 + feasible_ratio)
    a_s_feasible = a_s * feasible_factor

    shipped_total = a_s_feasible.sum(dim=1, keepdim=True)
    inv_share = inv_w_next / (total_wh_inv + 1e-8)
    deduct_w = inv_share * shipped_total
    inv_w_next = inv_w_next - deduct_w

    if Ls > 0:
        arrivals_s = (
            pipe_s[:, :, 0] if pipe_s is not None else torch.zeros_like(a_s_feasible)
        )
        new_pipe_s = (
            torch.cat([pipe_s[:, :, 1:], a_s_feasible.unsqueeze(-1)], dim=2)
            if pipe_s is not None
            else a_s_feasible.unsqueeze(-1)
        )
    else:
        arrivals_s = a_s_feasible
        new_pipe_s = pipe_s
    inv_s_mid = inv_s + arrivals_s

    beta = 10.0
    delta = demand_t - inv_s_mid
    sales = demand_t - torch.nn.functional.softplus(delta, beta=beta) / beta
    end_inv_s = inv_s_mid - sales
    underage = demand_t - sales

    hs = cost_params["holding_store"].reshape(-1)
    us = cost_params["underage_store"].reshape(-1)
    hw = cost_params["holding_warehouse"].reshape(-1)
    if hs.numel() == 1:
        hs = hs.expand(S)
    if us.numel() == 1:
        us = us.expand(S)
    if hw.numel() == 1:
        hw = hw.expand(W)

    holding_store_cost = (end_inv_s * hs.view(1, S)).sum(dim=1)
    underage_store_cost = (underage * us.view(1, S)).sum(dim=1)
    holding_warehouse_cost = (inv_w_next * hw.view(1, W)).sum(dim=1)
    cost = holding_store_cost + underage_store_cost + holding_warehouse_cost

    next_state = {
        "inventory_stores": end_inv_s,
        "inventory_warehouses": inv_w_next,
    }
    if Ls > 0:
        next_state["pipeline_stores"] = new_pipe_s
    if Lw > 0:
        next_state["pipeline_warehouses"] = new_pipe_w
    return next_state, cost
