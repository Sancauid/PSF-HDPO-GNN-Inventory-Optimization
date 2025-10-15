from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


def prepare_batch_for_simulation(
    batch: Any, architecture: str, device: torch.device
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """
    Prepare batch data for simulator.reset depending on the model architecture.

    Args:
      batch: Either a vanilla dictionary batch or a PyG Batch.
      architecture: 'vanilla' or 'gnn'.
      device: Target device for tensors.

    Returns:
      data_for_reset: Mapping with 'inventories', 'demands', 'cost_params', 'lead_times' (and
        for GNN includes 'pyg_batch', 'B', 'N').
      store_mask: Boolean mask over stores (reserved for filtering/use later).
    """
    if architecture == "vanilla":
        assert isinstance(batch, dict)
        data_for_reset: Dict[str, Any] = {
            "inventories": batch["inventories"],
            "demands": batch["demands"],
            "cost_params": batch["cost_params"],
            "lead_times": batch["lead_times"],
        }
        num_stores = data_for_reset["inventories"]["stores"].shape[1]
        store_mask = torch.ones(
            num_stores, dtype=torch.bool, device=data_for_reset["demands"].device
        )
        return data_for_reset, store_mask

    # GNN
    assert isinstance(batch, Batch)
    dense_x, mask = to_dense_batch(batch.x, batch.batch)
    B, N = dense_x.size(0), dense_x.size(1)
    dtype = dense_x.dtype

    inventories = {
        "stores": dense_x[..., 0].to(device=device, dtype=dtype),
        "warehouses": torch.zeros(B, 1, device=device, dtype=dtype),
    }

    if hasattr(batch, "demands"):
        # Expect flattened by nodes: [B*N, T] or per-graph [S,T] tiled across graphs
        d = batch.demands
        if d.dim() == 2:
            T = d.size(1)
            demands = d.view(B, N, T).permute(2, 0, 1).to(device=device, dtype=dtype)
        elif d.dim() == 3 and d.size(0) == B and d.size(1) == N:
            # [B,N,T] -> [T,B,N]
            demands = d.permute(2, 0, 1).to(device=device, dtype=dtype)
        elif d.dim() == 3 and d.size(1) == N:  # already [T,B,N]
            demands = d.to(device=device, dtype=dtype)
        else:
            raise ValueError("Unexpected demands shape in PyG batch")
    else:
        raise ValueError("PyG batch missing 'demands' attribute")

    cost_params = {
        "holding_store": torch.as_tensor(
            getattr(batch, "holding_store", 1.0), device=device, dtype=dtype
        ),
        "underage_store": torch.as_tensor(
            getattr(batch, "underage_store", 1.0), device=device, dtype=dtype
        ),
        "holding_warehouse": torch.as_tensor(
            getattr(batch, "holding_warehouse", 0.5), device=device, dtype=dtype
        ),
    }
    lead_times = {
        "stores": int(
            torch.as_tensor(getattr(batch, "lead_time_stores", 0)).flatten()[0].item()
        ),
        "warehouses": int(
            torch.as_tensor(getattr(batch, "lead_time_warehouses", 0))
            .flatten()[0]
            .item()
        ),
    }

    data_for_reset = {
        "inventories": inventories,
        "demands": demands,
        "cost_params": cost_params,
        "lead_times": lead_times,
        "pyg_batch": batch,
        "B": B,
        "N": N,
    }
    store_mask = torch.ones(N, dtype=torch.bool, device=device)
    return data_for_reset, store_mask


def run_simulation_episode(
    model: nn.Module,
    simulator: Any,
    batch: Any | None = None,
    periods: int = 1,
    ignore_periods: int = 0,
    data_for_reset: Dict[str, Any] | None = None,
    store_mask: torch.Tensor | None = None,
    architecture: str = "vanilla",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Roll out one full simulation episode and accumulate costs.

    Returns total cost and cost-to-report vectors (shape [B]).
    """
    inventories = data_for_reset["inventories"]
    demands = data_for_reset["demands"]
    cost_params = data_for_reset["cost_params"]
    lead_times = data_for_reset["lead_times"]
    current_state = {
        "inventory_stores": inventories["stores"],
        "inventory_warehouses": inventories["warehouses"],
    }

    # Always reset the provided simulator to align its internal state
    simulator.reset(
        inventories=inventories,
        demands=demands,
        cost_params=cost_params,
        lead_times=lead_times,
    )

    if architecture == "vanilla":
        B = demands.shape[1]
        num_warehouses = inventories["warehouses"].shape[1]
        total_episode_cost = torch.zeros(B, device=demands.device, dtype=demands.dtype)
        step_costs: list[torch.Tensor] = []
        for t in range(periods):
            x = torch.cat(
                [
                    current_state["inventory_stores"],
                    current_state["inventory_warehouses"],
                ],
                dim=1,
            )
            outputs = model(x)
            actions_stores = torch.sigmoid(outputs["stores"])
            actions_wh = torch.sigmoid(outputs["warehouses"])
            # apply store mask if provided
            if store_mask is not None:
                actions_stores = actions_stores * store_mask.view(1, -1).to(
                    actions_stores.device, actions_stores.dtype
                )
            if actions_wh.shape[1] == 1 and num_warehouses > 1:
                actions_wh = actions_wh.expand(-1, num_warehouses)
            next_state, step_cost = simulator.step(
                {"stores": actions_stores, "warehouses": actions_wh}
            )
            current_state = next_state
            total_episode_cost = total_episode_cost + step_cost
            step_costs.append(step_cost)
        effective = max(int(periods) - int(ignore_periods), 1)
        cost_to_report = (
            sum(step_costs[-effective:]) if step_costs else total_episode_cost
        )
        return total_episode_cost, cost_to_report.detach()

    # GNN
    pyg_batch: Batch = data_for_reset["pyg_batch"]
    B, N = int(data_for_reset["B"]), int(data_for_reset["N"])
    dtype, device = demands.dtype, demands.device
    num_warehouses = inventories["warehouses"].shape[1]
    total_episode_cost = torch.zeros(B, device=device, dtype=dtype)
    step_costs: list[torch.Tensor] = []
    for t in range(periods):
        current_store_inv = current_state["inventory_stores"]  # [B,N]
        current_batch_x = current_store_inv.view(B * N, -1)
        current_batch = Batch(
            x=current_batch_x, edge_index=pyg_batch.edge_index, batch=pyg_batch.batch
        )
        node_dict = model(current_batch.x, current_batch.edge_index)
        node_out = node_dict["stores"].view(B, N, -1)
        actions_stores = torch.nn.functional.softplus(node_out[:, :, 0])
        if store_mask is not None:
            actions_stores = actions_stores * store_mask.view(1, -1).to(
                actions_stores.device, actions_stores.dtype
            )
        actions_wh = torch.zeros(B, num_warehouses, device=device, dtype=dtype)
        next_state, step_cost = simulator.step(
            {"stores": actions_stores, "warehouses": actions_wh}
        )
        current_state = next_state
        total_episode_cost = total_episode_cost + step_cost
        step_costs.append(step_cost)
    effective = max(int(periods) - int(ignore_periods), 1)
    cost_to_report = sum(step_costs[-effective:]) if step_costs else total_episode_cost
    return total_episode_cost, cost_to_report.detach()


def calculate_losses(
    total_episode_cost: torch.Tensor,
    cost_to_report: torch.Tensor,
    total_real_stores: int,
    periods: int,
    ignore_periods: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalized losses (pure math).
    """
    effective_periods = max(int(periods) - int(ignore_periods), 1)
    denom_bwd = max(total_real_stores * int(periods), 1)
    denom_report = max(total_real_stores * effective_periods, 1)
    loss_for_backward = total_episode_cost.mean() / float(denom_bwd)
    loss_to_report = cost_to_report.mean() / float(denom_report)
    return loss_for_backward, loss_to_report


def perform_gradient_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> None:
    """
    Run a standard gradient step with clipping.
    """
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()
