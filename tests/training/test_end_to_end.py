"""
End-to-end tests for the differentiable RL training framework.

These tests validate both the mathematical accumulation logic of the simulation
and the gradient flow through the entire stack to ensure no tensors are
detached inadvertently.
"""

import pytest
import torch
from torch import nn

from hdpo_gnn.models.vanilla import VanillaPolicy
from hdpo_gnn.training.engine import (
    calculate_losses,
    run_simulation_episode,
)


def test_simulation_logic_with_mocks():
    """
    Verifies the episode cost accumulation and reporting logic.
    Validates that the model is called once per period and that
    cost shapes are correct.
    """

    class MockModel(nn.Module):
        def __init__(self, B: int, S: int) -> None:
            super().__init__()
            self.B = B
            self.S = S
            self.calls = 0

        def forward(self, x: torch.Tensor) -> dict:
            self.calls += 1
            return {
                "stores": torch.ones(self.B, self.S, dtype=x.dtype, device=x.device),
                "warehouses": torch.ones(self.B, 1, dtype=x.dtype, device=x.device),
            }

    B, S, T = 2, 3, 40
    ignore = 10

    inventories = {"stores": torch.full((B, S), 100.0), "warehouses": torch.zeros(B, 1)}
    demands = torch.zeros(T, B, S)
    data_for_reset = {
        "inventories": inventories,
        "demands": demands,
        "cost_params": {
            "holding_store": torch.tensor(1.0),
            "underage_store": torch.tensor(1.0),
            "holding_warehouse": torch.tensor(1.0),
        },
        "lead_times": {"stores": 0, "warehouses": 0},
    }
    store_mask = torch.tensor([True, True, False])
    model = MockModel(B, S)

    total_cost, cost_report = run_simulation_episode(
        model=model,
        simulator=None,
        batch=None,
        periods=T,
        ignore_periods=ignore,
        data_for_reset=data_for_reset,
        store_mask=store_mask,
        architecture="vanilla",
    )

    assert model.calls == T
    assert total_cost.shape == (B,)
    assert cost_report.shape == (B,)
    assert not torch.isnan(total_cost).any()
    assert not torch.isnan(cost_report).any()


def test_gradient_flow():
    """
    Numerical gradient check to ensure loss backpropagates through the entire
    pipeline without any detached tensors. This prevents silent gradient breaks
    which would cause zero gradients during training.
    """
    torch.manual_seed(0)
    B, S, W, T = 1, 2, 1, 2
    arch = "vanilla"

    # Build fixed data (double precision) to exercise differentiability
    inventories = {
        "stores": torch.ones(B, S, dtype=torch.double),
        "warehouses": torch.zeros(B, W, dtype=torch.double),
    }
    demands = torch.full((T, B, S), 1.0, dtype=torch.double)
    cost_params = {
        "holding_store": torch.tensor(1.0, dtype=torch.double),
        "underage_store": torch.tensor(1.0, dtype=torch.double),
        "holding_warehouse": torch.tensor(0.5, dtype=torch.double),
    }
    lead_times = {"stores": 0, "warehouses": 0}
    store_mask = torch.ones(S, dtype=torch.bool)

    def check_fn(weight: torch.Tensor) -> torch.Tensor:
        base = VanillaPolicy(input_size=S + W, output_size=S + 1).double()
        for p in base.parameters():
            p.requires_grad = False

        class WrappedModel(nn.Module):
            def __init__(
                self, base_model: VanillaPolicy, injected_weight: torch.Tensor
            ) -> None:
                super().__init__()
                self.base = base_model
                self.injected_weight = injected_weight

            def forward(self, x: torch.Tensor) -> dict:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                z0 = torch.nn.functional.linear(
                    x,
                    self.injected_weight,
                    bias=torch.zeros(128, dtype=x.dtype, device=x.device),
                )
                # Use a simple, fixed projection to S + 1 outputs to keep the mapping well-conditioned
                stores_logits = z0[:, :S]
                warehouse_logits = z0[:, S : S + 1]
                return {"stores": stores_logits, "warehouses": warehouse_logits}

        model = WrappedModel(base, weight)

        data_for_reset = {
            "inventories": inventories,
            "demands": demands,
            "cost_params": cost_params,
            "lead_times": lead_times,
        }

        total_cost, cost_report = run_simulation_episode(
            model=model,
            simulator=None,
            batch=None,
            periods=T,
            ignore_periods=0,
            data_for_reset=data_for_reset,
            store_mask=store_mask,
            architecture=arch,
        )
        loss_bwd, _ = calculate_losses(
            total_episode_cost=total_cost,
            cost_to_report=cost_report,
            total_real_stores=S,
            periods=T,
            ignore_periods=0,
        )
        return loss_bwd

    # Initialize a valid double-precision weight for the first Linear layer
    tmp_model = VanillaPolicy(input_size=S + W, output_size=S + 1).double()
    weight0 = tmp_model.net[0].weight.detach().clone().requires_grad_(True)

    assert torch.autograd.gradcheck(
        check_fn, (weight0,), eps=1e-6, atol=1e-4, raise_exception=True
    )
