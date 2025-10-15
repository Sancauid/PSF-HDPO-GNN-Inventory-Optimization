import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from hdpo_gnn.training.engine import (
    calculate_losses,
    perform_gradient_step,
    prepare_batch_for_simulation,
    run_simulation_episode,
)


@pytest.mark.parametrize(
    "total_cost,report_cost,stores,periods,ignore,expected_bwd,expected_rep",
    [
        (
            torch.tensor([100.0, 300.0]),
            torch.tensor([50.0, 150.0]),
            10,
            50,
            30,
            (100.0 + 300.0) / 2 / (10 * 50),
            (50.0 + 150.0) / 2 / (10 * 20),
        ),
        (torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), 10, 10, 0, 0.0, 0.0),
        (
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            0,
            5,
            2,
            1.0,
            1.0,
        ),  # zero stores -> denom clamps to 1
    ],
)
def test_calculate_losses(
    total_cost, report_cost, stores, periods, ignore, expected_bwd, expected_rep
):
    loss_bwd, loss_rep = calculate_losses(
        total_cost,
        report_cost,
        total_real_stores=stores,
        periods=periods,
        ignore_periods=ignore,
    )
    assert loss_bwd.item() == pytest.approx(expected_bwd)
    assert loss_rep.item() == pytest.approx(expected_rep)


def test_prepare_vanilla_batch():
    B, S, W, T = 4, 3, 1, 5
    batch = {
        "inventories": {"stores": torch.zeros(B, S), "warehouses": torch.zeros(B, W)},
        "demands": torch.zeros(T, B, S),
        "cost_params": {
            "holding_store": torch.tensor(1.0),
            "underage_store": torch.tensor(1.0),
            "holding_warehouse": torch.tensor(1.0),
        },
        "lead_times": {"stores": 1, "warehouses": 1},
    }
    data_for_reset, store_mask = prepare_batch_for_simulation(
        batch, "vanilla", device=torch.device("cpu")
    )
    assert isinstance(data_for_reset, dict)
    assert data_for_reset["inventories"]["stores"].shape == torch.Size([B, S])
    assert data_for_reset["demands"].shape == torch.Size([T, B, S])
    assert store_mask.shape[0] == S


def test_prepare_gnn_batch():
    B, S, T = 2, 3, 4
    # Create a flattened PyG Batch where demands is [B*N, T]
    x = torch.randn(B * S, 1)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]])
    data_list = []
    flat_demands = []
    for i in range(B):
        node_demands = (
            torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(S, 1) + i * 100
        )
        flat_demands.append(node_demands)
        d = Data(x=x[i * S : (i + 1) * S], edge_index=edge_index)
        d.demands = node_demands  # [S, T]
        data_list.append(d)
    batch = Batch.from_data_list(data_list)

    data_for_reset, store_mask = prepare_batch_for_simulation(
        batch, "gnn", device=torch.device("cpu")
    )
    assert data_for_reset["demands"].shape == torch.Size([T, B, S])
    assert data_for_reset["inventories"]["stores"].shape == torch.Size([B, S])
    assert store_mask.shape[0] == S
    dense = data_for_reset["demands"]  # [T,B,S]
    # Check values for first graph
    assert torch.allclose(dense[:, 0, :].t(), flat_demands[0])


def test_perform_gradient_step():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return self.w * x

    model = TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor(2.0, requires_grad=False)
    y = model(x)
    loss = (y - 0.0) ** 2
    old_w = model.w.item()
    perform_gradient_step(loss, model, opt, grad_clip_norm=1.0)
    assert model.w.item() != old_w


def test_run_simulation_episode_logic():
    class DummyModel(nn.Module):
        def forward(self, *args, **kwargs):
            if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                x = args[0]
                if x.dim() == 2:
                    Bn = x.size(0)
                    return {"stores": torch.ones(Bn, 1)}
            return {"stores": torch.ones(1, 1), "warehouses": torch.ones(1, 1)}

    B, S, T = 2, 3, 50
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
        "pyg_batch": Batch(),
        "B": B,
        "N": S,
    }
    store_mask = torch.tensor([True, True, False])
    model = DummyModel()

    total_cost, cost_report = run_simulation_episode(
        model=model,
        simulator=None,
        batch=None,
        periods=T,
        ignore_periods=30,
        data_for_reset=data_for_reset,
        store_mask=store_mask,
        architecture="gnn",
    )

    assert total_cost.shape == (B,)
    assert cost_report.shape == (B,)
    assert total_cost.sum() > 0
    assert cost_report.sum() > 0
    assert not torch.isnan(total_cost).any()
    assert not torch.isnan(cost_report).any()
