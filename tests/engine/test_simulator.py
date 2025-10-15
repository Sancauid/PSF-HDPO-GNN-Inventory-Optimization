import torch

from hdpo_gnn.engine.simulator import DifferentiableSimulator


def test_simulator_step():
    B, S, W, T = 2, 2, 1, 1

    inventories = {
        "stores": torch.zeros(B, S),
        "warehouses": torch.tensor([[100.0], [50.0]]),
    }
    demands = torch.zeros(T, B, S)

    cost_params = {
        "holding_store": 0.1,
        "underage_store": 1.0,
        "holding_warehouse": 0.05,
    }
    lead_times = {"stores": 0, "warehouses": 0}

    sim = DifferentiableSimulator()
    sim.reset(
        inventories=inventories,
        demands=demands,
        cost_params=cost_params,
        lead_times=lead_times,
    )

    actions = {
        "stores": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "warehouses": torch.zeros(B, W),
    }
    obs, cost = sim.step(actions)

    assert isinstance(cost, torch.Tensor)
    assert cost.shape == torch.Size([B])

    expected_store_inventory = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(obs["inventory_stores"], expected_store_inventory)
