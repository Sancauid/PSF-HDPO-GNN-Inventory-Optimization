import torch
from torch_geometric.data import Data

from hdpo_gnn.data.datasets import create_pyg_dataset, create_synthetic_data_dict


def test_create_synthetic_data_dict():
    dummy_config = {
        "problem_params": {
            "n_stores": 3,
            "n_warehouses": 2,
            "periods": 5,
        },
        "data_params": {
            "n_samples": 4,
        },
    }

    data = create_synthetic_data_dict(dummy_config)

    assert isinstance(data, dict)
    for key in ["inventories", "demands", "cost_params", "lead_times"]:
        assert key in data

    B = dummy_config["data_params"]["n_samples"]
    S = dummy_config["problem_params"]["n_stores"]
    W = dummy_config["problem_params"]["n_warehouses"]
    T = dummy_config["problem_params"]["periods"]

    inv = data["inventories"]
    assert isinstance(inv, dict)
    assert "stores" in inv and "warehouses" in inv
    assert inv["stores"].shape == torch.Size([B, S])
    assert inv["warehouses"].shape == torch.Size([B, W])

    demands = data["demands"]
    assert isinstance(demands, torch.Tensor)
    assert demands.shape == torch.Size([T, B, S])
    assert torch.all(demands >= 0)

    cost_params = data["cost_params"]
    assert isinstance(cost_params, dict)
    for key in ["holding_store", "underage_store", "holding_warehouse"]:
        assert key in cost_params and isinstance(cost_params[key], torch.Tensor)

    lead_times = data["lead_times"]
    assert isinstance(lead_times, dict)
    assert lead_times == {"stores": 2, "warehouses": 3}


def test_create_pyg_dataset():
    dummy_config = {
        "problem_params": {
            "n_stores": 3,
            "n_warehouses": 1,
            "periods": 5,
        },
        "data_params": {
            "n_samples": 4,
        },
    }

    data_dict = create_synthetic_data_dict(dummy_config)
    data_list = create_pyg_dataset(data_dict, dummy_config)

    assert isinstance(data_list, list)
    assert len(data_list) == dummy_config["data_params"]["n_samples"]
    assert isinstance(data_list[0], Data)

    g0 = data_list[0]
    assert hasattr(g0, "x") and hasattr(g0, "edge_index") and hasattr(g0, "demands")
    assert (
        hasattr(g0, "holding_store")
        and hasattr(g0, "underage_store")
        and hasattr(g0, "holding_warehouse")
    )
    assert hasattr(g0, "lead_time_stores") and hasattr(g0, "lead_time_warehouses")

    S = dummy_config["problem_params"]["n_stores"]
    assert g0.x.shape == torch.Size([S, 1])
    assert g0.edge_index.shape[0] == 2
    assert g0.edge_index.shape[1] == S * (S - 1)
