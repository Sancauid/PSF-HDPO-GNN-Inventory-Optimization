# tests/data/test_datasets.py
from omegaconf import OmegaConf
import torch
from src.hdpo_gnn.data.datasets import create_synthetic_data_dict, create_pyg_dataset

def get_test_config():
    yaml_string = """
    problem_params:
      periods: 10
      graph:
        nodes:
          - { id: 0, features: { has_external_supply: 1, is_demand_facing: 0 } }
          - { id: 1, features: { has_external_supply: 0, is_demand_facing: 1 } }
        edges:
          - [0, 1]
    data_params:
      n_samples: 4
    """
    return OmegaConf.create(yaml_string)

def test_create_synthetic_data_dict():
    config = get_test_config()
    data = create_synthetic_data_dict(config)

    B = config.data_params.n_samples
    N = len(config.problem_params.graph.nodes)
    T = config.problem_params.periods
    E = len(config.problem_params.graph.edges)

    assert data['inventories'].shape == (B, N)
    assert data['demands'].shape == (T, B, N)
    assert data['node_features'].shape == (N, 2)
    assert data['edge_index'].shape == (2, E)
    # Test demand masking
    assert torch.all(data['demands'][:, :, 0] == 0), "Warehouse node should have zero demand"
    assert torch.mean(data['demands'][:, :, 1]) > 0, "Store node should have positive average demand"

def test_create_pyg_dataset():
    config = get_test_config()
    data_dict = create_synthetic_data_dict(config)
    pyg_list = create_pyg_dataset(data_dict, config)

    B = config.data_params.n_samples
    N = len(config.problem_params.graph.nodes)
    T = config.problem_params.periods

    assert len(pyg_list) == B
    g = pyg_list[0]
    assert torch.equal(g.x, data_dict['node_features'])
    assert torch.equal(g.edge_index, data_dict['edge_index'])
    assert g.initial_inventory.shape == (N,)
    assert g.demands.shape == (T, N)