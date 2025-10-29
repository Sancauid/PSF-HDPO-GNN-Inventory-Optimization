# tests/utils/test_graph_parser.py
from omegaconf import OmegaConf
import torch
from src.hdpo_gnn.utils.graph_parser import parse_graph_topology

def test_parse_graph_topology():
    yaml_string = """
    problem_params:
      graph:
        nodes:
          - { id: 0, type: 'warehouse', features: { has_external_supply: 1, is_demand_facing: 0 } }
          - { id: 1, type: 'store',     features: { has_external_supply: 0, is_demand_facing: 1 } }
          - { id: 2, type: 'store',     features: { has_external_supply: 0, is_demand_facing: 1 } }
        edges:
          - [0, 1]
          - [0, 2]
    """
    config = OmegaConf.create(yaml_string)
    node_features, edge_index = parse_graph_topology(config)

    assert node_features.shape == (3, 2)
    expected_features = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
    assert torch.equal(node_features, expected_features)

    assert edge_index.shape == (2, 2)
    expected_edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    assert torch.equal(edge_index, expected_edge_index)
