"""
Graph topology parser for converting YAML-defined graphs to PyTorch tensors.
"""
from typing import Any, Dict, List, Tuple

import torch
from omegaconf import DictConfig, OmegaConf


def parse_graph_topology(config: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse graph topology from YAML config and create PyTorch tensors.
    
    Args:
        config: OmegaConf DictConfig containing graph topology under 'problem_params.graph'
        
    Returns:
        node_features: Tensor of shape [num_nodes, num_features] 
                      Feature order: [has_external_supply, is_demand_facing]
        edge_index: Tensor of shape [2, num_edges] in COO format
        
    Example YAML structure:
        problem_params:
          graph:
            nodes:
              - { id: 0, type: 'warehouse', features: { has_external_supply: 1, is_demand_facing: 0 } }
              - { id: 1, type: 'store',     features: { has_external_supply: 0, is_demand_facing: 1 } }
            edges:
              - [0, 1]
              - [0, 2]
    """
    graph_config = config.problem_params.graph
    
    # Parse nodes
    nodes = graph_config.nodes
    num_nodes = len(nodes)
    
    # Extract node features in fixed order: [has_external_supply, is_demand_facing]
    node_features = torch.zeros(num_nodes, 2, dtype=torch.float32)
    
    # Create node_id to index mapping
    node_id_to_idx = {}
    for i, node in enumerate(nodes):
        node_id = node.id
        node_id_to_idx[node_id] = i
        
        # Extract features in fixed order
        features = node.features
        node_features[i, 0] = float(features.has_external_supply)
        node_features[i, 1] = float(features.is_demand_facing)
    
    # Parse edges
    edges = graph_config.edges
    num_edges = len(edges)
    
    if num_edges == 0:
        # No edges - return empty edge_index
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        # Convert edge list to COO format
        edge_list = []
        for edge in edges:
            src_id, dst_id = edge
            if src_id not in node_id_to_idx or dst_id not in node_id_to_idx:
                raise ValueError(f"Edge [{src_id}, {dst_id}] references non-existent node IDs")
            src_idx = node_id_to_idx[src_id]
            dst_idx = node_id_to_idx[dst_id]
            edge_list.append([src_idx, dst_idx])
        
        # Convert to tensor [2, num_edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return node_features, edge_index


def get_node_type_counts(config: DictConfig) -> Dict[str, int]:
    """
    Count nodes by type from the graph configuration.
    
    Args:
        config: OmegaConf DictConfig containing graph topology
        
    Returns:
        Dictionary mapping node types to counts
    """
    graph_config = config.problem_params.graph
    nodes = graph_config.nodes
    
    type_counts = {}
    for node in nodes:
        node_type = node.type
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    return type_counts


def validate_graph_topology(config: DictConfig) -> None:
    """
    Validate the graph topology configuration.
    
    Args:
        config: OmegaConf DictConfig containing graph topology
        
    Raises:
        ValueError: If the graph topology is invalid
    """
    if not hasattr(config, 'problem_params') or not hasattr(config.problem_params, 'graph'):
        raise ValueError("Config must contain 'problem_params.graph'")
    
    graph_config = config.problem_params.graph
    
    if not hasattr(graph_config, 'nodes') or not hasattr(graph_config, 'edges'):
        raise ValueError("Graph config must contain 'nodes' and 'edges'")
    
    # Validate nodes
    nodes = graph_config.nodes
    if len(nodes) == 0:
        raise ValueError("Graph must have at least one node")
    
    node_ids = set()
    for i, node in enumerate(nodes):
        if not hasattr(node, 'id'):
            raise ValueError(f"Node {i} missing 'id' field")
        if not hasattr(node, 'type'):
            raise ValueError(f"Node {i} missing 'type' field")
        if not hasattr(node, 'features'):
            raise ValueError(f"Node {i} missing 'features' field")
        
        node_id = node.id
        if node_id in node_ids:
            raise ValueError(f"Duplicate node ID: {node_id}")
        node_ids.add(node_id)
        
        # Validate features
        features = node.features
        if not hasattr(features, 'has_external_supply'):
            raise ValueError(f"Node {node_id} missing 'has_external_supply' feature")
        if not hasattr(features, 'is_demand_facing'):
            raise ValueError(f"Node {node_id} missing 'is_demand_facing' feature")
    
    # Validate edges
    edges = graph_config.edges
    for i, edge in enumerate(edges):
        if len(edge) != 2:
            raise ValueError(f"Edge {i} must have exactly 2 elements")
        src_id, dst_id = edge
        if src_id not in node_ids:
            raise ValueError(f"Edge {i} references non-existent source node: {src_id}")
        if dst_id not in node_ids:
            raise ValueError(f"Edge {i} references non-existent destination node: {dst_id}")


def load_config_with_graph(setting_path: str, hyperparams_path: str) -> Tuple[DictConfig, torch.Tensor, torch.Tensor]:
    """
    Load configuration files and parse graph topology.
    
    Args:
        setting_path: Path to settings YAML file
        hyperparams_path: Path to hyperparameters YAML file
        
    Returns:
        config: OmegaConf DictConfig
        node_features: Tensor of shape [num_nodes, num_features]
        edge_index: Tensor of shape [2, num_edges]
    """
    # Load configs using OmegaConf
    config = OmegaConf.load(setting_path)
    hyperparams = OmegaConf.load(hyperparams_path)
    config = OmegaConf.merge(config, hyperparams)
    
    # Validate and parse graph topology
    validate_graph_topology(config)
    node_features, edge_index = parse_graph_topology(config)
    
    return config, node_features, edge_index
