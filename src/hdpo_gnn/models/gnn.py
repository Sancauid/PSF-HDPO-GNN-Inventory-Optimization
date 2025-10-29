from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv


class GNNPolicy(nn.Module):
    """
    An Edge GCN-based policy producing edge flow predictions.

    This model uses GCN layers to create node embeddings, then processes edge features
    (concatenated source and destination node embeddings) through an MLP to predict
    flow values for each edge. The architecture consists of two GCN layers for node
    representation learning, followed by an edge MLP for flow prediction.
    """

    def __init__(self, node_feature_size: int, output_size: int) -> None:
        """
        Initialize the Edge GNNPolicy.

        Args:
          node_feature_size: Number of features per node.
          output_size: Size of per-edge outputs (usually 1 for flow prediction).
        """
        super().__init__()
        hidden = 128
        
        # GCN layers for node embedding
        self.conv1 = GCNConv(node_feature_size, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.act = nn.ReLU()
        
        # Edge MLP for flow prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),  # Concatenated source + dest embeddings
            nn.ReLU(),
            nn.Linear(hidden, 1)  # Single flow value per edge
        )
        
        self.output_size = int(output_size)

    def forward(self, x: Tensor, edge_index: Tensor) -> dict[str, Tensor]:
        """
        Compute forward pass to predict edge flows.

        Args:
          x: Node features of shape [num_nodes, node_feature_size]. If using a
            Batch from PyG, this is the concatenated node matrix across graphs.
          edge_index: Graph connectivity in COO format of shape [2, num_edges].

        Returns:
          A dictionary with:
            - 'flows': tensor of edge flow predictions [num_edges, 1]
        """
        # Create node embeddings through GCN layers
        node_embeddings = self.conv1(x, edge_index)
        node_embeddings = self.act(node_embeddings)
        node_embeddings = self.conv2(node_embeddings, edge_index)
        node_embeddings = self.act(node_embeddings)
        
        # Extract source and destination nodes from edge_index
        row, col = edge_index[0], edge_index[1]  # [num_edges]
        
        # Create edge features by concatenating source and destination embeddings
        edge_features = torch.cat([
            node_embeddings[row],  # Source node embeddings [num_edges, hidden]
            node_embeddings[col]   # Destination node embeddings [num_edges, hidden]
        ], dim=1)  # [num_edges, 2 * hidden]
        
        # Predict edge flows through MLP
        edge_flows = self.edge_mlp(edge_features)  # [num_edges, 1]
        
        return {"flows": edge_flows}
