from typing import Tuple
import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import GCNConv


class GNNPolicy(nn.Module):
  """
  A simple GCN-based policy producing per-node outputs.

  Two layers of GCNConv with ReLU in between. The network takes node features
  and an edge_index (COO) graph structure from PyG and returns updated node
  embeddings. A final linear layer maps to per-node outputs of size 1.
  """

  def __init__(self, node_feature_size: int, output_size: int) -> None:
    """
    Initialize the GNNPolicy.

    Args:
      node_feature_size: Number of features per node.
      output_size: Size of per-node outputs (usually 1).
    """
    super().__init__()
    hidden = 128
    self.conv1 = GCNConv(node_feature_size, hidden)
    self.conv2 = GCNConv(hidden, hidden)
    self.act = nn.ReLU()
    self.head = nn.Linear(hidden, output_size)
    self.output_size = int(output_size)

  def forward(self, x: Tensor, edge_index: Tensor) -> dict[str, Tensor]:
    """
    Compute forward pass on a (batched) PyG graph.

    Args:
      x: Node features of shape [num_nodes, node_feature_size]. If using a
        Batch from PyG, this is the concatenated node matrix across graphs.
      edge_index: Graph connectivity in COO format of shape [2, num_edges].

    Returns:
      A dictionary with:
        - 'stores': tensor of per-node outputs [num_nodes, output_size]
    """
    x = self.conv1(x, edge_index)
    x = self.act(x)
    x = self.conv2(x, edge_index)
    x = self.act(x)
    out = self.head(x)
    return {"stores": out}


