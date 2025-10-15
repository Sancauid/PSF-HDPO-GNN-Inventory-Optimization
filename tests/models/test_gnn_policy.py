import torch
from torch_geometric.data import Data

from hdpo_gnn.models.gnn import GNNPolicy


def test_gnn_policy_forward_pass():
  num_nodes = 5
  node_feature_size = 7
  output_size = 1

  x = torch.randn(num_nodes, node_feature_size)
  edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
  data = Data(x=x, edge_index=edge_index)

  model = GNNPolicy(node_feature_size=node_feature_size, output_size=output_size)
  out = model(data.x, data.edge_index)

  assert isinstance(out, dict)
  assert 'stores' in out
  assert out['stores'].shape == torch.Size([num_nodes, output_size])


