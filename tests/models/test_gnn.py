# tests/models/test_gnn.py
import torch
from torch_geometric.data import Batch, Data
from src.hdpo_gnn.models.gnn import GNNPolicy

def test_edge_gnn_policy_forward_pass():
    B, N, F, E = 4, 5, 2, 8
    model = GNNPolicy(node_feature_size=F, output_size=1)
    
    # Create a batch of graphs
    data_list = []
    for _ in range(B):
        x = torch.randn(N, F)
        edge_index = torch.randint(0, N, (2, E), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    
    batch = Batch.from_data_list(data_list)
    
    out = model(batch.x, batch.edge_index)
    
    assert isinstance(out, dict)
    assert "flows" in out
    assert out["flows"].shape == (B * E, 1)
