# tests/training/test_end_to_end.py
import torch
from torch.autograd import gradcheck
from torch_geometric.data import Batch, Data

from src.hdpo_gnn.training.engine import run_simulation_episode, prepare_batch_for_simulation
from src.hdpo_gnn.engine.functional import transition_step

def test_end_to_end_gradcheck():
    DTYPE = torch.double
    DEVICE = torch.device('cpu')
    
    # 1. Setup a minimal, reproducible environment
    N, T, E = 2, 2, 1
    node_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    
    # Create a B=1 PyG Batch
    data = Data(
        x=node_features,
        edge_index=edge_index,
        initial_inventory=torch.tensor([10.0, 2.0], dtype=DTYPE),
        demands=torch.full((T, N), 5.0, dtype=DTYPE),
        holding_store=torch.tensor(1.0, dtype=DTYPE),
        underage_store=torch.tensor(9.0, dtype=DTYPE),
        holding_warehouse=torch.tensor(0.5, dtype=DTYPE)
    )
    data.demands = data.demands.unsqueeze(0) # Simulate Dataloader [B,T,N] shape
    pyg_batch = Batch.from_data_list([data]).to(DEVICE)
    
    # 2. Prepare data using the real function
    data_for_reset, batch_out = prepare_batch_for_simulation(pyg_batch, DEVICE)
    
    # 3. Define the function to be checked by gradcheck
    def gradcheck_func(edge_flows_input):
        # edge_flows_input is a flat tensor of all flows over the episode [T*E]
        
        class DummyModel(torch.nn.Module):
            def __init__(self, flows):
                super().__init__()
                self.flows = flows
                self.t = 0

            def forward(self, x, edge_index):
                # On each call, return the next flow from the input sequence
                flow_for_this_step = self.flows[self.t].unsqueeze(0).unsqueeze(-1)
                self.t += 1
                return {'flows': flow_for_this_step}

        model = DummyModel(edge_flows_input)
        
        total_cost, _ = run_simulation_episode(
            model, pyg_batch, data_for_reset, periods=T, ignore_periods=0
        )
        return total_cost.sum()

    # 4. Execute gradcheck
    # The input must be a flat tensor of flows for all timesteps
    flows_over_time = torch.randn(T * E, dtype=DTYPE, requires_grad=True)
    
    is_differentiable = gradcheck(gradcheck_func, (flows_over_time,), eps=1e-6, atol=1e-4)
    assert is_differentiable, "End-to-end gradcheck failed!"