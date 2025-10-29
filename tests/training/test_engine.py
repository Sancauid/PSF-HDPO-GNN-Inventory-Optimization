# tests/training/test_engine.py
import torch
from src.hdpo_gnn.training.engine import calculate_losses

def test_calculate_losses():
    total_cost = torch.tensor([1000.0, 1200.0]) # B=2
    report_cost = torch.tensor([500.0, 500.0])
    num_nodes = 4
    periods = 50
    ignore = 30
    
    loss_bwd, loss_rep = calculate_losses(
        total_cost, report_cost, num_nodes, periods, ignore
    )
    
    # New behavior: simple mean without normalization
    expected_bwd = 1100.0  # mean of [1000, 1200]
    expected_rep = 500.0   # mean of [500, 500]
    
    assert torch.isclose(loss_bwd, torch.tensor(expected_bwd))
    assert torch.isclose(loss_rep, torch.tensor(expected_rep))