import torch

from hdpo_gnn.training.engine import calculate_losses


def test_calculate_losses_normalization():
    total_cost = torch.tensor(1000.0)
    report_cost = torch.tensor(500.0)

    loss_bwd, loss_rep = calculate_losses(
        total_episode_cost=total_cost,
        cost_to_report=report_cost,
        total_real_stores=10,
        periods=50,
        ignore_periods=30,
    )

    assert loss_bwd.item() == 1000.0 / (10 * 50)
    assert loss_rep.item() == 500.0 / (10 * 20)
