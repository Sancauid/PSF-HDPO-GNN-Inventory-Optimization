import torch

from hdpo_gnn.training.trainer import Trainer


def test_calculate_losses_normalization():
    trainer = Trainer(  # minimal instance; unused args can be None
        model=None,  # type: ignore[arg-type]
        optimizer=None,  # type: ignore[arg-type]
        scheduler=None,  # type: ignore[arg-type]
        train_loader=[],  # type: ignore[arg-type]
        configs={
            "problem_params": {"n_stores": 10, "periods": 50, "ignore_periods": 30},
            "data_params": {"n_samples": 1},
            "model_params": {"architecture": "vanilla"},
        },
    )

    total_cost = torch.tensor(1000.0)
    report_cost = torch.tensor(500.0)

    loss_bwd, loss_rep = trainer._calculate_losses(
        total_episode_cost=total_cost,
        cost_to_report=report_cost,
        num_real_stores=10,
        periods=50,
        ignore_periods=30,
    )

    assert loss_bwd.item() == 1000.0 / (10 * 50)
    assert loss_rep.item() == 500.0 / (10 * 20)
