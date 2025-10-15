# train.py
"""
Main entry point for training HDPO policy networks.
"""
import argparse
import pprint
from typing import Any, Dict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.hdpo_gnn.data.datasets import create_pyg_dataset, create_synthetic_data_dict
from src.hdpo_gnn.models import GNNPolicy, VanillaPolicy
from src.hdpo_gnn.training.trainer import Trainer
from src.hdpo_gnn.utils.config_loader import load_configs


def main() -> None:
    """
    Parses CLI arguments, loads two YAML configuration files, merges them,
    and prints the resulting configuration mapping.
    """
    parser = argparse.ArgumentParser(description="Train an HDPO policy network.")
    parser.add_argument("setting_file", help="Path to the settings config file.")
    parser.add_argument("hyperparams_file", help="Path to the hyperparams config file.")
    parser.add_argument(
        "--model",
        choices=["vanilla", "gnn"],
        default=None,
        help="Model architecture override (defaults to config)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    args = parser.parse_args()

    config: Dict[str, Any] = load_configs(args.setting_file, args.hyperparams_file)
    pprint.pprint(config)

    arch = (
        args.model or config.get("model_params", {}).get("architecture", "vanilla")
    ).lower()
    if arch == "gnn":
        raw = create_synthetic_data_dict(config)
        pyg_list = create_pyg_dataset(raw, config)
        train_loader = PyGDataLoader(pyg_list, batch_size=len(pyg_list), shuffle=False)
    else:
        raw = create_synthetic_data_dict(config)
        stores = raw["inventories"]["stores"]
        warehouses = raw["inventories"]["warehouses"]
        demands_btS = raw["demands"].permute(1, 0, 2)
        dataset = TensorDataset(stores, warehouses, demands_btS)

        def collate_fn(samples):
            s_list, w_list, d_list = zip(*samples)
            s = torch.stack(s_list, dim=0)
            w = torch.stack(w_list, dim=0)
            d_btS = torch.stack(d_list, dim=0)
            demands = d_btS.permute(1, 0, 2)
            return {
                "inventories": {"stores": s, "warehouses": w},
                "demands": demands,
                "cost_params": raw["cost_params"],
                "lead_times": raw["lead_times"],
            }

        batch_size = int(config.get("data_params", {}).get("n_samples", stores.size(0)))
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

    if arch == "vanilla":
        input_size = int(config["problem_params"]["n_stores"]) + int(
            config["problem_params"]["n_warehouses"]
        )
        output_size = int(config["problem_params"]["n_stores"]) + 1
        model = VanillaPolicy(input_size=input_size, output_size=output_size)
    else:
        node_feature_size = 1
        output_size = 1
        model = GNNPolicy(node_feature_size=node_feature_size, output_size=output_size)

    optimizer = Adam(
        model.parameters(),
        lr=float(config.get("optimizer_params", {}).get("learning_rate", 3e-4)),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        configs=config,
    )
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
