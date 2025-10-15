# train.py
"""
Main entry point for training HDPO policy networks.
"""
from typing import Dict, Any
import argparse
import pprint
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.hdpo_gnn.utils.config_loader import load_configs
from src.hdpo_gnn.data.datasets import create_synthetic_data_dict
from src.hdpo_gnn.engine.simulator import DifferentiableSimulator
from src.hdpo_gnn.models import VanillaPolicy, GNNPolicy
from src.hdpo_gnn.training.trainer import Trainer

def main() -> None:
    """
    Parses CLI arguments, loads two YAML configuration files, merges them,
    and prints the resulting configuration mapping.
    """
    parser = argparse.ArgumentParser(description="Train an HDPO policy network.")
    parser.add_argument("setting_file", help="Path to the settings config file.")
    parser.add_argument("hyperparams_file", help="Path to the hyperparams config file.")
    parser.add_argument("--model", choices=["vanilla", "gnn"], default=None, help="Model architecture override (defaults to config)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    args = parser.parse_args()

    config: Dict[str, Any] = load_configs(args.setting_file, args.hyperparams_file)
    pprint.pprint(config)

    data = create_synthetic_data_dict(config)
    print("\nSynthetic data generated successfully.")
    inv = data["inventories"]
    print(f"inventories['stores'].shape = {tuple(inv['stores'].shape)}")
    print(f"inventories['warehouses'].shape = {tuple(inv['warehouses'].shape)}")
    print(f"demands.shape = {tuple(data['demands'].shape)}")

    # Build a minimal dataset/dataloader for training loop
    batch = {
        "inventories": inv,
        "demands": data["demands"],
        "cost_params": data["cost_params"],
        "lead_times": data["lead_times"],
    }
    # Wrap in a list to form an iterable of one batch
    train_loader = [batch]

    arch = (args.model or config.get("model_params", {}).get("architecture", "vanilla")).lower()
    if arch == "vanilla":
        input_size = int(config["problem_params"]["n_stores"]) + int(config["problem_params"]["n_warehouses"])
        output_size = int(config["problem_params"]["n_stores"]) + 1
        model = VanillaPolicy(input_size=input_size, output_size=output_size)
    else:
        node_feature_size = 8
        output_size = 1
        model = GNNPolicy(node_feature_size=node_feature_size, output_size=output_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("optimizer_params", {}).get("learning_rate", 3e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    simulator = DifferentiableSimulator()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        simulator=simulator,
        train_loader=train_loader,
        configs=config,
    )
    trainer.train(epochs=args.epochs)

if __name__ == '__main__':
    main()