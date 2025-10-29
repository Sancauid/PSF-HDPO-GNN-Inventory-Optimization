# train.py
"""
Main entry point for training HDPO GNN policy networks.
"""
import argparse
import pprint
from typing import Any, Dict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.hdpo_gnn.data.datasets import create_pyg_dataset, create_synthetic_data_dict
from src.hdpo_gnn.models import GNNPolicy
from src.hdpo_gnn.training.trainer import Trainer
from src.hdpo_gnn.utils.config_loader import load_configs_as_dictconfig


def main() -> None:
    """
    Parses CLI arguments, loads two YAML configuration files, merges them,
    and trains a GNN policy network.
    """
    parser = argparse.ArgumentParser(description="Train an HDPO GNN policy network.")
    parser.add_argument("setting_file", help="Path to the settings config file.")
    parser.add_argument("hyperparams_file", help="Path to the hyperparams config file.")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    args = parser.parse_args()

    # Load configs using OmegaConf
    config = load_configs_as_dictconfig(args.setting_file, args.hyperparams_file)
    pprint.pprint(config)

    # Create synthetic data and PyG dataset
    raw = create_synthetic_data_dict(config)
    pyg_list = create_pyg_dataset(raw, config)

    # Use mini-batching and shuffling for stochasticity
    batch_size = int(config.data_params.get("batch_size", 256))
    train_loader = PyGDataLoader(pyg_list, batch_size=batch_size, shuffle=True)

    # Create GNN model
    # Add 2 to the feature size for the dynamic (per-timestep) features:
    # 1. Current inventory at the node
    # 2. Current demand at the node
    node_feature_size = pyg_list[0].num_node_features + 2
    output_size = 1  # Edge flow prediction
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
