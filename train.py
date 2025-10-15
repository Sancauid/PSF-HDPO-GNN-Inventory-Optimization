# train.py
"""
Main entry point for training HDPO policy networks.
"""
from typing import Dict, Any
import argparse
import pprint
import torch

from src.hdpo_gnn.utils.config_loader import load_configs
from src.hdpo_gnn.data.datasets import create_synthetic_data_dict
from src.hdpo_gnn.engine.simulator import DifferentiableSimulator

def main() -> None:
    """
    Parses CLI arguments, loads two YAML configuration files, merges them,
    and prints the resulting configuration mapping.
    """
    parser = argparse.ArgumentParser(description="Train an HDPO policy network.")
    parser.add_argument("setting_file", help="Path to the settings config file.")
    parser.add_argument("hyperparams_file", help="Path to the hyperparams config file.")
    args = parser.parse_args()

    config: Dict[str, Any] = load_configs(args.setting_file, args.hyperparams_file)
    pprint.pprint(config)

    data = create_synthetic_data_dict(config)
    print("\nSynthetic data generated successfully.")
    inv = data["inventories"]
    print(f"inventories['stores'].shape = {tuple(inv['stores'].shape)}")
    print(f"inventories['warehouses'].shape = {tuple(inv['warehouses'].shape)}")
    print(f"demands.shape = {tuple(data['demands'].shape)}")
    cp = data["cost_params"]
    print(f"cost_params.holding_store.shape = {tuple(cp['holding_store'].shape)}")
    print(f"cost_params.underage_store.shape = {tuple(cp['underage_store'].shape)}")
    print(f"cost_params.holding_warehouse.shape = {tuple(cp['holding_warehouse'].shape)}")

    simulator = DifferentiableSimulator()
    observation = simulator.reset(
        inventories=data["inventories"],
        demands=data["demands"],
        cost_params=data["cost_params"],
        lead_times=data["lead_times"],
    )

    periods: int = int(config["problem_params"]["periods"])
    batch_size: int = int(config["data_params"]["n_samples"])
    n_stores: int = int(config["problem_params"]["n_stores"])
    n_warehouses: int = int(config["problem_params"]["n_warehouses"]) 

    for t in range(periods):
        actions = {
        "stores": torch.zeros(batch_size, n_stores),
        "warehouses": torch.zeros(batch_size, n_warehouses),
        }
        new_observation, cost = simulator.step(actions)
        print(f"t={t}, Cost shape: {cost.shape}, Avg cost: {cost.mean().item():.4f}")
        observation = new_observation

    print("\nEpisode simulation completed successfully.")

if __name__ == '__main__':
    main()