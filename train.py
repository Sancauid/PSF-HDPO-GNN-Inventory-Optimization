# train.py
"""
Main entry point for training HDPO policy networks.
"""
from typing import Dict, Any
import argparse
import pprint

from src.hdpo_gnn.utils.config_loader import load_configs
from src.hdpo_gnn.data.datasets import create_synthetic_data_dict

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

if __name__ == '__main__':
    main()