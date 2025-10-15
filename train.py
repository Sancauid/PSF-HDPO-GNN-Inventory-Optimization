# train.py
"""
Main entry point for training HDPO policy networks.
"""
from typing import Dict, Any
import argparse
import pprint

from src.hdpo_gnn.utils.config_loader import load_configs

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

if __name__ == '__main__':
    main()