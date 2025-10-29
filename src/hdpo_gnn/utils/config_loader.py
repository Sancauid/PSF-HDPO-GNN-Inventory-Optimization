# src/hdpo_gnn/utils/config_loader.py
"""
Utility functions for loading and merging configuration files using OmegaConf.
"""
from typing import Any, Dict, Union

from omegaconf import DictConfig, OmegaConf


def load_configs(setting_path: str, hyperparams_path: str) -> Union[Dict[str, Any], DictConfig]:
    """
    Loads and merges a settings YAML file and a hyperparameters YAML file using OmegaConf.
    Values from hyperparameters override values from settings. Nested mappings
    are merged recursively.
    
    Args:
        setting_path: Path to the settings YAML file
        hyperparams_path: Path to the hyperparameters YAML file
        
    Returns:
        Merged configuration as DictConfig (or dict for backward compatibility)
    """
    # Load configs using OmegaConf
    setting_config = OmegaConf.load(setting_path)
    hyperparams_config = OmegaConf.load(hyperparams_path)
    
    # Merge configurations (hyperparams override settings)
    merged_config = OmegaConf.merge(setting_config, hyperparams_config)
    
    # Convert to dict for backward compatibility with existing code
    return OmegaConf.to_container(merged_config, resolve=True)


def load_configs_as_dictconfig(setting_path: str, hyperparams_path: str) -> DictConfig:
    """
    Loads and merges configuration files, returning OmegaConf DictConfig.
    
    Args:
        setting_path: Path to the settings YAML file
        hyperparams_path: Path to the hyperparameters YAML file
        
    Returns:
        Merged configuration as OmegaConf DictConfig
    """
    setting_config = OmegaConf.load(setting_path)
    hyperparams_config = OmegaConf.load(hyperparams_path)
    return OmegaConf.merge(setting_config, hyperparams_config)
