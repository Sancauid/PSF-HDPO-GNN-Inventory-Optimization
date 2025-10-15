# src/hdpo_gnn/utils/config_loader.py
"""
Utility functions for loading and merging configuration files.
"""
from typing import Any, Dict

import yaml


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a new dictionary with keys from base overridden by override.
    Nested dictionaries are merged recursively.
    """
    result: Dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = _deep_merge_dicts(base_value, override_value)
        else:
            result[key] = override_value
    return result


def _load_yaml(path: str) -> Dict[str, Any]:
    """
    Loads a YAML file and returns a mapping.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping in {path}")
    return data


def load_configs(setting_path: str, hyperparams_path: str) -> Dict[str, Any]:
    """
    Loads and merges a settings YAML file and a hyperparameters YAML file.
    Values from hyperparameters override values from settings. Nested mappings
    are merged recursively.
    """
    setting_config = _load_yaml(setting_path)
    hyperparams_config = _load_yaml(hyperparams_path)
    return _deep_merge_dicts(setting_config, hyperparams_config)
