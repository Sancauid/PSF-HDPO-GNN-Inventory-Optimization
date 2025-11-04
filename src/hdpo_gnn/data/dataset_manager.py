# src/hdpo_gnn/data/dataset_manager.py
import logging
from typing import Any, Dict, List, Tuple

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.hdpo_gnn.utils.environment_builder import Environment

log = logging.getLogger(__name__)

class PyGInventoryDataset(Dataset):
    """A custom PyTorch Geometric Dataset to hold a list of Data objects."""
    def __init__(self, data_list: List[Data]):
        super().__init__()
        self.data_list = data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

class DatasetManager:
    """
    Orchestrates data generation, splitting, and loading for HDPO experiments.

    This class reads the environment, data, and feature configurations to produce
    training, validation (dev), and test DataLoader objects, handling both
    synthetic and real data sources.
    """
    def __init__(self, env: Environment, data_config: DictConfig, features_config: DictConfig):
        log.info("--- Initializing Dataset Manager ---")
        log.debug(f"RECEIVED env of type {type(env)}, data_config, features_config.")

        self.env = env
        self.data_config = data_config
        self.features_config = features_config

        self.train_dataset: PyGInventoryDataset = None
        self.dev_dataset: PyGInventoryDataset = None
        self.test_dataset: PyGInventoryDataset = None

        self._prepare_datasets()
        log.info("--- Dataset Manager Initialized Successfully ---")

    def get_data_loaders(self) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
        """Returns the prepared train, dev, and test PyG DataLoaders."""
        log.debug("--> Calling get_data_loaders...")

        train_loader = PyGDataLoader(
            self.train_dataset, batch_size=self.data_config.batch_size, shuffle=True
        )
        dev_loader = PyGDataLoader(
            self.dev_dataset, batch_size=self.data_config.scenarios.dev, shuffle=False
        )
        test_loader = PyGDataLoader(
            self.test_dataset, batch_size=self.data_config.scenarios.test, shuffle=False
        )

        log.debug(f"<-- get_data_loaders RETURNING three PyGDataLoader instances.")
        return train_loader, dev_loader, test_loader

    def _prepare_datasets(self):
        """Main orchestrator for creating datasets based on the split policy."""
        log.debug("--> Calling _prepare_datasets...")
        split_policy = self.data_config.split_policy
        log.debug(f"Identified split_policy: '{split_policy}'")

        if split_policy == "random":
            self._create_synthetic_datasets()
        elif split_policy == "by_period":
            self._create_real_data_datasets()
        else:
            raise NotImplementedError(f"Split policy '{split_policy}' is not recognized.")
        log.debug("<-- _prepare_datasets COMPLETED.")

    def _create_synthetic_datasets(self):
        """Handles generation and splitting for synthetic data."""
        log.debug("--> Calling _create_synthetic_datasets...")

        # Generate one large set of scenarios for training and development
        num_train_dev_samples = self.data_config.scenarios.train + self.data_config.scenarios.dev
        log.info(f"Generating {num_train_dev_samples} synthetic scenarios for train/dev split...")
        train_dev_dict = self._generate_scenarios(
            num_samples=num_train_dev_samples,
            episode_length=max(self.data_config.episode_length.train, self.data_config.episode_length.dev)
        )

        # Generate a separate set of scenarios for testing (often longer episodes)
        log.info(f"Generating {self.data_config.scenarios.test} synthetic scenarios for test set...")
        test_dict = self._generate_scenarios(
            num_samples=self.data_config.scenarios.test,
            episode_length=self.data_config.episode_length.test
        )
        
        # Split the train/dev data by sample index
        log.debug("Splitting train/dev data by sample index...")
        split_idx = self.data_config.scenarios.train
        train_dict = {}
        dev_dict = {}

        for key, value in train_dev_dict.items():
            # Only split tensors that have a batch dimension matching num_train_dev_samples
            if isinstance(value, torch.Tensor) and value.shape[0] == num_train_dev_samples:
                log.debug(f"...... Splitting tensor '{key}' at index {split_idx}.")
                train_dict[key] = value[:split_idx]
                dev_dict[key] = value[split_idx:]
            else:
                # For everything else (static tensors like edge_index, or dicts like edge_params),
                # just copy it to both splits.
                log.debug(f"...... Copying static data '{key}' to both splits.")
                train_dict[key] = value
                dev_dict[key] = value

        # Convert dictionaries to PyG Datasets
        self.train_dataset = self._convert_dict_to_pyg_dataset(train_dict, self.data_config.episode_length.train)
        self.dev_dataset = self._convert_dict_to_pyg_dataset(dev_dict, self.data_config.episode_length.dev)
        self.test_dataset = self._convert_dict_to_pyg_dataset(test_dict, self.data_config.episode_length.test)
        
        log.debug("<-- _create_synthetic_datasets COMPLETED.")

    def _create_real_data_datasets(self):
        """Handles loading and splitting for real data."""
        log.debug("--> Calling _create_real_data_datasets...")
        # TODO: Implement this logic based on researcher's code
        # 1. Load the full demand tensor from file
        # 2. Generate per-sample parameters (e.g., underage_cost with 'uniform_per_sample')
        # 3. Create three separate data dictionaries (train, dev, test)
        # 4. In each dict, slice the demand tensor according to `data_config.periods`
        # 5. Convert each dict to a PyG dataset
        raise NotImplementedError("Real data loading is not yet implemented.")
        log.debug("<-- _create_real_data_datasets COMPLETED.")
        
    def _generate_scenarios(self, num_samples: int, episode_length: int) -> Dict[str, Any]:
        """
        Generates a dictionary of synthetic data tensors for a set of scenarios.
        This is an evolution of the original `create_synthetic_data_dict`.
        """
        log.debug(f"...... Calling _generate_scenarios. RECEIVED num_samples={num_samples}, episode_length={episode_length}")

        # --- Generate Demands ---
        demand_cfg = self.env.config.demand.synthetic_data_config
        
        # In a multi-store setting, demand is correlated
        is_demand_facing = self.env.static_node_features[:, self.env.node_type_map['store'] + 1] == 1
        store_indices = torch.where(is_demand_facing)[0]
        num_stores = len(store_indices)
        
        # Sample per-store mean and std from ranges
        mean_low, mean_high = demand_cfg.mean_range
        cv_low, cv_high = demand_cfg.cv_range
        store_means = torch.rand(num_stores) * (mean_high - mean_low) + mean_low
        store_cvs = torch.rand(num_stores) * (cv_high - cv_low) + cv_low
        store_stds = store_means * store_cvs

        stds_outer = store_stds.unsqueeze(1) * store_stds.unsqueeze(0)
        covariance_matrix = demand_cfg.correlation * stds_outer
        covariance_matrix.diagonal().copy_(store_stds**2)
        
        try:
            mvn = torch.distributions.MultivariateNormal(loc=store_means, covariance_matrix=covariance_matrix)
        except ValueError as e:
            log.error(f"Failed to create MultivariateNormal. Covariance matrix might not be positive semi-definite. Error: {e}")
            # Add a small value to the diagonal for numerical stability (jitter)
            jitter = 1e-6 * torch.eye(num_stores)
            mvn = torch.distributions.MultivariateNormal(loc=store_means, covariance_matrix=covariance_matrix + jitter)
            log.warning("Added jitter to covariance matrix to proceed.")

        store_demands_flat = mvn.rsample((num_samples * episode_length,))
        demands = torch.zeros(num_samples, episode_length, self.env.num_nodes)
        demands[:, :, store_indices] = store_demands_flat.view(num_samples, episode_length, num_stores)
        demands.clamp_(min=0.0)

        # --- Generate Initial Inventories ---
        # Using a simple exponential distribution for initial inventories as in the original code
        inventory_dist = torch.distributions.Exponential(rate=0.15)
        initial_inventories = inventory_dist.rsample((num_samples, self.env.num_nodes))
        
        # --- Generate per-sample parameters ---
        # This handles the 'uniform_per_sample' case from the researcher's code
        underage_costs = self.env.underage_costs.unsqueeze(0).expand(num_samples, -1).clone()
        p_cfg = self.env.config.parameters.nodes.stores.underage_cost
        if p_cfg.sampling_method == 'uniform_per_sample':
            log.debug("Sampling underage costs per sample...")
            per_sample_costs = torch.rand(num_samples, num_stores) * (p_cfg.range[1] - p_cfg.range[0]) + p_cfg.range[0]
            underage_costs[:, store_indices] = per_sample_costs

        data_dict = {
            "initial_inventories": initial_inventories, # Shape: [B, N]
            "demands": demands,                         # Shape: [B, T, N]
            "static_node_features": self.env.static_node_features, # Shape: [N, F]
            "edge_index": self.env.edge_index,          # Shape: [2, E]
            "holding_costs": self.env.holding_costs.unsqueeze(0).expand(num_samples, -1),
            "underage_costs": underage_costs,
            "edge_params": self.env.edge_params
        }
        log.debug(f"...... _generate_scenarios RETURNING data_dict with tensors of shapes: "
                  f"demands={demands.shape}, initial_inventories={initial_inventories.shape}")
        return data_dict

    def _convert_dict_to_pyg_dataset(self, data_dict: Dict[str, Any], episode_length: int) -> PyGInventoryDataset:
        """Converts a dictionary of tensors into a PyG Dataset object."""
        log.debug(f"...... Calling _convert_dict_to_pyg_dataset for episode_length={episode_length}")
        
        num_samples = data_dict["initial_inventories"].shape[0]
        data_list = []
        for i in range(num_samples):
            # Trim demands to the correct episode length for this dataset (train vs dev)
            demands_for_sample = data_dict["demands"][i, :episode_length, :]

            g = Data(
                x=data_dict["static_node_features"],
                edge_index=data_dict["edge_index"],
                initial_inventory=data_dict["initial_inventories"][i],
                demands=demands_for_sample, # Shape: [T, N]
                holding_costs=data_dict["holding_costs"][i],
                underage_costs=data_dict["underage_costs"][i],
                # TODO: A robust implementation would flatten edge_params into edge_attr
                # For now, we pass it as a Python attribute
                edge_params=data_dict["edge_params"]
            )
            data_list.append(g)

        log.debug(f"...... _convert_dict_to_pyg_dataset RETURNING PyGInventoryDataset with {len(data_list)} graphs.")
        return PyGInventoryDataset(data_list)