# src/hdpo_gnn/models/factory.py
import logging
from typing import List

import torch.nn as nn
from omegaconf import DictConfig

from src.hdpo_gnn.utils.environment_builder import Environment
from .paper_gnn import PaperGNNPolicy

log = logging.getLogger(__name__)

class ModelFactory:
    """
    Creates and configures the policy network model based on the experiment config.

    This factory is responsible for:
    1. Selecting the correct model class (e.g., PaperGNNPolicy, Vanilla MLP).
    2. Dynamically calculating the input feature dimensions based on the config.
    3. Instantiating the model with the correct hyperparameters.
    """
    def __init__(self, model_config: DictConfig, features_config: DictConfig, env: Environment):
        log.info("--- Initializing Model Factory ---")
        log.debug(f"RECEIVED model_config, features_config, and env of type {type(env)}.")
        
        self.model_config = model_config
        self.features_config = features_config
        self.env = env
        
        self.architecture_class = self.model_config.architecture_class
        log.info(f"Target model architecture: '{self.architecture_class}'")

    def create_model(self) -> nn.Module:
        """
        Instantiates and returns the configured PyTorch model.
        """
        log.debug("--> Calling create_model...")

        if self.architecture_class == "gnn" or self.architecture_class == "paper_gnn":
            if self.architecture_class == "gnn":
                log.info("Architecture 'gnn' requested, building PaperGNNPolicy for faithful replication.")
            model = self._create_paper_gnn_policy()
        else:
            raise NotImplementedError(
                f"Model architecture '{self.architecture_class}' is not supported by the factory."
            )
        
        log.debug(f"<-- create_model RETURNING model of type {type(model)}")
        return model

    def _calculate_node_feature_size(self) -> int:
        """
        Dynamically and accurately calculates the size of the input node feature vector.
        """
        log.debug("...... Calling _calculate_node_feature_size...")
        
        size = 0
        
        # Static features from the environment (one-hot encoding of node type)
        static_size = self.env.static_node_features.shape[1]
        size += static_size
        log.debug(f"......... [1] Base size from static features (node type one-hot): {static_size}. Total: {size}")
        
        # Dynamic features from the simulation state
        dynamic_features: List[str] = self.features_config.dynamic
        if "inventory_on_hand" in dynamic_features:
            size += 1
            log.debug(f"......... [2] Added 1 for dynamic feature: 'inventory_on_hand'. Total: {size}")
        if "outstanding_orders" in dynamic_features:
            max_lead_time = 0
            for edge_p in self.env.edge_params.values():
                if 'lead_time' in edge_p:
                    # Ensure lead_time is an integer
                    max_lead_time = max(max_lead_time, int(edge_p['lead_time']))
            size += int(max_lead_time)
            log.debug(f"......... [3] Added {int(max_lead_time)} for dynamic feature: 'outstanding_orders'. Total: {size}")

        # Static parameters passed as features to the model (for meta-learning)
        static_pass_features: List[str] = self.features_config.static
        if "holding_cost" in static_pass_features:
            size += 1
            log.debug(f"......... [4] Added 1 for static feature: 'holding_cost'. Total: {size}")
        if "underage_cost" in static_pass_features:
            size += 1
            log.debug(f"......... [5] Added 1 for static feature: 'underage_cost'. Total: {size}")
        if "lead_time" in static_pass_features:
            # For node features, we'll use a single value representing max incoming lead time.
            size += 1
            log.debug(f"......... [6] Added 1 for static feature: 'lead_time'. Total: {size}")

        # Demand history features
        if self.features_config.get("demand_history") and self.features_config.demand_history.past_periods > 0:
            past_periods = self.features_config.demand_history.past_periods
            size += past_periods
            log.debug(f"......... [7] Added {past_periods} for demand history. Total: {size}")
        
        log.debug(f"...... _calculate_node_feature_size RETURNING final total size: {size}")
        return size

    def _create_paper_gnn_policy(self) -> PaperGNNPolicy:
        """Builds the faithful implementation of the paper's GNN architecture."""
        log.debug("...... Building PaperGNNPolicy...")
        params = self.model_config.gnn_params
        
        node_feature_size = self._calculate_node_feature_size()
        output_size = 1 # Single flow value per edge

        model = PaperGNNPolicy(
            node_feature_size=node_feature_size,
            output_size=output_size,
            hidden_dim=params.module_width,
            num_message_passing_layers=params.message_passing_layers
        )
        return model
            