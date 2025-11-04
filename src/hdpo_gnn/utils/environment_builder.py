# src/hdpo_gnn/utils/environment_builder.py
import logging
from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig

# Get the logger for this module
log = logging.getLogger(__name__)

class Environment:
    """
    Builds and holds the complete definition of an inventory network problem.
    """
    def __init__(self, config: DictConfig):
        log.info("--- Initializing Environment ---")
        log.debug(f"RECEIVED config of type {type(config)}")
        
        self.config = config
        
        # Initialize attributes
        self.nodes: List[Dict] = []
        self.node_type_map: Dict[str, int] = {}
        self.node_type_counts: Dict[str, int] = {}
        self.num_nodes: int = 0
        self.edge_index: torch.Tensor = None
        self.num_edges: int = 0
        self.static_node_features: torch.Tensor = None
        self.holding_costs: torch.Tensor = None
        self.underage_costs: torch.Tensor = None
        self.procurement_costs: torch.Tensor = None
        self.edge_params: Dict[Tuple[int, int], Dict] = {} # Maps (u, v) -> {params}
        self.warehouse_ids: List[int] = []  # List of warehouse node IDs for FEL
        
        # Build the environment step-by-step
        self._build_topology()
        self._initialize_parameters()
        self._create_static_node_features()
        self._set_dynamics()
        
        # Set warehouse_ids after nodes are built
        self.warehouse_ids = [n['id'] for n in self.nodes if n['type'] == 'warehouse']

        log.info(f"--- Environment Initialized Successfully. Final state: "
                 f"{self.num_nodes} nodes, {self.num_edges} edges. ---")

    def _build_topology(self):
        """Generates nodes and edges based on topology rules in the config."""
        log.debug("--> Calling _build_topology...")
        
        rules = self.config.topology.rules
        node_counts = rules.node_counts
        log.debug(f"RECEIVED rules: {rules}")
        
        # 1. Generate Nodes
        current_id = 0
        self.nodes.append({'id': current_id, 'type': 'supplier'})
        current_id += 1
        
        node_type_idx = 0
        for node_type, count in node_counts.items():
            self.node_type_counts[node_type] = count
            if node_type not in self.node_type_map:
                self.node_type_map[node_type] = node_type_idx
                node_type_idx += 1
            for _ in range(count):
                self.nodes.append({'id': current_id, 'type': node_type})
                current_id += 1
        
        self.num_nodes = len(self.nodes)
        
        # 2. Generate Edges
        edge_policy = rules.edge_policy
        edge_list = self._generate_edges_from_policy(edge_policy)
        
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.num_edges = self.edge_index.shape[1]
        
        log.debug(f"<-- _build_topology COMPLETED. PRODUCED attributes: "
                  f"num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
                  f"edge_index shape={self.edge_index.shape}")

    def _generate_edges_from_policy(self, edge_policy: str) -> List[List[int]]:
        """Helper function to generate edge list based on a policy string."""
        log.debug(f"...... Calling _generate_edges_from_policy. RECEIVED edge_policy='{edge_policy}'")
        
        edge_list = []
        node_ids_by_type = {ntype: [] for ntype in self.node_type_counts.keys()}
        for node in self.nodes:
            if node['type'] != 'supplier':
                node_ids_by_type[node['type']].append(node['id'])

        if edge_policy == "supplier_to_all_stores":
            for store_id in node_ids_by_type['store']:
                edge_list.append([0, store_id])
        
        elif edge_policy == "warehouse_to_all_stores":
            for wh_id in node_ids_by_type['warehouse']:
                for store_id in node_ids_by_type['store']:
                    edge_list.append([wh_id, store_id])
            for wh_id in node_ids_by_type['warehouse']:
                edge_list.append([0, wh_id])
        
        elif edge_policy == "mwms_spatial_clustering":
            log.warning("Edge policy 'mwms_spatial_clustering' is a TODO. Using fallback.")
            for wh_id in node_ids_by_type['warehouse']:
                for store_id in node_ids_by_type['store']:
                    edge_list.append([wh_id, store_id])
            for wh_id in node_ids_by_type['warehouse']:
                edge_list.append([0, wh_id])
        
        else:
            raise NotImplementedError(f"Edge policy '{edge_policy}' is not recognized.")

        log.debug(f"...... _generate_edges_from_policy RETURNING edge_list of length {len(edge_list)}")
        return edge_list
        

    def _initialize_parameters(self):
        """Samples or sets static parameters like costs and lead times."""
        log.debug("--> Calling _initialize_parameters...")
        params_cfg = self.config.parameters
        log.debug(f"RECEIVED params_cfg: {params_cfg}")

        # Initialize tensors to hold per-node values
        self.holding_costs = torch.zeros(self.num_nodes)
        self.underage_costs = torch.zeros(self.num_nodes)
        self.edge_params = {}

        # --- Assign Node Parameters ---
        log.debug("...... Assigning node parameters...")
        for node in self.nodes:
            node_id = node['id']
            node_type = node['type']
            
            if node_type == 'store':
                p_cfg = params_cfg.nodes.stores
                # Sample a single value for this store
                h_cost = self._sample_param(p_cfg.holding_cost, 1)
                p_cost = self._sample_param(p_cfg.underage_cost, 1)
                if h_cost is not None: self.holding_costs[node_id] = h_cost
                if p_cost is not None: self.underage_costs[node_id] = p_cost

            elif node_type == 'warehouse':
                p_cfg = params_cfg.nodes.warehouses
                h_cost = self._sample_param(p_cfg.holding_cost, 1)
                if h_cost is not None: self.holding_costs[node_id] = h_cost
                # Warehouses have no underage cost, so it remains 0.

        log.info("...... Assigned node costs. Verifying...")
        log.info(f"......... Sum of holding costs: {torch.sum(self.holding_costs).item():.2f}")
        log.info(f"......... Sum of underage costs: {torch.sum(self.underage_costs).item():.2f}")

        # --- Assign Edge Parameters ---
        # ... (the rest of the function can remain the same)
        log.debug("...... Assigning edge parameters...")
        node_id_to_type = {n['id']: n['type'] for n in self.nodes}
        
        for i in range(self.num_edges):
            u, v = self.edge_index[:, i].tolist()
            u_type, v_type = node_id_to_type[u], node_id_to_type[v]
            edge_key = (u,v)
            self.edge_params[edge_key] = {}
            
            # This logic is a bit complex, but let's assume it's correct for now
            if u_type == 'supplier' and v_type == 'warehouse':
                cfg = params_cfg.edges.supplier_to_warehouse
                wh_ids = [n['id'] for n in self.nodes if n['type'] == 'warehouse']
                wh_idx = wh_ids.index(v)
                # Need to handle if value is a list or not
                lt_val = cfg.lead_time.get('values', cfg.lead_time.value)
                pc_val = cfg.procurement_cost.get('values', cfg.procurement_cost.value)
                self.edge_params[edge_key]['lead_time'] = lt_val[wh_idx] if isinstance(lt_val, (list, tuple)) else lt_val
                self.edge_params[edge_key]['procurement_cost'] = pc_val[wh_idx] if isinstance(pc_val, (list, tuple)) else pc_val

            elif u_type == 'supplier' and v_type == 'store':
                cfg = params_cfg.edges.supplier_to_store
                self.edge_params[edge_key]['lead_time'] = cfg.lead_time.value

            elif u_type == 'warehouse' and v_type == 'store':
                cfg = params_cfg.edges.warehouse_to_store
                self.edge_params[edge_key]['lead_time'] = cfg.lead_time.value
        
        log.debug("...... Creating procurement cost tensor...")
        self.procurement_costs = torch.zeros(self.num_edges)
        for i in range(self.num_edges):
            u, v = self.edge_index[:, i].tolist()
            if u == 0:
                edge_key = (u, v)
                if 'procurement_cost' in self.edge_params[edge_key]:
                    self.procurement_costs[i] = self.edge_params[edge_key]['procurement_cost']
        log.debug("...... Procurement cost tensor created.")
        
        log.debug("...... Assigned edge parameters.")
        log.debug(f"<-- _initialize_parameters COMPLETED. PRODUCED attributes: self.holding_costs, self.underage_costs, self.edge_params")

    def _sample_param(self, p_cfg: DictConfig, size: int) -> torch.Tensor:
        """Helper to sample a parameter tensor based on its config."""
        log.debug(f"...... Calling _sample_param. RECEIVED p_cfg={p_cfg}, size={size}")
        method = p_cfg.sampling_method
        result = None
        if method == "fixed":
            result = torch.full((size,), float(p_cfg.value))
        elif method == "uniform":
            result = torch.rand(size) * (p_cfg.range[1] - p_cfg.range[0]) + p_cfg.range[0]
        elif method == "from_list":
            assert len(p_cfg.values) == size, f"List length mismatch for '{p_cfg}'"
            result = torch.tensor(p_cfg.values, dtype=torch.float32)
        elif method == "uniform_per_sample":
            log.debug("...... 'uniform_per_sample' will be handled by DatasetManager. RETURNING None.")
            return None
        else:
            raise NotImplementedError(f"Sampling method '{method}' not recognized.")
        
        log.debug(f"...... _sample_param RETURNING tensor of shape {result.shape if result is not None else 'None'}")
        return result

    def _create_static_node_features(self):
        """Creates the static node feature matrix for the GNN."""
        log.debug("--> Calling _create_static_node_features...")
        
        num_types = len(self.node_type_map) + 1 # +1 for supplier
        features = torch.zeros(self.num_nodes, num_types)
        
        for node in self.nodes:
            node_id, node_type = node['id'], node['type']
            if node_type == 'supplier':
                features[node_id, 0] = 1.0
            else:
                type_idx = self.node_type_map[node_type]
                features[node_id, type_idx + 1] = 1.0
        
        self.static_node_features = features
        log.debug(f"<-- _create_static_node_features COMPLETED. RETURNING tensor of shape {features.shape}")

    def _set_dynamics(self):
        """Sets high-level simulation dynamics from the config."""
        log.debug("--> Calling _set_dynamics...")
        self.unmet_demand_assumption = self.config.dynamics.unmet_demand_assumption
        self.objective = self.config.dynamics.objective
        log.debug(f"<-- _set_dynamics COMPLETED. PRODUCED attributes: "
                  f"unmet_demand_assumption='{self.unmet_demand_assumption}', objective='{self.objective}'")