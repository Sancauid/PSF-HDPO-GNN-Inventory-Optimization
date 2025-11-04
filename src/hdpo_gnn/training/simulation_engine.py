# src/hdpo_gnn/training/simulation_engine.py
import logging
from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from ..engine import functional as F
from ..engine import fel
from ..engine.pipeline import DifferentiablePipeline

log = logging.getLogger(__name__)


def run_simulation_episode(
    model: nn.Module,
    pyg_batch: Batch,
    env: 'Environment',
    config: DictConfig,
    is_training: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    log.debug("...... Calling run_simulation_episode...")
    
    data_config = config.data
    features_config = config.features

    device = pyg_batch.x.device
    num_samples = pyg_batch.num_graphs
    num_nodes = env.num_nodes
    
    is_transshipment_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if env.config.dynamics.is_transshipment:
        warehouse_ids = [n['id'] for n in env.nodes if n['type'] == 'warehouse']
        if warehouse_ids:
            is_transshipment_mask[warehouse_ids] = True
            log.debug(f"......... Transshipment mask applied to nodes: {warehouse_ids}")

    current_inventories = pyg_batch.initial_inventory.view(num_samples, num_nodes)
    holding_costs = pyg_batch.holding_costs.view(num_samples, num_nodes)
    underage_costs = pyg_batch.underage_costs.view(num_samples, num_nodes)
    
    episode_length = data_config.episode_length.train if is_training else data_config.episode_length.dev
    demands = pyg_batch.demands.view(num_samples, episode_length, num_nodes)
    
    edge_index_unbatched = env.edge_index.to(device)
    static_node_features_env = env.static_node_features.to(device)
    procurement_costs_env = env.procurement_costs.to(device)

    unique_lts = {int(p['lead_time']) for p in env.edge_params.values() if 'lead_time' in p}
    max_lead_time = max(unique_lts) if unique_lts else 0
    log.debug(f"......... Initializing DifferentiablePipeline with unique lead times: {unique_lts}, max: {max_lead_time}")
    pipeline = DifferentiablePipeline(num_samples, num_nodes, list(unique_lts))
    pipeline.to(device)
    
    edge_lead_times = torch.zeros(env.num_edges, dtype=torch.long, device=device)
    for i in range(env.num_edges):
        u, v = edge_index_unbatched[:, i].tolist()
        if 'lead_time' in env.edge_params[(u, v)]:
            edge_lead_times[i] = env.edge_params[(u, v)]['lead_time']

    step_costs = []
    static_node_features_unbatched = pyg_batch.x.view(num_samples, num_nodes, -1)
    
    for t in range(episode_length):
        demand_t = demands[:, t, :]
        
        feature_list_unbatched = [static_node_features_unbatched]
        
        if "inventory_on_hand" in features_config.dynamic:
            feature_list_unbatched.append(current_inventories.unsqueeze(-1))
        if "outstanding_orders" in features_config.dynamic:
            outstanding_orders = torch.zeros(num_samples, num_nodes, max_lead_time, device=device)
            for buffer in pipeline.buffers():
                buffer_lt = buffer.shape[-1]
                if buffer_lt > 0:
                    outstanding_orders[:, :, :buffer_lt] += buffer
            feature_list_unbatched.append(outstanding_orders)
        
        static_pass_features: list = features_config.static
        if "holding_cost" in static_pass_features:
            feature_list_unbatched.append(holding_costs.unsqueeze(-1))
        if "underage_cost" in static_pass_features:
            feature_list_unbatched.append(underage_costs.unsqueeze(-1))
        if "lead_time" in static_pass_features:
            lead_time_feature = torch.zeros_like(current_inventories).unsqueeze(-1)
            feature_list_unbatched.append(lead_time_feature)
        
        dynamic_x_unbatched = torch.cat(feature_list_unbatched, dim=-1)
        
        if t == 0:
            log.debug(f"......... Constructed dynamic feature tensor with shape: {dynamic_x_unbatched.shape}")
            log.debug(f"......... Model expects input features of size: {model.embed_node[0].in_features}")
        
        assert dynamic_x_unbatched.shape[-1] == model.embed_node[0].in_features, \
            f"Feature size mismatch! Constructed features: {dynamic_x_unbatched.shape[-1]}, Model expects: {model.embed_node[0].in_features}"

        dynamic_x_batched = dynamic_x_unbatched.view(num_samples * num_nodes, -1)

        actions = model(dynamic_x_batched, pyg_batch.edge_index)
        raw_edge_logits = actions['flows']
        
        feasible_edge_flows = fel.apply_fel(raw_edge_logits, current_inventories, env, config)
        feasible_edge_flows = feasible_edge_flows.view(num_samples, -1)

        arrivals = pipeline.get_arrivals()
        
        next_inventories, step_cost = F.transition_step(
            current_inventories=current_inventories + arrivals,
            edge_flows=feasible_edge_flows,
            node_features=static_node_features_env,
            edge_index=edge_index_unbatched,
            demand_t=demand_t,
            holding_costs=holding_costs,
            underage_costs=underage_costs,
            procurement_costs=procurement_costs_env,
            is_transshipment_mask=is_transshipment_mask,
            unmet_demand_assumption=env.unmet_demand_assumption
        )

        pipeline.advance_step()
        
        orders_to_nodes = torch.zeros_like(current_inventories)
        dest_nodes = edge_index_unbatched[1]
        orders_to_nodes.scatter_add_(1, dest_nodes.unsqueeze(0).expand(num_samples, -1), feasible_edge_flows)

        incoming_lts = torch.zeros(num_nodes, dtype=torch.long, device=device)
        for i in range(env.num_edges):
            dest_node = dest_nodes[i]
            incoming_lts[dest_node] = edge_lead_times[i]
        
        pipeline.place_orders(orders_to_nodes, incoming_lts)
        
        current_inventories = next_inventories
        step_costs.append(step_cost)
        
    warmup = data_config.warmup_periods.train if is_training else data_config.warmup_periods.dev
    total_episode_cost = torch.stack(step_costs, dim=0).sum(dim=0)
    reported_costs = torch.stack(step_costs[warmup:], dim=0).sum(dim=0)

    log.debug("...... run_simulation_episode RETURNING episode costs and reported costs.")
    return total_episode_cost, reported_costs