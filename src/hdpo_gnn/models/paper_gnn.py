from typing import Dict

import torch
from torch import Tensor, nn
from torch_scatter import scatter

class PaperGNNPolicy(nn.Module):
    """
    Paper GNN Policy implementing Algorithm 1 from the paper.
    
    This is a manual message-passing architecture that replicates the exact
    architecture described in the paper, using modular MLPs for each component.
    The architecture uses correlated edge and node updates with skip connections.
    """
    
    def __init__(
        self,
        node_feature_size: int,
        output_size: int = 1,
        hidden_dim: int = 32,
        num_message_passing_layers: int = 2,
    ) -> None:
        """
        Initialize the PaperGNNPolicy.
        
        Args:
            node_feature_size: Number of input features per node (static + dynamic).
            output_size: Size of per-edge outputs (usually 1 for flow prediction).
            hidden_dim: Hidden dimension for all MLPs (default 32 as per Table 6).
            num_message_passing_layers: Number of message passing iterations L_MP (default 2).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_message_passing_layers = num_message_passing_layers
        self.output_size = int(output_size)
        
        # Initial node embedding: maps node features to node embeddings
        self.embed_node = nn.Sequential(
            nn.Linear(node_feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Initial edge embedding: maps edge features (concatenated node embeddings) to edge embeddings
        # Input is 2 * hidden_dim (source + destination node embeddings)
        self.embed_edge = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Node update MLP: used in message passing to update node embeddings
        # Input: incoming messages (hidden_dim) + outgoing messages (hidden_dim) + current node embedding (hidden_dim)
        self.update_node = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Edge update MLP: used in message passing to update edge embeddings
        # Input: source node embedding (hidden_dim) + dest node embedding (hidden_dim) + current edge embedding (hidden_dim)
        self.update_edge = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Readout MLP: maps final edge embeddings to action logits
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
        )
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass implementing Algorithm 1.
        
        Args:
            x: Node features of shape [N_total, F] where N_total = B*N (batched nodes).
            edge_index: Graph connectivity in COO format of shape [2, E_total] where E_total = B*E.
        
        Returns:
            A dictionary with:
                - 'flows': tensor of edge flow predictions [E_total, output_size]
        """
        # Algorithm 1, Lines 3-4: Initialize node and edge embeddings
        # Line 3: Create initial node embeddings
        node_embeddings = self.embed_node(x)  # [N_total, hidden_dim]
        
        # Line 4: Create initial edge features and embeddings
        source_nodes = edge_index[0]  # [E_total]
        dest_nodes = edge_index[1]    # [E_total]
        
        # Concatenate source and destination node embeddings for each edge
        edge_features = torch.cat([
            node_embeddings[source_nodes],  # [E_total, hidden_dim]
            node_embeddings[dest_nodes],   # [E_total, hidden_dim]
        ], dim=1)  # [E_total, 2 * hidden_dim]
        
        # Create initial edge embeddings
        edge_embeddings = self.embed_edge(edge_features)  # [E_total, hidden_dim]
        
        # Algorithm 1, Lines 5-14: Message passing loop
        for _ in range(self.num_message_passing_layers):
            # Aggregate messages from incoming and outgoing edges for each node
            # Incoming edges: edges where this node is the destination
            # Outgoing edges: edges where this node is the source
            
            # Aggregate incoming messages (from edges where node is destination)
            incoming_messages = scatter(
                edge_embeddings,  # [E_total, hidden_dim]
                dest_nodes,       # [E_total] - indices for destination nodes
                dim=0,            # aggregate along node dimension
                dim_size=node_embeddings.shape[0],  # total number of nodes
                reduce='sum',     # sum aggregation
            )  # [N_total, hidden_dim]
            
            # Aggregate outgoing messages (from edges where node is source)
            outgoing_messages = scatter(
                edge_embeddings,  # [E_total, hidden_dim]
                source_nodes,     # [E_total] - indices for source nodes
                dim=0,            # aggregate along node dimension
                dim_size=node_embeddings.shape[0],  # total number of nodes
                reduce='sum',     # sum aggregation
            )  # [N_total, hidden_dim]
            
            # Combine incoming and outgoing messages with current node embedding
            # Concatenate to preserve information from both directions
            node_update_input = torch.cat([
                incoming_messages,  # [N_total, hidden_dim]
                outgoing_messages,  # [N_total, hidden_dim]
                node_embeddings,    # [N_total, hidden_dim]
            ], dim=1)  # [N_total, 3 * hidden_dim]
            
            # Update node embeddings with skip connection
            node_update = self.update_node(node_update_input)  # [N_total, hidden_dim]
            node_embeddings = node_embeddings + node_update  # Skip connection: new_h = old_h + update
            
            # Update edge embeddings
            # Input: source node embedding + dest node embedding + current edge embedding
            edge_update_input = torch.cat([
                node_embeddings[source_nodes],  # [E_total, hidden_dim]
                node_embeddings[dest_nodes],   # [E_total, hidden_dim]
                edge_embeddings,                # [E_total, hidden_dim]
            ], dim=1)  # [E_total, 3 * hidden_dim]
            
            edge_update = self.update_edge(edge_update_input)  # [E_total, hidden_dim]
            edge_embeddings = edge_embeddings + edge_update  # Skip connection: new_e = old_e + update
        
        # Algorithm 1, Lines 15-17: Readout
        # Pass final edge embeddings through readout MLP to get action logits
        flows = self.readout(edge_embeddings)  # [E_total, output_size]
        
        return {"flows": flows}

