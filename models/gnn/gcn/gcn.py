import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer."""
    
    def __init__(self, in_features, out_features, use_bias=True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            use_bias: Whether to use bias
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Normalized adjacency matrix (sparse tensor) [num_nodes, num_nodes]
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Linear transformation: X @ W
        x = torch.matmul(x, self.weight)
        
        # Add bias
        if self.bias is not None:
            x = x + self.bias
        
        # Graph convolution: A @ X
        x = torch.sparse.mm(adj, x)
        
        return x
    
    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class GCN(nn.Module):
    """Graph Convolutional Network."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers=2, dropout=0.5, use_bias=True):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_bias: Whether to use bias in GCN layers
        """
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNLayer(input_dim, hidden_dim, use_bias))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, use_bias))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(GCNLayer(hidden_dim, output_dim, use_bias))
        else:
            # Single layer case
            self.layers[0] = GCNLayer(input_dim, output_dim, use_bias)
    
    def forward(self, x, adj):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            adj: Normalized adjacency matrix (sparse tensor) [num_nodes, num_nodes]
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            
            # Apply activation and dropout for all layers except the last
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def get_embeddings(self, x, adj):
        """Get node embeddings (alias for forward)."""
        return self.forward(x, adj)


class GCNEmbedding(nn.Module):
    """GCN with learnable node embeddings."""
    
    def __init__(self, num_nodes, embedding_dim, hidden_dim=None,
                 num_layers=2, dropout=0.5, use_bias=True):
        """
        Args:
            num_nodes: Number of nodes (items)
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension (default: same as embedding_dim)
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_bias: Whether to use bias
        """
        super(GCNEmbedding, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = embedding_dim
        
        # Learnable node embeddings
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # GCN layers
        self.gcn = GCN(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_bias=use_bias
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize node embeddings."""
        nn.init.xavier_uniform_(self.node_embedding.weight)
    
    def forward(self, adj, node_ids=None):
        """
        Forward pass.
        
        Args:
            adj: Normalized adjacency matrix (sparse tensor)
            node_ids: Specific node IDs to get embeddings for (optional)
                     If None, returns embeddings for all nodes
        
        Returns:
            Node embeddings
        """
        # Get initial embeddings
        x = self.node_embedding.weight  # [num_nodes, embedding_dim]
        
        # Apply GCN
        x = self.gcn(x, adj)
        
        # Return specific nodes if requested
        if node_ids is not None:
            x = x[node_ids]
        
        return x
    
    def get_all_embeddings(self, adj):
        """Get embeddings for all nodes."""
        return self.forward(adj, node_ids=None)
