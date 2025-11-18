import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):

        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        
        # Attention mechanism
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, x, adj):
        # Apply dropout to input features
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Linear transformation
        Wx = torch.mm(x, self.W)  # [num_nodes, out_features]
        
        # Compute attention coefficients
        attention = self._compute_attention(Wx, adj)
        
        # Aggregate neighbor features (attention is already normalized)
        h_prime = torch.sparse.mm(attention, Wx)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _compute_attention(self, Wx, adj):
        num_nodes = Wx.size(0)
        
        # Get edge indices from sparse adjacency matrix
        edge_index = adj._indices()
        num_edges = edge_index.size(1)
        
        # Compute attention logits for each edge
        # e_ij = LeakyReLU(a^T [Wx_i || Wx_j])
        Wx_i = Wx[edge_index[0]]  # Source nodes [num_edges, out_features]
        Wx_j = Wx[edge_index[1]]  # Target nodes [num_edges, out_features]
        
        # Concatenate and compute attention
        edge_features = torch.cat([Wx_i, Wx_j], dim=1)  # [num_edges, 2*out_features]
        e = self.leakyrelu(torch.matmul(edge_features, self.a).squeeze())  # [num_edges]
        
        # Apply attention dropout to edge weights (not the sparse structure)
        if self.training and self.dropout > 0:
            # Create dropout mask for edges
            dropout_mask = torch.bernoulli(torch.ones_like(e) * (1 - self.dropout))
            e = e * dropout_mask / (1 - self.dropout)  # Scale to maintain expected value
        
        # Create sparse attention matrix - UPDATED LINE
        attention_sparse = torch.sparse_coo_tensor(
            edge_index,
            e,
            size=(num_nodes, num_nodes),
            dtype=torch.float32
        )
        
        # Apply softmax (per node)
        attention_sparse = self._sparse_softmax(attention_sparse)
        
        return attention_sparse

    
    def _sparse_softmax(self, sparse_tensor):
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()
        shape = sparse_tensor.size()
        
        # Get row indices
        row_indices = indices[0]
        
        # Compute max per row for numerical stability
        max_values = torch.full((shape[0],), float('-inf'), device=values.device)
        max_values.scatter_reduce_(0, row_indices, values, reduce='amax', include_self=False)
        max_values = torch.where(torch.isinf(max_values), torch.zeros_like(max_values), max_values)
        
        # Subtract max and compute exp
        exp_values = (values - max_values[row_indices]).exp()
        
        # Compute sum per row
        sum_per_row = torch.zeros(shape[0], device=values.device)
        sum_per_row.scatter_add_(0, row_indices, exp_values)
        
        # Avoid division by zero
        sum_per_row = torch.where(sum_per_row == 0, torch.ones_like(sum_per_row), sum_per_row)
        
        # Normalize
        normalized_values = exp_values / sum_per_row[row_indices]
        
        # UPDATED LINE
        return torch.sparse_coo_tensor(indices, normalized_values, size=shape, dtype=torch.float32)

    
    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class MultiHeadGATLayer(nn.Module):
    
    def __init__(self, in_features, out_features, num_heads=8, 
                 dropout=0.6, alpha=0.2, concat=True):

        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        # Create multiple attention heads
        self.attentions = nn.ModuleList([
            GATLayer(in_features, out_features, dropout, alpha, concat=True)
            for _ in range(num_heads)
        ])
    
    def forward(self, x, adj):

        # Apply each attention head
        head_outputs = [att(x, adj) for att in self.attentions]
        
        if self.concat:
            # Concatenate outputs
            return torch.cat(head_outputs, dim=1)
        else:
            # Average outputs
            return torch.mean(torch.stack(head_outputs), dim=0)


class GAT(nn.Module):
    """Graph Attention Network."""
    
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_heads=8, num_layers=2, dropout=0.6, alpha=0.2):

        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        
        # First layer (multi-head with concatenation)
        self.layers.append(
            MultiHeadGATLayer(input_dim, hidden_dim, num_heads, dropout, alpha, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, 
                                num_heads, dropout, alpha, concat=True)
            )
        
        # Output layer (averaging instead of concatenation)
        if num_layers > 1:
            self.layers.append(
                MultiHeadGATLayer(hidden_dim * num_heads, output_dim,
                                1, dropout, alpha, concat=False)
            )
        else:
            # Single layer case
            self.layers[0] = MultiHeadGATLayer(input_dim, output_dim,
                                              num_heads, dropout, alpha, concat=False)
    
    def forward(self, x, adj):

        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            
            # Apply dropout between layers (not within GAT layers)
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def get_embeddings(self, x, adj):
        return self.forward(x, adj)


class GATEmbedding(nn.Module):
    
    def __init__(self, num_nodes, embedding_dim, hidden_dim=None,
                 num_heads=8, num_layers=2, dropout=0.6, alpha=0.2):

        super(GATEmbedding, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = max(embedding_dim // num_heads, 1)
        
        # Learnable node embeddings
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # GAT layers
        self.gat = GAT(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            alpha=alpha
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
    
    def forward(self, adj, node_ids=None):

        # Get initial embeddings
        x = self.node_embedding.weight
        
        # Apply GAT
        x = self.gat(x, adj)
        
        # Return specific nodes if requested
        if node_ids is not None:
            x = x[node_ids]
        
        return x
    
    def get_all_embeddings(self, adj):
        return self.forward(adj, node_ids=None)
