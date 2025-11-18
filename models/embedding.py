import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn.gcn.gcn import GCNEmbedding
from gnn.gcn.gat import GATEmbedding


class GRU4REC(nn.Module):
    """Baseline GRU4REC without GNN."""
    
    def __init__(self, num_items, embedding_dim, hidden_size, 
                 num_layers=1, dropout=0.2):
        """
        Args:
            num_items: Number of items
            embedding_dim: Item embedding dimension
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super(GRU4REC, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Item embeddings
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        
        # GRU
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_items)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, item_sequence, lengths=None):
        """
        Forward pass.
        
        Args:
            item_sequence: [batch_size, seq_len] - Item IDs
            lengths: [batch_size] - Actual sequence lengths (optional)
        
        Returns:
            logits: [batch_size, num_items] - Next item prediction scores
        """
        batch_size, seq_len = item_sequence.shape
        
        # Get embeddings
        item_emb = self.item_embedding(item_sequence)  # [B, L, D]
        
        # Pack sequence if lengths provided
        if lengths is not None:
            item_emb = nn.utils.rnn.pack_padded_sequence(
                item_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # GRU
        gru_out, hidden = self.gru(item_emb)
        
        # Unpack if packed
        if lengths is not None:
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=seq_len
            )
        
        # Use last hidden state
        # hidden: [num_layers, batch_size, hidden_size]
        last_hidden = hidden[-1]  # [batch_size, hidden_size]
        
        # Predict next item
        logits = self.fc(last_hidden)  # [batch_size, num_items]
        
        return logits
    
    def predict_top_k(self, item_sequence, k=10, lengths=None):
        """
        Predict top-k items.
        
        Args:
            item_sequence: [batch_size, seq_len]
            k: Number of items to recommend
            lengths: Sequence lengths (optional)
        
        Returns:
            top_k_items: [batch_size, k] - Top-k item IDs
            top_k_scores: [batch_size, k] - Scores
        """
        logits = self.forward(item_sequence, lengths)
        top_k_scores, top_k_items = torch.topk(logits, k, dim=1)
        return top_k_items, top_k_scores


class GNN_GRU4REC(nn.Module):
    """GRU4REC with GNN-enhanced item embeddings."""
    
    def __init__(self, num_items, embedding_dim, hidden_size,
                 gnn_type='gcn', gnn_layers=2, gnn_hidden_dim=None,
                 num_heads=4, gru_layers=1, dropout=0.2, gnn_dropout=0.5):
        """
        Args:
            num_items: Number of items
            embedding_dim: Item embedding dimension
            hidden_size: GRU hidden size
            gnn_type: 'gcn' or 'gat'
            gnn_layers: Number of GNN layers
            gnn_hidden_dim: GNN hidden dimension (default: same as embedding_dim)
            num_heads: Number of attention heads (for GAT)
            gru_layers: Number of GRU layers
            dropout: GRU dropout rate
            gnn_dropout: GNN dropout rate
        """
        super(GNN_GRU4REC, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.gnn_type = gnn_type
        
        # GNN for item embeddings
        if gnn_type == 'gcn':
            self.gnn = GCNEmbedding(
                num_nodes=num_items,
                embedding_dim=embedding_dim,
                hidden_dim=gnn_hidden_dim or embedding_dim,
                num_layers=gnn_layers,
                dropout=gnn_dropout
            )
        elif gnn_type == 'gat':
            self.gnn = GATEmbedding(
                num_nodes=num_items,
                embedding_dim=embedding_dim,
                hidden_dim=gnn_hidden_dim or (embedding_dim // num_heads),
                num_heads=num_heads,
                num_layers=gnn_layers,
                dropout=gnn_dropout
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # GRU
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_items)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, item_sequence, adj, lengths=None):
        """
        Forward pass.
        
        Args:
            item_sequence: [batch_size, seq_len] - Item IDs
            adj: Adjacency matrix (sparse tensor) for GNN
            lengths: [batch_size] - Actual sequence lengths (optional)
        
        Returns:
            logits: [batch_size, num_items] - Next item prediction scores
        """
        batch_size, seq_len = item_sequence.shape
        
        # Get GNN-enhanced item embeddings
        item_embeddings = self.gnn(adj)  # [num_items, embedding_dim]
        
        # Get sequence embeddings
        seq_embeddings = item_embeddings[item_sequence]  # [B, L, D]
        
        # Pack sequence if lengths provided
        if lengths is not None:
            seq_embeddings = nn.utils.rnn.pack_padded_sequence(
                seq_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # GRU
        gru_out, hidden = self.gru(seq_embeddings)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_size]
        
        # Predict next item
        logits = self.fc(last_hidden)  # [batch_size, num_items]
        
        return logits
    
    def predict_top_k(self, item_sequence, adj, k=10, lengths=None):
        """
        Predict top-k items.
        
        Args:
            item_sequence: [batch_size, seq_len]
            adj: Adjacency matrix
            k: Number of items to recommend
            lengths: Sequence lengths (optional)
        
        Returns:
            top_k_items: [batch_size, k] - Top-k item IDs
            top_k_scores: [batch_size, k] - Scores
        """
        logits = self.forward(item_sequence, adj, lengths)
        top_k_scores, top_k_items = torch.topk(logits, k, dim=1)
        return top_k_items, top_k_scores
    
    def get_item_embeddings(self, adj):
        """
        Get GNN-enhanced item embeddings.
        
        Args:
            adj: Adjacency matrix
        
        Returns:
            item_embeddings: [num_items, embedding_dim]
        """
        return self.gnn(adj)


class HybridGNN_GRU4REC(nn.Module):
    """Hybrid model combining multiple GNN types."""
    
    def __init__(self, num_items, embedding_dim, hidden_size,
                 gnn_types=['gcn', 'gat'], fusion='concat',
                 gnn_layers=2, num_heads=4, gru_layers=1, 
                 dropout=0.2, gnn_dropout=0.5):
        """
        Args:
            num_items: Number of items
            embedding_dim: Item embedding dimension
            hidden_size: GRU hidden size
            gnn_types: List of GNN types to use
            fusion: How to combine GNN outputs ('concat', 'mean', 'attention')
            gnn_layers: Number of GNN layers
            num_heads: Number of attention heads (for GAT)
            gru_layers: Number of GRU layers
            dropout: GRU dropout rate
            gnn_dropout: GNN dropout rate
        """
        super(HybridGNN_GRU4REC, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.gnn_types = gnn_types
        self.fusion = fusion
        
        # Multiple GNNs
        self.gnns = nn.ModuleDict()
        for gnn_type in gnn_types:
            if gnn_type == 'gcn':
                self.gnns[gnn_type] = GCNEmbedding(
                    num_nodes=num_items,
                    embedding_dim=embedding_dim,
                    num_layers=gnn_layers,
                    dropout=gnn_dropout
                )
            elif gnn_type == 'gat':
                self.gnns[gnn_type] = GATEmbedding(
                    num_nodes=num_items,
                    embedding_dim=embedding_dim,
                    hidden_dim=embedding_dim // num_heads,
                    num_heads=num_heads,
                    num_layers=gnn_layers,
                    dropout=gnn_dropout
                )
        
        # Fusion layer
        if fusion == 'concat':
            gru_input_dim = embedding_dim * len(gnn_types)
        elif fusion == 'attention':
            self.attention_weights = nn.Parameter(torch.ones(len(gnn_types)))
            gru_input_dim = embedding_dim
        else:  # mean
            gru_input_dim = embedding_dim
        
        # GRU
        self.gru = nn.GRU(
            gru_input_dim,
            hidden_size,
            gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_items)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def fuse_embeddings(self, embeddings_list):
        """
        Fuse multiple GNN embeddings.
        
        Args:
            embeddings_list: List of [num_items, embedding_dim] tensors
        
        Returns:
            fused_embeddings: [num_items, fused_dim]
        """
        if self.fusion == 'concat':
            return torch.cat(embeddings_list, dim=-1)
        elif self.fusion == 'mean':
            return torch.mean(torch.stack(embeddings_list), dim=0)
        elif self.fusion == 'attention':
            # Weighted sum with learnable weights
            weights = F.softmax(self.attention_weights, dim=0)
            weighted_embs = [w * emb for w, emb in zip(weights, embeddings_list)]
            return sum(weighted_embs)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")
    
    def forward(self, item_sequence, adj, lengths=None):
        """
        Forward pass.
        
        Args:
            item_sequence: [batch_size, seq_len]
            adj: Adjacency matrix (or dict of adjacency matrices)
            lengths: Sequence lengths (optional)
        
        Returns:
            logits: [batch_size, num_items]
        """
        batch_size, seq_len = item_sequence.shape
        
        # Get embeddings from each GNN
        embeddings_list = []
        for gnn_type, gnn in self.gnns.items():
            emb = gnn(adj)  # [num_items, embedding_dim]
            embeddings_list.append(emb)
        
        # Fuse embeddings
        fused_embeddings = self.fuse_embeddings(embeddings_list)
        
        # Get sequence embeddings
        seq_embeddings = fused_embeddings[item_sequence]  # [B, L, fused_dim]
        
        # Pack sequence if lengths provided
        if lengths is not None:
            seq_embeddings = nn.utils.rnn.pack_padded_sequence(
                seq_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # GRU
        gru_out, hidden = self.gru(seq_embeddings)
        
        # Use last hidden state
        last_hidden = hidden[-1]
        
        # Predict next item
        logits = self.fc(last_hidden)
        
        return logits
    
    def predict_top_k(self, item_sequence, adj, k=10, lengths=None):
        """Predict top-k items."""
        logits = self.forward(item_sequence, adj, lengths)
        top_k_scores, top_k_items = torch.topk(logits, k, dim=1)
        return top_k_items, top_k_scores

if __name__ == "__main__":
    print("=" * 50)
    print("Testing GNN-GRU4REC Models")
    print("=" * 50)
    
    import numpy as np
    import scipy.sparse as sp
    from gnn.gcn.graph_utils import GraphBuilder
    
    # 간단한 테스트 데이터
    num_users = 100
    num_items = 50
    user_item_matrix = sp.random(num_users, num_items, density=0.05, format='csr')
    
    # 그래프 구축
    print("\n[1] Building graph...")
    graph_builder = GraphBuilder(user_item_matrix)
    item_adj = graph_builder.build_item_graph(threshold=1, normalize=True, add_self_loop=True)
    print("✓ Graph built successfully")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[2] Using device: {device}")
    item_adj = item_adj.to(device)
    
    # 테스트 데이터
    batch_size = 8
    seq_len = 5
    item_sequence = torch.randint(1, num_items, (batch_size, seq_len), device=device)
    lengths = torch.full((batch_size,), seq_len, device=device)
    
    print(f"\n[3] Test input shape: {item_sequence.shape}")
    
    # 1. Baseline GRU4REC
    print("\n" + "=" * 50)
    print("Testing Baseline GRU4REC")
    print("=" * 50)
    try:
        model = GRU4REC(
            num_items=num_items,
            embedding_dim=64,
            hidden_size=128,
            num_layers=1
        ).to(device)
        
        logits = model(item_sequence, lengths)
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 2. GCN + GRU4REC
    print("\n" + "=" * 50)
    print("Testing GCN + GRU4REC")
    print("=" * 50)
    try:
        model = GNN_GRU4REC(
            num_items=num_items,
            embedding_dim=64,
            hidden_size=128,
            gnn_type='gcn',
            gnn_layers=2
        ).to(device)
        
        logits = model(item_sequence, item_adj, lengths)
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 3. GAT + GRU4REC
    print("\n" + "=" * 50)
    print("Testing GAT + GRU4REC")
    print("=" * 50)
    try:
        model = GNN_GRU4REC(
            num_items=num_items,
            embedding_dim=64,
            hidden_size=128,
            gnn_type='gat',
            num_heads=4
        ).to(device)
        
        logits = model(item_sequence, item_adj, lengths)
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 4. Hybrid Model
    print("\n" + "=" * 50)
    print("Testing Hybrid GNN + GRU4REC")
    print("=" * 50)
    try:
        model = HybridGNN_GRU4REC(
            num_items=num_items,
            embedding_dim=64,
            hidden_size=128,
            gnn_types=['gcn', 'gat'],
            fusion='attention'
        ).to(device)
        
        logits = model(item_sequence, item_adj, lengths)
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
