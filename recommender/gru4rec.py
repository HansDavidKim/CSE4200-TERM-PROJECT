import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=None, num_layers=1, dropout=0.0, pretrained_embeddings=None, freeze_embeddings=False):
        """
        GRU4Rec model for next item prediction.
        
        Args:
            input_size (int): Number of items (vocabulary size).
            hidden_size (int): Dimension of hidden state.
            output_size (int): Number of items (vocabulary size) - usually same as input_size.
            embedding_dim (int, optional): Dimension of item embeddings. Required if pretrained_embeddings is None.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout probability.
            pretrained_embeddings (torch.Tensor, optional): Pre-trained item embeddings.
            freeze_embeddings (bool): Whether to freeze pre-trained embeddings.
        """
        super(GRU4Rec, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Item Embedding Layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
            embedding_dim = pretrained_embeddings.shape[1]
        else:
            if embedding_dim is None:
                embedding_dim = hidden_size
            self.embedding = nn.Embedding(input_size, embedding_dim)
            
        self.dropout = nn.Dropout(dropout)
            
        # GRU Layer
        # Input to GRU is (batch, seq_len, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x (torch.Tensor): Input sequence of item IDs. Shape: (batch, seq_len)
            hidden (torch.Tensor, optional): Initial hidden state.
            
        Returns:
            logits (torch.Tensor): Output logits. Shape: (batch, seq_len, output_size)
            hidden (torch.Tensor): Final hidden state.
        """
        # Embed items
        # Shape: (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # GRU forward
        # out shape: (batch, seq_len, hidden_size)
        out, hidden = self.gru(embedded, hidden)
        
        # Linear projection to item space
        # logits shape: (batch, seq_len, output_size)
        logits = self.fc(out)
        
        return logits, hidden
