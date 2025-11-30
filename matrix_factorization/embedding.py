import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    def __init__(self, file_path, k=10):
        """
        Initialize the embedding model.
        
        Args:
            file_path (str): Path to the data file.
            k (int): Number of latent factors (dimensions) for embeddings.
        """
        self.file_path = file_path
        self.k = k
        self.user_item_matrix = None
        self.user_id_map = None
        self.item_id_map = None
        self.user_embeddings = None
        self.item_embeddings = None

    def load_data(self):
        """
        Load data from CSV and create a user-item matrix.
        """
        print(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return

        # Determine value column
        if 'reward' in df.columns:
            value_col = 'reward'
        elif 'value' in df.columns:
            value_col = 'value'
        else:
            print("Error: No 'reward' or 'value' column found.")
            return

        # Create pivot table
        # Fill missing values with 0 (assuming 0 means no interaction/rating)
        # Handle duplicates by taking the mean of rewards
        print(f"Creating pivot table using '{value_col}'...")
        pivot_df = df.pivot_table(index='user_id', columns='item_id', values=value_col, aggfunc='mean').fillna(0)
        
        # Store mappings to convert between matrix indices and original IDs
        self.user_id_map = {idx: user_id for idx, user_id in enumerate(pivot_df.index)}
        self.item_id_map = {idx: item_id for idx, item_id in enumerate(pivot_df.columns)}
        
        self.user_item_matrix = pivot_df.values
        print(f"Matrix shape: {self.user_item_matrix.shape}")

    def get_matrix(self):
        """
        Return the numpy matrix.
        """
        if self.user_item_matrix is None:
            self.load_data()
        return self.user_item_matrix

    @abstractmethod
    def train(self):
        """
        Train the model to extract embeddings.
        """
        pass

    def get_embeddings(self):
        """
        Return user and item embeddings.
        """
        if self.user_embeddings is None or self.item_embeddings is None:
            self.train()
            
        return self.user_embeddings, self.item_embeddings

class SVDEmbedding(EmbeddingModel):
    def train(self):
        """
        Perform SVD to extract embeddings using scikit-learn.
        """
        if self.user_item_matrix is None:
            self.load_data()
            
        if self.user_item_matrix is None:
            return

        print(f"Performing SVD with k={self.k} using scikit-learn...")
        
        # Initialize TruncatedSVD
        svd = TruncatedSVD(n_components=self.k, random_state=42)
        
        # Fit and transform the matrix
        # user_embeddings corresponds to U * Sigma (in SVD terms)
        self.user_embeddings = svd.fit_transform(self.user_item_matrix)
        
        # item_embeddings corresponds to V^T (components_)
        # We transpose it to match the shape (n_items, k)
        self.item_embeddings = svd.components_.T
        
        print("SVD Embeddings extracted.")
        print(f"User Embeddings shape: {self.user_embeddings.shape}")
        print(f"Item Embeddings shape: {self.item_embeddings.shape}")

class PCAEmbedding(EmbeddingModel):
    def train(self):
        """
        Perform PCA to extract embeddings using scikit-learn.
        """
        if self.user_item_matrix is None:
            self.load_data()
            
        if self.user_item_matrix is None:
            return

        print(f"Performing PCA with k={self.k} using scikit-learn...")
        
        # Initialize PCA
        pca = PCA(n_components=self.k, random_state=42)
        
        # Fit and transform the matrix
        # user_embeddings corresponds to the transformed data (principal components scores)
        self.user_embeddings = pca.fit_transform(self.user_item_matrix)
        
        # item_embeddings corresponds to the components (eigenvectors)
        # We transpose it to match the shape (n_items, k)
        self.item_embeddings = pca.components_.T
        
        print("PCA Embeddings extracted.")
        print(f"User Embeddings shape: {self.user_embeddings.shape}")
        print(f"Item Embeddings shape: {self.item_embeddings.shape}")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

# Fix for OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class GraphEmbedding(EmbeddingModel):
    def __init__(self, file_path, k=10, learning_rate=0.01, epochs=50):
        super().__init__(file_path, k)
        self.lr = learning_rate
        self.epochs = epochs
        # Sparse operations are not fully supported on MPS yet, so we force CPU for MPS.
        # But for CUDA (Colab), we should use GPU.
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"GraphEmbedding using device: {self.device}")
        
        self.adj_matrix = None
        self.num_users = 0
        self.num_items = 0
        self.graph = None

    def load_data(self):
        """
        Load implicit data and build adjacency matrix.
        """
        print(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return

        # Assume implicit feedback (value=1)
        # Re-map IDs to ensure contiguous range [0, N-1] and [0, M-1]
        self.user_id_map = {id: i for i, id in enumerate(df['user_id'].unique())}
        self.item_id_map = {id: i for i, id in enumerate(df['item_id'].unique())}
        self.num_users = len(self.user_id_map)
        self.num_items = len(self.item_id_map)
        
        user_indices = [self.user_id_map[uid] for uid in df['user_id']]
        item_indices = [self.item_id_map[iid] for iid in df['item_id']]
        
        self.interactions = list(zip(user_indices, item_indices))
        
        print(f"Num Users: {self.num_users}, Num Items: {self.num_items}")
        
        # Build Adjacency Matrix
        self._build_adjacency_matrix(user_indices, item_indices)

    def _build_adjacency_matrix(self, user_indices, item_indices):
        """
        Build normalized adjacency matrix for LightGCN.
        A = [0, R; R^T, 0]
        """
        print("Building adjacency matrix...")
        n_nodes = self.num_users + self.num_items
        
        # Create R matrix (User-Item interaction)
        # Rows: Users, Cols: Items
        # We use (data, (row, col)) format
        R = sp.coo_matrix((np.ones(len(user_indices)), (user_indices, item_indices)), 
                          shape=(self.num_users, self.num_items))
        
        # Build A = [0, R; R^T, 0]
        # Top left: 0 (Users x Users)
        # Top right: R (Users x Items)
        # Bottom left: R^T (Items x Users)
        # Bottom right: 0 (Items x Items)
        
        # We can construct this directly using sparse hstack/vstack
        top = sp.hstack([sp.csr_matrix((self.num_users, self.num_users)), R])
        bottom = sp.hstack([R.T, sp.csr_matrix((self.num_items, self.num_items))])
        adj = sp.vstack([top, bottom])
        
        # Normalize A: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        # Convert to PyTorch sparse tensor
        coo = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape = torch.Size(coo.shape)
        
        self.graph = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
        print("Adjacency matrix built.")

    def _sample_batch(self, batch_size=1024):
        """
        Sample a batch of (user, pos_item, neg_item) for BPR loss.
        """
        users = np.random.randint(0, self.num_users, batch_size)
        pos_items = []
        neg_items = []
        
        # Create a set of interactions for fast lookup
        interaction_set = set(self.interactions)
        
        for u in users:
            # Find positive items for this user
            # This is slow if done naively every time. 
            # Optimization: Pre-group items by user.
            # For simplicity in this script, we'll assume we can find one.
            # Let's optimize by pre-processing in load_data if needed.
            # Here, we'll just pick a random interaction from the list and check if it matches user
            # Actually, better to sample from self.interactions directly
            pass
        
        # Better sampling strategy:
        indices = np.random.randint(0, len(self.interactions), batch_size)
        sampled_interactions = [self.interactions[i] for i in indices]
        
        users = [u for u, i in sampled_interactions]
        pos_items = [i for u, i in sampled_interactions]
        
        for u in users:
            while True:
                neg_i = np.random.randint(0, self.num_items)
                if (u, neg_i) not in interaction_set:
                    neg_items.append(neg_i)
                    break
                    
        return torch.tensor(users).to(self.device), \
               torch.tensor(pos_items).to(self.device), \
               torch.tensor(neg_items).to(self.device)

    @abstractmethod
    def train(self):
        pass

class LightGCNEmbedding(GraphEmbedding):
    def __init__(self, file_path, k=10, layers=3, **kwargs):
        super().__init__(file_path, k, **kwargs)
        self.layers = layers
        self.embedding = None # Learnable embeddings

    def train(self):
        if self.graph is None:
            self.load_data()
            
        print(f"Training LightGCN with k={self.k}, layers={self.layers}...")
        
        # Initialize embeddings
        # Shape: (Num_Users + Num_Items, k)
        self.embedding = nn.Embedding(self.num_users + self.num_items, self.k).to(self.device)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
        optimizer = optim.Adam(self.embedding.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            # Forward pass (LightGCN propagation)
            all_embeddings = [self.embedding.weight]
            ego_embeddings = self.embedding.weight
            
            for _ in range(self.layers):
                ego_embeddings = torch.sparse.mm(self.graph, ego_embeddings)
                all_embeddings.append(ego_embeddings)
            
            # Final embedding is mean of all layers
            final_embeddings = torch.stack(all_embeddings, dim=1)
            final_embeddings = torch.mean(final_embeddings, dim=1)
            
            # BPR Loss
            users, pos_items, neg_items = self._sample_batch()
            
            # Get embeddings for batch
            u_emb = final_embeddings[users]
            pos_emb = final_embeddings[self.num_users + pos_items] # Offset item indices
            neg_emb = final_embeddings[self.num_users + neg_items]
            
            # Calculate scores
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            
            # Regularization (optional but recommended)
            reg_loss = (1/2) * (self.embedding.weight[users].norm(2).pow(2) + 
                                self.embedding.weight[self.num_users + pos_items].norm(2).pow(2) + 
                                self.embedding.weight[self.num_users + neg_items].norm(2).pow(2)) / float(len(users))
            
            total_loss = loss + 1e-4 * reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss.item():.4f}")
        
        # Store final embeddings
        with torch.no_grad():
            all_embeddings = [self.embedding.weight]
            ego_embeddings = self.embedding.weight
            for _ in range(self.layers):
                ego_embeddings = torch.sparse.mm(self.graph, ego_embeddings)
                all_embeddings.append(ego_embeddings)
            final_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
            
            self.user_embeddings = final_embeddings[:self.num_users].cpu().numpy()
            self.item_embeddings = final_embeddings[self.num_users:].cpu().numpy()
            
        print("LightGCN Embeddings extracted.")
        print(f"User Embeddings shape: {self.user_embeddings.shape}")
        print(f"Item Embeddings shape: {self.item_embeddings.shape}")

class GCNEmbedding(GraphEmbedding):
    def __init__(self, file_path, k=10, layers=2, **kwargs):
        super().__init__(file_path, k, **kwargs)
        self.layers = layers
        self.embedding = None
        self.W = None # Weight matrices for each layer

    def train(self):
        if self.graph is None:
            self.load_data()
            
        print(f"Training GCN with k={self.k}, layers={self.layers}...")
        
        # Initialize embeddings
        self.embedding = nn.Embedding(self.num_users + self.num_items, self.k).to(self.device)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Initialize weights for each layer: W_l (k x k)
        self.W = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.k, self.k)) for _ in range(self.layers)]).to(self.device)
        for w in self.W:
            nn.init.xavier_uniform_(w)
            
        optimizer = optim.Adam(list(self.embedding.parameters()) + list(self.W.parameters()), lr=self.lr)
        
        for epoch in range(self.epochs):
            # Forward pass (GCN)
            # H^{(l+1)} = \sigma(\tilde{A} H^{(l)} W^{(l)})
            h = self.embedding.weight
            
            for i in range(self.layers):
                h = torch.sparse.mm(self.graph, h) # \tilde{A} H
                h = torch.mm(h, self.W[i]) # * W
                h = torch.relu(h) # \sigma
            
            final_embeddings = h
            
            # BPR Loss
            users, pos_items, neg_items = self._sample_batch()
            
            u_emb = final_embeddings[users]
            pos_emb = final_embeddings[self.num_users + pos_items]
            neg_emb = final_embeddings[self.num_users + neg_items]
            
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            h = self.embedding.weight
            for i in range(self.layers):
                h = torch.sparse.mm(self.graph, h)
                h = torch.mm(h, self.W[i])
                h = torch.relu(h)
            final_embeddings = h
            
            self.user_embeddings = final_embeddings[:self.num_users].cpu().numpy()
            self.item_embeddings = final_embeddings[self.num_users:].cpu().numpy()
            
        print("GCN Embeddings extracted.")

class GATEmbedding(GraphEmbedding):
    def __init__(self, file_path, k=10, layers=1, heads=2, **kwargs):
        super().__init__(file_path, k, **kwargs)
        self.layers = layers
        self.heads = heads
        self.embedding = None
        # Simplified GAT: We will use a single layer GAT-like propagation for demonstration
        # Implementing full multi-head GAT from scratch with sparse tensors is complex.
        # We will implement a simplified version:
        # Attention is computed based on node features.
        
    def train(self):
        if self.graph is None:
            self.load_data()
            
        print(f"Training GAT with k={self.k} (Simplified)...")
        
        # Initialize embeddings
        self.embedding = nn.Embedding(self.num_users + self.num_items, self.k).to(self.device)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Attention mechanism parameters
        # a: 2*k -> 1
        self.a = nn.Parameter(torch.FloatTensor(2 * self.k, 1)).to(self.device)
        nn.init.xavier_uniform_(self.a)
        
        self.W = nn.Parameter(torch.FloatTensor(self.k, self.k)).to(self.device)
        nn.init.xavier_uniform_(self.W)
        
        optimizer = optim.Adam(list(self.embedding.parameters()) + [self.a, self.W], lr=self.lr)
        
        # For sparse GAT, we need edge indices
        # self.graph is a sparse tensor. We can get indices.
        indices = self.graph.coalesce().indices()
        src, dst = indices[0], indices[1]
        
        for epoch in range(self.epochs):
            # Forward pass
            h = self.embedding.weight
            Wh = torch.mm(h, self.W) # (N, k)
            
            # Compute attention scores for edges
            # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            Wh_src = Wh[src]
            Wh_dst = Wh[dst]
            
            # Concatenate
            cat = torch.cat([Wh_src, Wh_dst], dim=1) # (E, 2k)
            e = torch.matmul(cat, self.a).squeeze() # (E,)
            e = torch.nn.functional.leaky_relu(e, negative_slope=0.2)
            
            # Softmax over neighbors
            # We need to use scatter_softmax or similar.
            # Since we don't have torch_scatter, we can approximate or implement a basic softmax per row.
            # For simplicity in this "from scratch" version without extra libs:
            # We will use the normalized adjacency structure but weight it by 'e'.
            # This is a simplification. True GAT requires softmax over neighbors.
            # Workaround: exp(e) / sum(exp(e)) per dst node.
            
            # Let's use a simpler attention approximation:
            # Just use the attention coefficients as edge weights in the sparse matrix
            # But we need to normalize them.
            
            # Since implementing efficient sparse softmax is hard without torch_scatter,
            # we will use a simplified attention:
            # alpha_ij = sigmoid(e_ij)  (instead of softmax)
            alpha = torch.sigmoid(e)
            
            # Create new sparse adjacency with attention weights
            adj_att = torch.sparse_coo_tensor(indices, alpha, self.graph.shape).to(self.device)
            
            # Propagation: h' = adj_att * Wh
            h_prime = torch.sparse.mm(adj_att, Wh)
            h_prime = torch.nn.functional.elu(h_prime)
            
            final_embeddings = h_prime
            
            # BPR Loss
            users, pos_items, neg_items = self._sample_batch()
            
            u_emb = final_embeddings[users]
            pos_emb = final_embeddings[self.num_users + pos_items]
            neg_emb = final_embeddings[self.num_users + neg_items]
            
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            # Final forward
            h = self.embedding.weight
            Wh = torch.mm(h, self.W)
            Wh_src = Wh[src]
            Wh_dst = Wh[dst]
            cat = torch.cat([Wh_src, Wh_dst], dim=1)
            e = torch.matmul(cat, self.a).squeeze()
            e = torch.nn.functional.leaky_relu(e, negative_slope=0.2)
            alpha = torch.sigmoid(e)
            adj_att = torch.sparse_coo_tensor(indices, alpha, self.graph.shape).to(self.device)
            h_prime = torch.sparse.mm(adj_att, Wh)
            final_embeddings = torch.nn.functional.elu(h_prime)
            
            self.user_embeddings = final_embeddings[:self.num_users].cpu().numpy()
            self.item_embeddings = final_embeddings[self.num_users:].cpu().numpy()
            
        print("GAT Embeddings extracted.")