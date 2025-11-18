import numpy as np
import scipy.sparse as sp
import torch


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    
    Normalized adjacency: D^(-1/2) @ A @ D^(-1/2)
    
    Args:
        adj: scipy sparse matrix
    
    Returns:
        Normalized sparse matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def add_self_loops(adj):
    """
    Add self-loops to adjacency matrix.
    
    Args:
        adj: scipy sparse matrix [N, N]
    
    Returns:
        Adjacency matrix with self-loops
    """
    return adj + sp.eye(adj.shape[0])


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy sparse matrix to torch sparse tensor.
    
    Args:
        sparse_mx: scipy sparse matrix
    
    Returns:
        torch.sparse.FloatTensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    # Use torch.sparse_coo_tensor instead of deprecated FloatTensor
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)



def build_item_cooccurrence_graph(user_item_matrix, threshold=0):
    """
    Build item co-occurrence graph from user-item interaction matrix.
    
    Args:
        user_item_matrix: [num_users, num_items] sparse matrix
        threshold: Minimum co-occurrence count to create edge
    
    Returns:
        item_item_adj: [num_items, num_items] sparse adjacency matrix
    """
    # Item-Item co-occurrence: I^T @ I
    item_item = user_item_matrix.T @ user_item_matrix
    
    # Remove self-loops (will be added later if needed)
    item_item.setdiag(0)
    
    # Apply threshold
    if threshold > 0:
        item_item = item_item.multiply(item_item >= threshold)
    
    # Make binary (optional)
    item_item = (item_item > 0).astype(np.float32)
    
    return item_item


def build_user_item_bipartite_graph(user_item_matrix):
    """
    Build user-item bipartite graph.
    
    Args:
        user_item_matrix: [num_users, num_items] sparse matrix
    
    Returns:
        bipartite_adj: [(num_users + num_items), (num_users + num_items)] sparse matrix
    """
    num_users, num_items = user_item_matrix.shape
    
    # Create bipartite adjacency matrix
    # [  0    R  ]
    # [ R^T   0  ]
    zero_user = sp.csr_matrix((num_users, num_users))
    zero_item = sp.csr_matrix((num_items, num_items))
    
    bipartite_adj = sp.vstack([
        sp.hstack([zero_user, user_item_matrix]),
        sp.hstack([user_item_matrix.T, zero_item])
    ])
    
    return bipartite_adj


def preprocess_adj(adj, normalize=True, add_self_loop=True):
    """
    Preprocess adjacency matrix for GNN.
    
    Args:
        adj: scipy sparse matrix
        normalize: Whether to apply symmetric normalization
        add_self_loop: Whether to add self-loops
    
    Returns:
        torch.sparse.FloatTensor
    """
    if add_self_loop:
        adj = add_self_loops(adj)
    
    if normalize:
        adj = normalize_adj(adj)
    
    return sparse_mx_to_torch_sparse_tensor(adj)


def degree_normalize_adj(adj):
    """
    Row-normalize adjacency matrix (for GAT).
    
    D^(-1) @ A
    
    Args:
        adj: scipy sparse matrix
    
    Returns:
        Row-normalized sparse matrix
    """
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(adj)


class GraphBuilder:
    """Helper class for building various graph structures."""
    
    def __init__(self, user_item_matrix):
        """
        Args:
            user_item_matrix: [num_users, num_items] scipy sparse matrix
        """
        self.user_item_matrix = user_item_matrix
        self.num_users, self.num_items = user_item_matrix.shape
    
    def build_item_graph(self, threshold=0, normalize=True, add_self_loop=True):
        """Build and preprocess item co-occurrence graph."""
        adj = build_item_cooccurrence_graph(self.user_item_matrix, threshold)
        return preprocess_adj(adj, normalize, add_self_loop)
    
    def build_bipartite_graph(self, normalize=True, add_self_loop=True):
        """Build and preprocess user-item bipartite graph."""
        adj = build_user_item_bipartite_graph(self.user_item_matrix)
        return preprocess_adj(adj, normalize, add_self_loop)
    
    def get_statistics(self):
        """Get graph statistics."""
        item_adj = build_item_cooccurrence_graph(self.user_item_matrix)
        
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': self.user_item_matrix.nnz,
            'sparsity': 1 - (self.user_item_matrix.nnz / (self.num_users * self.num_items)),
            'avg_items_per_user': self.user_item_matrix.nnz / self.num_users,
            'avg_users_per_item': self.user_item_matrix.nnz / self.num_items,
            'item_graph_edges': item_adj.nnz,
            'item_graph_density': item_adj.nnz / (self.num_items ** 2)
        }


def edge_index_to_sparse_adj(edge_index, num_nodes, edge_weight=None):
    """
    Convert edge_index (PyG format) to sparse adjacency matrix.
    
    Args:
        edge_index: [2, num_edges] tensor
        num_nodes: Number of nodes
        edge_weight: [num_edges] tensor (optional)
    
    Returns:
        torch.sparse.FloatTensor [num_nodes, num_nodes]
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))
    
    return torch.sparse.FloatTensor(
        edge_index,
        edge_weight,
        torch.Size([num_nodes, num_nodes])
    )


def sparse_adj_to_edge_index(adj):
    """
    Convert sparse adjacency matrix to edge_index (PyG format).
    
    Args:
        adj: torch.sparse.FloatTensor or scipy sparse matrix
    
    Returns:
        edge_index: [2, num_edges] tensor
        edge_weight: [num_edges] tensor
    """
    if isinstance(adj, torch.sparse.FloatTensor):
        edge_index = adj._indices()
        edge_weight = adj._values()
    else:
        # scipy sparse matrix
        adj = adj.tocoo()
        edge_index = torch.from_numpy(
            np.vstack((adj.row, adj.col)).astype(np.int64)
        )
        edge_weight = torch.from_numpy(adj.data.astype(np.float32))
    
    return edge_index, edge_weight
