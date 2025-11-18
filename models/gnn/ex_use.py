# example_usage.py
import numpy as np
import scipy.sparse as sp
import torch
from gcn.graph_utils import GraphBuilder, preprocess_adj
from gcn.gcn import GCNEmbedding
from gcn.gat import GATEmbedding

# 1. 데이터 준비
num_users = 1000
num_items = 500
density = 0.01

# Random user-item matrix 생성 (실제로는 데이터에서 로드)
user_item_matrix = sp.random(num_users, num_items, density=density, format='csr')

# 2. 그래프 빌드
graph_builder = GraphBuilder(user_item_matrix)

# 통계 확인
stats = graph_builder.get_statistics()
print("Graph Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Item co-occurrence graph 생성
item_adj = graph_builder.build_item_graph(threshold=2, normalize=True, add_self_loop=True)

# 3. GCN 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
item_adj = item_adj.to(device)

gcn_model = GCNEmbedding(
    num_nodes=num_items,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.5
).to(device)

# Forward pass
item_embeddings_gcn = gcn_model(item_adj)
print(f"\nGCN embeddings shape: {item_embeddings_gcn.shape}")

# 4. GAT 사용
gat_model = GATEmbedding(
    num_nodes=num_items,
    embedding_dim=128,
    hidden_dim=32,
    num_heads=4,
    num_layers=2,
    dropout=0.6
).to(device)

# Forward pass
item_embeddings_gat = gat_model(item_adj)
print(f"GAT embeddings shape: {item_embeddings_gat.shape}")

# 5. 특정 아이템들의 임베딩만 가져오기
item_ids = torch.tensor([0, 10, 20, 30], device=device)
specific_embeddings = gcn_model(item_adj, node_ids=item_ids)
print(f"\nSpecific items embeddings shape: {specific_embeddings.shape}")
