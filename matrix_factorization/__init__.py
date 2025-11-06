### matrix_factorization/__init__.py
from enum import Enum

class Method(Enum):
    SVD = 'svd'
    PCA = 'pca'
    NeuMF = 'neumf'
    GCN = 'gcn'
    GAT = 'gat'