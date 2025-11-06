### matrix_factorization/linear_mf.py
###
### SVD, PCA Matrix Factorization
### Implemented with scikit-learn

### Enumerator Declared in __init__.py 
from matrix_factorization import Method
from matrix_factorization.utils import get_embedder

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

class EmbeddingModel:
    def __init__(self, method : Method, n_components: int):
        self.model = get_embedder(method=method, n_components=n_components)

    def fit(self, data: np.ndarray):
        self.model.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.model.transform(data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(data)

class NeuralEmbedder(nn.Module, ABC):
    def __init__(self, n_components: int, random_state: int):
        super().__init__()
        self.n_components = n_components

        ### Fixed Random State for Experiment Reproducibility
        self.random_state = random_state

    ### Adopted 'fit' instead of train
    ### for matching interface between neural embedding models and linear embedding models.
    @abstractmethod
    def fit(self, data: np.ndarray):
        pass

    @abstractmethod
    def transform(self, data: np.ndarray):
        pass

    @abstractmethod
    def fit_transform(self, data: np.ndarray):
        pass

    @abstractmethod
    def forward(self, data: torch.Tensor):
        pass

class NeuMF(NeuralEmbedder):
    def __init__(self, n_components: int, random_state: int):
        super().__init__(n_components=n_components, random_state=random_state)
    
    ### Type-casting required :
    ### np.ndarray -> torch.Tensor

    ### TODO : Implement NeuMF fit, transform, fit_transform
    ### You must use tqdm for training so we can observe the progress.
    def fit(self, data: np.ndarray):
        pass

    def transform(self, data: np.ndarray):
        pass

    def fit_transform(self, data: np.ndarray):
        pass

class GCN(NeuralEmbedder):
    def __init__(self, n_components: int, random_state: int):
        super().__init__(n_components=n_components, random_state=random_state)

    def fit(self, data):
        pass

    def transform(self, data: np.ndarray):
        pass

    def fit_transform(self, data: np.ndarray):
        pass

class GAT(NeuralEmbedder):
    def __init__(self, n_components: int, random_state: int):
        super().__init__(n_components=n_components, random_state=random_state)

    def fit(self, data):
        pass

    def transform(self, data: np.ndarray):
        pass

    def fit_transform(self, data: np.ndarray):
        pass

### For Testing
if __name__ == '__main__':
    model = EmbeddingModel(method=Method.PCA, n_components=2)

    data = np.random.rand(10,10)
    print('Data before Transformation')
    print(data)

    embedding = model.fit_transform(data)
    print('Data after Transformation')
    print(embedding)