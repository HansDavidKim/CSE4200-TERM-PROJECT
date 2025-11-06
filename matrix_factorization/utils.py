### matrix_factorization/utils,py
from matrix_factorization import Method

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

def get_embedder(method: Method, n_components: int):
    if method == Method.PCA:
        model = PCA(n_components=n_components, svd_solver='auto', random_state=42)
    
    elif method == Method.SVD:
        model = TruncatedSVD(n_components=n_components, random_state=42)

    else:
        raise ValueError("Not Available Method")

    return model