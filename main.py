import typer, numpy as np
app = typer.Typer()

from matrix_factorization.models import EmbeddingModel
from matrix_factorization import Method

@app.command()
def hello_world():
    print('Hello, World!')

@app.command()
def train_embedding():
    data = np.random.rand(10, 10)
    model = EmbeddingModel(method=Method.PCA, n_components=2)

    model.fit(data)

if __name__ == '__main__':
    app()