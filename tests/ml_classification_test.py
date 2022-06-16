import pytest
import torch

from sklearn.datasets import make_blobs
from src.nueramic_mathml import LogisticRegression, LogisticRegressionRBF
from src.nueramic_mathml.ml import metrics

torch.random.manual_seed(7)
_x, _y = make_blobs(1000, centers=2, random_state=7)
_x, _y = torch.tensor(_x), torch.tensor(_y)

tests = [
    (_x, _y,)
]


@pytest.mark.parametrize('x, y', tests)
def test_log(x, y):
    model = LogisticRegression(2)
    model.fit(x, y, epochs=1000)

    assert metrics.accuracy(y, model.predict(x)) > 0.95


@pytest.mark.parametrize('x, y', tests)
def test_log_rbf(x, y):
    model = LogisticRegressionRBF(x[:100])
    model.fit(x, y, epochs=2000)

    assert metrics.accuracy(y, model.predict(x)) > 0.95
