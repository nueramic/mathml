import pytest
import torch

from sklearn.datasets import make_blobs
from src.nueramic_mathml.ml import LogisticRegression, LogisticRegressionRBF, SVM
from src.nueramic_mathml.ml import metrics

torch.random.manual_seed(7)
_x, _y = make_blobs(1000, centers=2, random_state=7)
_x, _y = torch.tensor(_x), torch.tensor(_y)

tests = [
    (_x, _y,)
]


@pytest.mark.parametrize('x, y', tests)
def test_log(x, y):
    model = LogisticRegression()
    model.fit(x, y)

    assert metrics.accuracy(y, model.predict(x)) > 0.95


@pytest.mark.parametrize('x, y', tests)
def test_log_rbf(x, y):
    model = LogisticRegressionRBF(x[:100])
    model.fit(x, y)

    assert metrics.accuracy(y, model.predict(x)) > 0.95


@pytest.mark.parametrize('x, y', tests)
def test_svm(x, y):
    model = SVM()
    model.fit(x, y)

    assert metrics.accuracy(y, model.predict(x)) > 0.95
