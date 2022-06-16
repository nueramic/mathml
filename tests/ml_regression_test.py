import pytest
import torch

from src.nueramic_mathml.ml import LinearRegression, PolynomialRegression, ExponentialRegression
from src.nueramic_mathml.ml import metrics

torch.random.manual_seed(7)
x_1 = torch.rand(1000, 4)
y_1 = x_1 @ torch.tensor([[1., 2., 3., 4.]]).T + 5

x_2 = torch.rand(10_000, 40)
y_2 = x_2 @ torch.tensor([[1., 2., 3., 4.] * 10]).T + 5

test_regression = [
    (x_1, y_1),
    (x_2, y_2)
]


@pytest.mark.parametrize('x, y', test_regression)
def test_linear(x, y):
    model = LinearRegression()
    model.fit(x, y)
    yp = model(x).flatten()
    assert metrics.r2_score(y.flatten(), yp) > 0.99


@pytest.mark.parametrize('x, y', test_regression)
def test_poly1(x, y):
    model = PolynomialRegression(1)
    model.fit(x, y)
    yp = model(x).flatten()
    assert metrics.r2_score(y.flatten(), yp) > 0.99


@pytest.mark.parametrize('x, y', test_regression)
def test_poly2(x, y):
    model = PolynomialRegression(2)
    model.fit(x, y)
    yp = model(x).flatten()
    assert metrics.r2_score(y.flatten(), yp) > 0.99


@pytest.mark.parametrize('x, y', test_regression)
def test_expon(x, y):
    model = ExponentialRegression()
    model.fit(x, y)
    yp = model(x).flatten()
    assert metrics.r2_score(y.flatten(), yp) > 0.95
