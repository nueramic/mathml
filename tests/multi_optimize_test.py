import pytest
import torch

from src.nueramic_mathml import bfgs, HiddenPrints

test_functions = [  # func, [a, b], true_point
    (lambda x: (x ** 2).sum(), torch.tensor([-2]), torch.tensor([0]), 'parabola'),
    (lambda x: (x ** 2).sum(), torch.arange(0, 1000).double(), torch.zeros(1000).double(), 'paraboloid'),
]


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_bfgs(function, x0, expected, name):
    with HiddenPrints():
        assert bfgs(function, x0)[0] == pytest.approx(expected, abs=1e-5)
