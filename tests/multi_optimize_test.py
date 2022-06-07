import pytest
import torch

from src.nueramic_mathml import bfgs, gd_constant_step, gd_frac_step, gd_optimal_step, nonlinear_cgm

test_functions = [  # func, [a, b], true_point
    (lambda x: (x ** 2).sum(), torch.tensor([-2]), torch.tensor([0]), 'parabola'),
    (lambda x: (x ** 2).sum(), torch.arange(0, 1000).double(), torch.zeros(1000).double(), 'paraboloid')
]


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_bfgs(function, x0, expected, name):
    assert bfgs(function, x0)[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_gd_c(function, x0, expected, name):
    assert gd_constant_step(function, x0)[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_gd_f(function, x0, expected, name):
    assert gd_frac_step(function, x0)[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_gd_o(function, x0, expected, name):
    assert gd_optimal_step(function, x0)[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_gd_o(function, x0, expected, name):
    assert nonlinear_cgm(function, x0)[0] == pytest.approx(expected, abs=1e-5)
