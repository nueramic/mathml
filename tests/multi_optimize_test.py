import pytest
import torch

from src.nueramic_mathml import *

test_functions = [  # func, [a, b], true_point
    (lambda x: (x ** 2).sum(), torch.tensor([-2]), torch.tensor([0]), 'parabola'),
    (lambda x: (x ** 2).sum(), torch.arange(0, 1000).double(), torch.zeros(1000).double(), 'paraboloid')
]


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_bfgs(function, x0, expected, name):
    assert bfgs(function, x0)[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_gd_c(function, x0, expected, name):
    assert gd_constant(function, x0)[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_gd_f(function, x0, expected, name):
    assert gd_frac(function, x0)[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_gd_o(function, x0, expected, name):
    assert gd_optimal(function, x0)[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('function, x0, expected, name', test_functions)
def test_cgm(function, x0, expected, name):
    assert nonlinear_cgm(function, x0)[0] == pytest.approx(expected, abs=1e-5)


test_functions_ineq_constr = [
    (
        lambda x: - torch.cos(x).sum(),
        torch.tensor([-0.4, 1], dtype=torch.float64),
        [lambda x: x[0] + 1, lambda x: x[1] + 2],
        torch.tensor([0, 0]).double()
    ),
    (
        lambda x: x.sum(),
        torch.tensor([1., 1.], dtype=torch.float64),
        [lambda x: x[0] + 1, lambda x: x[1] + 1],
        torch.tensor([-1, -1]).double()
    ),
    (
        lambda x: (x[0] + 1) ** 2 + x[1] ** 2,
        torch.tensor([0.3, 0.1], dtype=torch.float64),
        [lambda x: 5 - (x ** 2).sum()],
        torch.tensor([-1, 0]).double()
    ),
    (
        lambda x: (x[0] + 1) ** 2 + x[1] ** 2,
        torch.tensor([0.2, 0.7], dtype=torch.float64),
        [lambda x: 1 - x.abs().sum()],
        torch.tensor([-1, 0]).double()
    )
]


@pytest.mark.parametrize('function, x0, inequalities, expected', test_functions_ineq_constr)
def test_primal_dual(function, x0, inequalities, expected):
    assert primal_dual_interior(function, x0, inequalities)[0] == pytest.approx(expected, abs=1e-1)


@pytest.mark.parametrize('function, x0, inequalities, expected', test_functions_ineq_constr)
def test_log_barrier_solver(function, x0, inequalities, expected):
    assert log_barrier_solver(function, x0, inequalities)[0] == pytest.approx(expected, abs=1e-1)


test_functions_eq_constr = [
    (
        lambda x: (x ** 2).sum(),
        torch.tensor([2, -2]).double(),
        [lambda x: x[0] - 1],
        torch.tensor([1, 0])
    ),
    (
        lambda x: torch.sin(x).sum(),
        torch.tensor([0, -1]).double(),
        [lambda x: x[0] - 1],
        torch.tensor([1, -torch.pi / 2])
    ),
    (
        lambda x: (x[0] - 1) ** 2 + x[1] ** 2,
        torch.tensor([2, -2]).double(),
        [lambda x: x[1] - 1],
        torch.tensor([1, 1])
    )
]


@pytest.mark.parametrize('function, x0, equalities, expected', test_functions_eq_constr)
def test_constrained_lagrangian_solver(function, x0: torch.Tensor, equalities, expected):
    assert constrained_lagrangian_solver(function, x0, equalities)[0] == pytest.approx(expected, abs=1e-1)
