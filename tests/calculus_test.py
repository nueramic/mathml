import pytest
import torch

from src.nueramic_mathml import gradient, jacobian, hessian

# Tests Gradient
test_functions_grad = [
    (
        lambda x: (x ** 2).sum(),
        torch.tensor([1., 2.]),
        torch.tensor([2., 4.])
    ),
    (
        lambda x: 1,
        torch.tensor([0., 0.]),
        torch.tensor([0., 0.])
    ),
    (
        lambda x: torch.sin(x[0] * x[1]),
        torch.tensor([torch.pi / 2, 2]),
        torch.tensor([-2., -torch.pi / 2])
    ),
    (
        lambda x: x.sum(),
        torch.tensor([5] * 1000),
        torch.tensor([1] * 1000)
    ),
    (
        lambda x: (x ** 2).sum(),
        torch.arange(1, 1000),
        torch.arange(1, 1000) * 2
    ),
    (
        lambda x: x.prod(),
        torch.arange(1, 10),
        torch.arange(1, 10).prod() / torch.arange(1, 10)
    )
]


@pytest.mark.parametrize('function, x0, expected', test_functions_grad)
def test_gradient(function, x0, expected):
    assert torch.allclose(gradient(function, x0.double()), expected.double(), atol=1e-6, rtol=1e-4)


# Tests Jacobian
test_functions_jac = [
    (  # test from example
        [
            lambda x: x[0] ** 2 + x[1],
            lambda x: 2 * x[0] + 5 * x[1],
            lambda x: x[0] * x[1]
        ],
        torch.tensor([-1, 2]),
        torch.tensor([[-2., 1.], [2., 5.], [2., -1.]], dtype=torch.float64)
    ),
    (
        [
            lambda x: 1
        ] * 10,
        torch.arange(-10, 10).double(),
        torch.zeros(10, 20).double()
    ),
    (
        [
            lambda x: torch.sin(x).sum(),
            lambda x: torch.sin(x).sum()
        ],
        torch.ones(6) * torch.pi,
        torch.ones(2, 6, dtype=torch.float64) * -1
    )
]


@pytest.mark.parametrize('functions, x0, expected', test_functions_jac)
def test_jacobian(functions, x0, expected):
    jac = jacobian(functions, x0)
    assert torch.allclose(jac, expected)


# Tests Hessian
test_functions_hes = [
    (
        lambda x: (x ** 2).sum(),
        torch.tensor([1., 2.]),
        torch.tensor([[2., 0.], [0., 2.]]).double()
    ),
    (
        lambda x: 1,
        torch.tensor([0., 0.]),
        torch.tensor([[0., 0.], [0., 0.]]).double()
    ),
    (
        lambda x: torch.sin(x[0] + x[1]),
        torch.tensor([torch.pi / 2, torch.pi / 2]),
        torch.tensor([[0, 0], [0, 0]]).double()
    ),
]


@pytest.mark.parametrize('function, x0, expected', test_functions_hes)
def test_hessian(function, x0, expected: torch.Tensor):
    hes = hessian(function, x0)
    assert torch.allclose(hes, expected, atol=1e-6, rtol=1e-4)
