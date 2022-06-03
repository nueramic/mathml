import pytest

from src.nueramic_mathml.calculus import *

test_functions = [
    (lambda x: (x ** 2).sum(), torch.tensor([1., 2.]), torch.tensor([2., 4.])),
    (lambda x: 1, torch.tensor([0., 0.]), torch.tensor([0., 0.])),
    (lambda x: torch.sin(x[0] * x[1]), torch.tensor([torch.pi / 2, 2]), torch.tensor([-2., -torch.pi / 2])),
    (lambda x: x.sum(), torch.tensor([5] * 1000), torch.tensor([1] * 1000)),
    (lambda x: (x ** 2).sum(), torch.arange(1, 1000), torch.arange(1, 1000) * 2),
    (lambda x: x.prod(), torch.arange(1, 10), torch.arange(1, 10).prod() / torch.arange(1, 10))
]


@pytest.mark.parametrize('function, x0, expected', test_functions)
def test_gradient(function, x0, expected):
    assert torch.allclose(gradient(function, x0.double()), expected.double(), atol=1e-6, rtol=1e-4)
