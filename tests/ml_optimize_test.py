import pytest
import torch.nn

from src.nueramic_mathml.ml import NueSGD, SimulatedAnnealing

torch.random.manual_seed(7)
x_ = torch.rand(100, 2) * 100
y_ = x_ @ torch.tensor([[1., 2.]]).T
model_ = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
criterion = torch.nn.MSELoss()
model_.loss = lambda x, y: criterion(model_(x), y)


test = [
    (model_, x_, y_, torch.tensor([1, 2]).float())
]


@pytest.mark.parametrize('model, x, y, expected', test)
def test_sgd(model, x, y, expected):
    optimizer = NueSGD(model)
    optimizer.optimize(x, y, epochs=1000)

    assert torch.allclose(list(model.parameters())[0].data.flatten(), expected)


@pytest.mark.parametrize('model, x, y, expected', test)
def test_sa(model, x, y, expected):
    optimizer = SimulatedAnnealing(model)
    optimizer.optimize(x, y)

    assert torch.allclose(list(model.parameters())[0].data.flatten(), expected, rtol=1e-1)
