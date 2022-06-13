import sys
from typing import Iterable, Optional

import numpy as np
import torch

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict


class NueSGD:

    def __init__(self, parameters: Iterable, lr: float = 1e-4):
        """
        Implementation of classic SGD (stochastic gradient descent) optimization algorithm.

        :param parameters: model parameters
        :param lr: learning rate. Multiplier of gradient step: x = x - lr * grad(x)
        """
        self.parameters = list(parameters)
        self.lr = lr

    @torch.no_grad()
    def step(self) -> None:
        """
        Update parameters data

        W = W - lr * Grad(W)

        :return: None
        """
        for param in self.parameters:
            param.data -= self.lr * param.grad.data

    def zero_grad(self) -> None:
        """
        Make the gradients equal to zero

        :return: None
        """
        for param in self.parameters:
            if param.grad is not None:
                param.grad.data.zero_()


def fit_by_sgd(model: torch.nn.Module,
               x: torch.Tensor,
               y: torch.Tensor,
               epochs: int = 1,
               batch_size: int = 1,
               lr: float = 1e-4,
               verbose: bool = True,
               lamb: float = 0.3) -> [torch.nn.Module, dict]:
    """
    Function apply MySGD optimizer, and train model.

    :param model: some pytorch model that can be called and have a ".loss" method
    :param x: training set
    :param y: target value
    :param epochs: max number of sgd implements
    :param lr: learning rate for sgd step
    :param batch_size: size of batch for each epoch. default is 1
    :param verbose: print flag 10 iterations of training
    :param lamb: rate of history loss evaluation
    :return: trained model and history
    """
    optimizer = NueSGD(model.parameters(), lr=lr)
    q_new = model.loss(x, y)  # Q - functional evaluation
    print_epochs = np.geomspace(1, epochs, 10, dtype=int)

    history = {'q_loss': []}

    for epoch in range(epochs):
        i = torch.randint(0, x.shape[0], [batch_size])  # choose batch

        # optimization
        optimizer.zero_grad()
        loss = model.loss(x[i], y[i])
        loss.backward()
        optimizer.step()

        # Q calculation
        q_pre = q_new
        q_new = q_pre * (1 - lamb) + loss.item() * lamb

        # history updating
        history['q_loss'].append(q_new.item())

        if epoch + 1 in print_epochs and verbose:
            model.print(f'epoch: {epoch + 1: 5d} | Q: {q_new: 0.4f}')

        if abs(q_new - q_pre) < 1e-6:
            break

    return model, history


class HistorySA(TypedDict):

    type_ball: str
    iteration: list
    point: Optional[list]
    loss: list


class SimulatedAnnealing:

    def __init__(self,
                 parameters: Iterable,
                 type_area: Literal['circle', 'neighborhood'] = 'circle',
                 init_temp: float = 10_000,
                 radius: float = 1,
                 temp_multiplier: float = 0.9):
        """
        Initialization of SimulatedAnnealing algorithm.

        :param parameters: model parameters
        :param type_area: if type_area is circle, new point (x_k+1) would be chosen from Ball(center = 0, radius).
        if type_area is neighborhood, new point would be chosen from Ball(center = x_k, radius). [1]_
        :param init_temp: initial temperature. Default is 10_000
        :param radius: ball's radius


        .. rubric:: References

        .. [1] https://en.wikipedia.org/wiki/Ball_(mathematics)
        """
        self.parameters = list(parameters)
        self.temp = init_temp
        self.area = type_area
        self.radius = radius
        self.temp_multiplier = temp_multiplier
        self.history: HistorySA = {
            'type_ball': type_area,
            'iteration': [],
            'point': None if len(self.parameters) > 1 else [],
            'loss': []
        }

    @torch.no_grad()
    def step(self) -> None:
        """


        :return:
        """
