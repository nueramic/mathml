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
    type_ball: tuple
    iteration: list
    point: Optional[list]
    loss: list


class SimulatedAnnealing:

    def __init__(self,
                 model: torch.nn.Module,
                 type_center: Literal['zero', 'neighborhood'] = 'neighborhood',
                 init_temp: float = 10_000,
                 radius: float = 4,
                 temp_multiplier: float = 0.9):
        """
        Initialization of SimulatedAnnealing algorithm. Minimize real number models (non-discrete)

        :param model: some pytorch model
        :param type_center: if type_center is zero, new point (x_k+1) would be chosen from Ball(center = 0, radius).
                            elif neighborhood, new point would be chosen from Ball(center = x_k, radius). [1]_
        :param init_temp: initial temperature. Default is 10_000
        :param radius: ball's radius

        .. rubric:: References
        .. [1] https://en.wikipedia.org/wiki/Ball_(mathematics)

        """

        self.temp = init_temp
        self.center = type_center
        self.radius = radius
        self.temp_multiplier = temp_multiplier
        self.history = {
            'type_ball': (type_center, radius),
            'iteration': [],
            'point': None if len(list(model.parameters())) > 1 else [],
            'loss': [],
            'best_point': None if len(list(model.parameters())) > 1 else [],
            'best_loss': []
        }
        self.best_state = model.state_dict()
        self.model = model
        self.min_temp = 1e-8
        self.best_loss = torch.inf

    @torch.no_grad()
    def optimize(self, x: torch.Tensor, y: torch.Tensor):
        """
        Generator of Simulated Annealing steps.

        :math:`\\rule{125mm}{0.7pt} \\\\`
        :math:`c = x_{pre} \\text{ if type area is `neighborhood' else } c = \\theta - \\text{zero} \\\\`
        :math:`x_{cur} \\sim \\mathcal{U}(c, r) \\qquad p \\sim \\mathcal{U}[0, 1]\\\\`

        :math:`\\text{if } f(x_{cur}) < f(x_{best}): \\\\`
        :math:`\\qquad x_{pre} = x_{best} = x_{cur}\\\\`

        :math:`\\text{elif } \\displaystyle \\exp\\left(\\frac{f(x_{pre}) - f(x_{cur})}{T}\\right) > p:\\\\`
        :math:`\\qquad x_{pre} = x_{cur}\\\\`

        :math:`T = T \\cdot \\delta`
        :math:`\\rule{125mm}{0.7pt} \\\\`

        :param x: training set
        :param y: target value

        .. code-block:: python3

            >>> torch.random.manual_seed(7)

            >>> xr = torch.rand(100, 3)
            >>> w = torch.tensor([[1., 2., 3.]]).T
            >>> yr = xr @ w + 2

            >>> model = torch.nn.Sequential(torch.nn.Linear(3, 1))
            >>> model.loss = lambda _x, _y: torch.nn.MSELoss()(model(_x), _y)

            >>> optimizer = SimulatedAnnealing(model, temp_multiplier=0.01)

            >>> for verbose in optimizer.optimize(xr, yr):
            >>>     print(verbose)
            iteration:    1 | loss: 97.5745
            iteration:    2 | loss: 231.5806
            iteration:    3 | loss: 3.4633
            iteration:    4 | loss: 3.7009
            iteration:    5 | loss: 26.9238
            iteration:    6 | loss: 6.5509
            iteration:    7 | loss: 21.4261

            >>> model.loss(xr, yr)
            tensor(3.4633, grad_fn=<MseLossBackward0>)


        """
        while self.temp > self.min_temp:

            pre_loss = self.model.loss(x, y).item()

            # init loss, iter, point
            if len(self.history['iteration']) == 0:
                self.best_loss = pre_loss
                self.history['iteration'].append(0)
                self.history['loss'].append(pre_loss)
                self.history['best_loss'].append(pre_loss)

                if self.history['point'] is not None:
                    for param in self.model.parameters():
                        self.history['point'].append(param.data)
                        self.history['best_point'].append(param.data)

            # choose new point
            for param in self.model.parameters():
                c = param.data if self.center == 'neighborhood' else torch.zeros_like(param.data)
                x_cur = (torch.rand_like(c) - 0.5) * 2 * self.radius + c
                param.data = x_cur

                if self.history['point'] is not None:
                    self.history['point'].append(param.data.flatten())

            # calc new loss
            cur_loss = self.model.loss(x, y).item()

            # check criterion
            if cur_loss < self.best_loss:
                self.best_state = self.model.state_dict()
                self.best_loss = cur_loss

            if cur_loss <= pre_loss:
                pass

            elif torch.e ** ((pre_loss - cur_loss) / self.temp) > torch.rand(1):
                pass

            else:
                self.model.load_state_dict(self.best_state)

            # update history
            self.history['iteration'].append(self.history['iteration'][-1] + 1)
            self.history['loss'].append(cur_loss)
            self.history['best_loss'].append(self.best_loss)

            if self.history['point'] is not None:
                self.history['best_point'].append(list(self.best_state.values())[0])

            # update temp
            self.temp *= self.temp_multiplier

            yield f'iteration: {self.history["iteration"][-1]:4d} | loss: {self.history["loss"][-1]:.4f}'

        else:
            # set best parameters
            self.model.load_state_dict(self.best_state)
