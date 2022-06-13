from __future__ import annotations

import torch


class LinearRegression(torch.nn.Module):

    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.bias = bias
        self.w = None

    def init_weights(self, x: torch.Tensor) -> None:
        """
        Initializing weights

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        """

        self.w = torch.nn.Linear(x.shape[0], 1, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns w @ x + b, w is the weights for each parameter, and b is the bias

        .. math::

            \\hat Y_{n \\times 1} = X_{n \\times m} \\cdot W_{m \\times 1} + b_{n \\times 1} =
            \\begin{bmatrix}
            w_1 x_{1, 1} + w_2 x_{1, 2} + \\dots + w_m + x_{1, m} \\\\
            \\vdots \\\\
            w_1 x_{n, 1} + w_2 x_{n, 2} + \\dots + w_m + x_{n, m} \\\\
            \\end{bmatrix}

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        :return:
        """
        if self.w is None:
            self.init_weights(x)

        return self.w(x)

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1000) -> LinearRegression:
        """

        :param x:
        :param y:
        :param epochs:
        :return:
        """


if __name__ == '__main__':
    from sklearn.datasets import make_regression

    model = LinearRegression()
    _x, _y = make_regression()
    model(_x)
