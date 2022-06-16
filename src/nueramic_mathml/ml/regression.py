from __future__ import annotations

import sys
from itertools import combinations_with_replacement

import numpy as np
import torch

if sys.version_info >= (3, 8):
    pass
else:
    pass


class LinearRegression(torch.nn.Module):
    """

    Model:

    .. math::

        \\hat y(x) = w_0 + w_1 \\cdot x_1 + w_2 \\cdot x_2 + \\dots + w_m \\cdot x_m

    """

    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.bias = bias
        self.w = None
        self.best_state = None
        self.best_loss = torch.inf

    def init_weights(self, x: torch.Tensor) -> None:
        """
        Initializing weights

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        """

        self.w = torch.nn.Linear(x.shape[1], 1, bias=self.bias)
        self.best_state = self.w.state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns x @ w + b, w is the weights for each parameter, and b is the bias

        .. math::

            \\hat Y_{n \\times 1} = X_{n \\times m} \\cdot W_{m \\times 1} + b_{n \\times 1} =
            \\begin{bmatrix}
            w_1 x_{1, 1} + w_2 x_{1, 2} + \\dots + w_m + x_{1, m} \\\\
            \\vdots \\\\
            w_1 x_{n, 1} + w_2 x_{n, 2} + \\dots + w_m + x_{n, m} \\\\
            \\end{bmatrix}

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        :return: probabilities of 1st class
        """
        if self.w is None:
            self.init_weights(x)

        return self.w(x)

    def fit(self,
            x: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 1000,
            lr: float = 1e-4,
            l1_constant: float = 0.1,
            l2_constant: float = 0.1,
            show_epoch: int = 0) -> torch.nn.Module:
        """
        Returns trained model Linear Regression

        .. rubric:: Target function

        Training happens by minimizing loss function:

        .. math::

            \\mathcal{L}(w) = \\lambda_{1} \\Vert w \\Vert_{1} + \\lambda_{2} \\Vert w \\Vert_{2} +
            \\frac{1}{n}\\sum_{i = 1}^{n} (x_{i} w - y_i)^2 \\longrightarrow \\min_{w}

        :math:`x_{i} \\in \\mathbb{R}^{1\\times m}, w \\in \\mathbb{R}^{m \\times 1}, y_{i} \\in \\mathbb{R}^{1}`

        :param x: training set
        :param y: target value
        :param epochs: max number of sgd implements
        :param lr: Adam optimizer learning rate
        :param show_epoch: amount of showing epochs
        :param l1_constant: parameter of l1 regularization
        :param l2_constant: parameter of l2 regularization
        :return: trained model
        """
        x = x.float()
        y = y.float().flatten()

        model(x)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_constant)
        print_epochs = np.unique(np.linspace(1, epochs, min(epochs, show_epoch), dtype=int))

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            output = criterion(self.forward(x).flatten(), y.flatten())
            print(output.item())
            if output.item() < self.best_loss:
                self.best_state = self.state_dict()

            if l1_constant > 0:
                for layer in self.parameters():
                    output += l1_constant * layer.data.sum()

            output.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch + 1 in print_epochs:
                    self.print(f'Epoch: {epoch: 5d} | CrossEntropyLoss: {output.item(): 0.5f}')

        else:
            self.load_state_dict(self.best_state)

        return self


class PolynomialRegression(torch.nn.Module):
    """

    Model:

    .. math::

        \\hat y(x) = w_0 + w_1 \\cdot x_1 + w_2 \\cdot x_2 + \\dots + w_m \\cdot x_m

    """

    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.bias = bias
        self.w = None
        self.best_state = None
        self.best_loss = torch.inf


# def polynomial_regression(x: np.ndarray,
#                           y: np.ndarray,
#                           degree: int,
#                           reg_type: Literal['l1', 'l2', None] = None,
#                           epsilon: float = 1e-4,
#                           const_l1: float = 1e-1,
#                           const_l2: float = 1e-1) -> np.ndarray:
#     """
#     Make polynomial regression with a tikhonov or a lasso regularization or without any regularization.
#     Step: 1. Get x, 2. Make polynomial features, 3. Make linear regression::
#
#         >>> x_ = np.array([[-1], [0], [1]])
#         >>> y_ = np.array([1, 0, 1])
#         >>> np.round(polynomial_regression(x_, y_, 2))
#         [ 0. -0.  1.]
#
#     :param x: array of predictors
#     :param y: array of variable to predict
#     :param degree: degree for PolynomialFeatures
#     :param reg_type: type of regularization
#     :param epsilon: accuracy for optimization methods
#     :param const_l1: constant for L1 regularization
#     :param const_l2: constant for L2 regularization
#     :return: array of regression coefficients
#     """
#


class PolyTransform:

    """
    Makes polynomial functions of degree :math:`k`

    We have matrix :math:`x \\in \\mathbb{R}^{n \\times m}` with m features and n observations.
    Let :math:`c_1` -- first column of :math:`x`, :math:`c_2` -- second column of :math:`x` and etc.
    And we make all features like:

    .. math::

        c_1^{\\alpha_1} \\cdot c_2^{\\alpha_2} \\cdot \\dots \\cdot c_m^{\\alpha_m}

    subject to :math:`\\alpha_i \\in \\mathbb{Z}_+` and :math:`\\sum_{i = 1}^{m} \\alpha_i \\leq k`

    :param feature_names:

    """

    def __init__(self, degree: int = 2):
        """
        :param degree: max degree of polynomial features
        """
        self.degree = degree
        self.feature_names = []
        self.columns = []
        self.number_features = None

    def fit(self, x: torch.Tensor, columns: list[str] | None = None):

        column_indexes = list(range(x.shape[1]))

        # make feature names from column names
        if columns is not None:
            for k in range(self.degree + 1):
                self.feature_names.extend(combinations_with_replacement(columns, k))

        # make indexes sets
        for k in range(self.degree + 1):
            self.columns.extend(combinations_with_replacement(column_indexes, k))

        # make columns from indexes
        if columns is None:
            for names in self.columns:
                self.feature_names.append(tuple(map(lambda x: 'col_' + str(x), names)))

        feat = ['1']

        # make feature names with
        for i in range(1, len(self.feature_names)):
            val, cnt = np.unique(self.feature_names[i], return_counts=True)
            val = list(map(str, val))
            cnt = list(map(str, cnt))

            feat.append(' '.join(map(lambda x: '^'.join(x), zip(val, cnt))))

        self.feature_names = feat

        self.number_features = len(self.feature_names)

    def transform(self, x: torch.Tensor) -> torch.Tensor:

        transformed = [torch.ones(x.shape[0], 1)]

        for cols in self.columns[1:]:
            transformed.append(x[:, cols].prod(dim=1, keepdim=True))

        return torch.concat(transformed, dim=1)

    def fit_transform(self, x: torch.tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)


if __name__ == '__main__':
    model = LinearRegression()
    _x = torch.rand(10, 3)
    _y = _x @ torch.tensor([[1, 2, 3.]]).T + 5

    model.fit(_x, _y, lr=1e-2)
    print(model(_x) - _y)
