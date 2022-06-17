from __future__ import annotations

import sys
from itertools import combinations_with_replacement
from typing import Callable

import numpy as np
import torch

if sys.version_info >= (3, 8):
    pass
else:
    pass


class BaseRegressionModel(torch.nn.Module):
    """
    Base model for regression.
    
    :ivar w: weights of model
    :ivar _best_state: _best_state while model training
    :ivar _best_loss: _best_loss while model training
    """

    def __init__(self) -> None:
        """
        Initialization of base model for different regression
        """
        super(BaseRegressionModel, self).__init__()
        self.w = None
        self._best_state = None
        self._best_loss = torch.inf
        self.bias = True

    @staticmethod
    def transform(x: torch.Tensor) -> torch.Tensor:
        """
        Returns transformed x. Default is return x without changes

        :param x: torch tensor
        :return: transformed torch tensor
        """
        return x

    def forward(self, x: torch.Tensor, transformed: bool = True) -> torch.Tensor:
        """
        Returns transform(x) @ w, w is the weights for each parameter, transform(x) is some transformed matrix.

        #. Linear transformed - same matrix
        #. Polynomial transform check polynomial
        #. Exponential transform check exponential

        For linear returns x @ w + b, w is the weights for each parameter, and b is the bias

        .. math::

            \\hat Y_{n \\times 1} = X_{n \\times m} \\cdot W_{m \\times 1} + b \\cdot I_{n \\times 1} =
            \\begin{bmatrix}
            w_1 x_{1, 1} + w_2 x_{1, 2} + \\dots + w_m + x_{1, m} + b\\\\
            \\vdots \\\\
            w_1 x_{n, 1} + w_2 x_{n, 2} + \\dots + w_m + x_{n, m} + b \\\\
            \\end{bmatrix}

        For non linear:

        .. math::

            \\hat Y_{n \\times 1} = X_{\\operatorname{transformed}} \\cdot W

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        :param transformed: flag of transformed x
        :return: regression value
        """

        if self.w is None:
            self.init_weights(x)

        if not transformed:
            x = self.transform(x)

        return self.w(x)

    _forward = forward

    def init_weights(self, x: torch.Tensor) -> None:
        """
        Initializing weights

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        """

        self.w = torch.nn.Linear(x.shape[1], 1, bias=self.bias)
        self._best_state = self.w.state_dict()

    def fit(self,
            x: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 2000,
            lr: float = 1e-4,
            l1_constant: float = 0.,
            l2_constant: float = 0.,
            show_epoch: int = 0,
            print_function: Callable = print) -> torch.nn.Module:
        """
        Returns trained model of Regression

        .. rubric:: Target function

        Training happens by minimizing loss function:

        .. math::

            \\mathcal{L}(w) = \\lambda_{1} \\Vert w \\Vert_{1} + \\lambda_{2} \\Vert w \\Vert_{2} +
            \\frac{1}{n}\\sum_{i = 1}^{n} (\\hat y_i - y_i)^2 \\longrightarrow \\min_{w}

        :math:`x_{i} \\in \\mathbb{R}^{1\\times m}, w \\in \\mathbb{R}^{m \\times 1}, y_{i} \\in \\mathbb{R}^{1}`

        :param x: training set
        :param y: target value
        :param epochs: max number of sgd implements
        :param lr: Adam optimizer learning rate
        :param show_epoch: amount of showing epochs
        :param l1_constant: parameter of l1 regularization
        :param l2_constant: parameter of l2 regularization
        :param print_function: a function that will print verbose
        :return: trained model
        """
        x = x.float()
        x = self.transform(x)
        y = y.float().flatten()

        return self._training(x, y, epochs, lr, l1_constant, l2_constant, show_epoch, print_function)

    def _training(self,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  epochs: int = 2000,
                  lr: float = 1e-4,
                  l1_constant: float = 0.,
                  l2_constant: float = 0.,
                  show_epoch: int = 0,
                  print_function: Callable = print) -> torch.nn.Module:
        """
        Gradient descent to minimize loss function

        :param x: training set
        :param y: target value
        :param epochs: max number of sgd implements
        :param lr: Adam optimizer learning rate
        :param show_epoch: amount of showing epochs
        :param l1_constant: parameter of l1 regularization
        :param l2_constant: parameter of l2 regularization
        :param print_function: a function that will print verbose
        :return:
        """
        self.forward(x)  # model initializing

        # set starting point from by analytical formula
        x_t = PolyTransform(1).fit_transform(x)

        try:
            b = torch.linalg.inv(x_t.T @ x_t) @ x_t.T @ y
            _w = list(self.w.parameters())
            _w[0].data = b[1:].reshape(1, -1)
            _w[1].data = b[0].reshape(1, -1)

        except torch.linalg.LinAlgError:
            pass

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_constant)
        print_epochs = np.unique(np.linspace(1, epochs, min(epochs, show_epoch), dtype=int))

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            output = criterion(self._forward(x).flatten(), y.flatten())
            if output.item() < self._best_loss:
                self._best_state = self.state_dict()

            if l1_constant > 0:
                for layer in self.parameters():
                    output += l1_constant * layer.data.sum()

            output.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch + 1 in print_epochs:
                    print_function(f'Epoch: {epoch: 5d} | MSE: {output.item(): 0.5f}')

            if output.item() < 1e-10:
                if show_epoch > 0:
                    print_function(f'Epoch: {epoch: 5d} | MSE < 1e-10 | Early stop')
        else:
            self.load_state_dict(self._best_state)

        return self


class LinearRegression(BaseRegressionModel):
    """
    Model:

    .. math::

        \\hat y(x) = w_0 + w_1 \\cdot x_1 + w_2 \\cdot x_2 + \\dots + w_m \\cdot x_m

    """

    def __init__(self, bias: bool = True) -> None:
        super(LinearRegression, self).__init__()
        self.bias = bias


class PolynomialRegression(BaseRegressionModel):
    """

    Polynomial regression model:

    .. math::
        \\hat y(x) = \\sum_{\\alpha_1 + \\dots + \\alpha_m \\leq k} w_i \\cdot x_1 ^ {\\alpha_1} \\circ
        x_2 ^ {\\alpha_2} \\cdot \\dots \\circ x_m ^ {\\alpha_m}

    :math:`\\alpha_i \\in \\mathbb{Z}_+, x_i - i` column from :math:`x` matrix

    :math:`x_i, y, \\hat y \\in \\mathbb{R}^{n \\times 1}, \\circ` - hadamard product (like np.array * np.array)

    """

    def __init__(self, degree: int) -> None:
        """
        :param degree: degree of polynomial regression
        """
        super().__init__()
        assert degree > 0, 'degree must be positive'
        self.degree = int(degree)
        self.transformer = PolyTransform(self.degree)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns poly-transformed data

        :param x: torch tensor
        :return: transformed tensor
        """
        if not self.transformer.is_fitted:
            self.transformer.fit(x)

        return self.transformer.transform(x)[:, 1:]

    def __call__(self, x: torch.Tensor, transformed: bool = False) -> torch.Tensor:
        """
        Returns x_pf @ w , w is the weights for each parameter, x_pf is poly transformed matrix

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        :param transformed: flag of transformed x
        :return: regression value
        """
        return self.forward(x, transformed)


class ExponentialRegression(BaseRegressionModel):
    """

    Exponential regression

    .. math::

        \\hat y_i = \\exp(w_0 + w_1 \\cdot x_1 + w_2 \\cdot x_2 + \\dots + w_m \\cdot x_m)

    """

    def __init__(self):
        super(ExponentialRegression, self).__init__()

    def forward(self, x: torch.Tensor, transformed: bool = True) -> torch.Tensor:
        """
        Returns exponential regression function

        .. math::
            \\hat y = \\exp (x \\cdot w + b)

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        :param transformed: flag of transformed x
        :return: regression value
        """
        if self.w is None:
            self.init_weights(x)

        if not transformed:
            x = self.transform(x)

        return torch.exp(self.w(x))

    def fit(self,
            x: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 5000,
            lr: float = 1e-4,
            l1_constant: float = 0.,
            l2_constant: float = 0.,
            show_epoch: int = 0,
            print_function: Callable = print) -> torch.nn.Module:
        """
        Returns trained model of exponential Regression

        :param x: training set
        :param y: target value
        :param epochs: max number of sgd implements
        :param lr: Adam optimizer learning rate
        :param show_epoch: amount of showing epochs
        :param l1_constant: parameter of l1 regularization
        :param l2_constant: parameter of l2 regularization
        :param print_function: a function that will print verbose
        :return: trained model
        """
        x = x.float()
        assert (y > 0).all(), 'y must be positive'
        y = torch.log(y.float().flatten())

        self._forward(x)  # model initializing

        return self._training(x, y, epochs, lr, l1_constant, l2_constant, show_epoch)


class PolyTransform:
    """
    Makes polynomial features of degree :math:`k`

    We have matrix :math:`x \\in \\mathbb{R}^{n \\times m}` with m features and n observations.
    Let :math:`c_1` -- first column of :math:`x`, :math:`c_2` -- second column of :math:`x` etc.
    And we make all features like:

    .. math::

        c_1^{\\alpha_1} \\circ c_2^{\\alpha_2} \\circ \\dots \\circ c_m^{\\alpha_m}

    subject to :math:`\\alpha_i \\in \\mathbb{Z}_+` and :math:`\\sum_{i = 1}^{m} \\alpha_i \\leq k`

    :ivar feature_names: a list with the names of objects and degrees in order, as in the matrix
    :ivar columns: indexes of columns after fitting
    :ivar number_features: number of features after transform

    """

    def __init__(self, degree: int = 2) -> None:
        """
        :param degree: max degree of polynomial features
        """
        self.degree = degree
        self.feature_names = []
        self.columns = []
        self.number_features = None
        self.is_fitted = False

    def fit(self, x: torch.Tensor, columns: list[str] | None = None) -> PolyTransform:
        """
        Fitting of transformer. After fitting appears feature_names, columns, number_features

        :param x: matrix with m columns
        :param columns: string names of each column
        :return: self instance
        """

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
                self.feature_names.append(tuple(map(lambda _x: 'col_' + str(_x), names)))

        feat = ['1']

        # make feature names with
        for i in range(1, len(self.feature_names)):
            val, cnt = np.unique(self.feature_names[i], return_counts=True)
            val = list(map(str, val))
            cnt = list(map(str, cnt))

            feat.append(' '.join(map(lambda _x: '^'.join(_x), zip(val, cnt))))

        self.feature_names = feat
        self.is_fitted = True
        self.number_features = len(self.feature_names)

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns transformed input x by the indexes from self columns list

        :param x: Tensor with m columns like in fitting
        :return: transformed tensor
        """
        transformed = [torch.ones(x.shape[0], 1)]

        for cols in self.columns[1:]:
            transformed.append(x[:, cols].prod(dim=1, keepdim=True))

        return torch.concat(transformed, dim=1)

    def fit_transform(self, x: torch.tensor) -> torch.Tensor:
        """
        Returns transformed x. Call fit and after transform

        :param x: Tensor with m columns like in fitting
        :return: transformed tensor
        """
        self.fit(x)
        return self.transform(x)


if __name__ == '__main__':
    from metrics import r2_score

    torch.random.manual_seed(7)
    model = LinearRegression()
    _x = torch.rand(1000, 4)
    _y = _x @ torch.randint(0, 10, (4, 1)).float() + 5

    model.fit(_x, _y)

    print(r2_score(model(_x).flatten(), _y.flatten()))

    model2 = PolynomialRegression(3)
    model2.fit(_x, _y)

    print(r2_score(model2(_x).flatten(), _y.flatten()))

    model3 = ExponentialRegression()
    _y = torch.exp(_y)

    model3.fit(_x, _y)
    print(r2_score(model3(_x).flatten(), _y.flatten()))
