from __future__ import annotations

import sys
from typing import Callable

import numpy as np
import torch

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from .metrics import binary_classification_report, best_threshold, f_score


class BaseClassification(torch.nn.Module):

    def __init__(self):
        super(BaseClassification, self).__init__()

    def metrics_tab(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Returns metrics dict with recall, precision, accuracy, f1, auc roc scores

        :param x: training set
        :param y: target value
        :param: dict with recall, precision, accuracy, f1, auc roc scores
        :return: dict with 5 metrics
        """
        y_prob: torch.Tensor = self.forward(x)
        try:
            y_pred: torch.Tensor = (y_prob > self.best_threshold) * 1
        except AttributeError:
            y_pred = self.predict(x)
            y_prob: None = None

        return binary_classification_report(y, y_pred, y_prob)


class LogisticRegression(BaseClassification):
    """
    Binary classification model

    Let :math:`x \\in \\mathbb{R}^{n \\times m}, \\ w \\in \\mathbb{R}^{m \\times 1}, \\ I = [1]_{n \\times 1}`,
    :math:`x_i` -- is a row and :math:`x_i \\in \\mathbb{R}^{1 \\times m}`

    Model:

    .. math::

        \\mathbb{P}(y_i = 1 | w) = \\frac{1}{1 + \\exp (x_i \\cdot w + b)}


    """

    def __init__(self, kernel: Literal['linear', 'perceptron'] = 'linear'):
        """
        :param kernel: 'linear' or 'perceptron'. linear - basic logistic regression, perceptron - nn with 2
                       hidden layer with dim1 = 1024, dim2 = 512
        """
        super(LogisticRegression, self).__init__()

        self.sigmoid = torch.nn.Sigmoid()
        self.weights = None
        self.kernel = kernel
        if kernel not in ['linear', 'perceptron']:
            raise TypeError('Invalid kernel. Choose "linear" or "perceptron"')

        self.best_threshold = 0.5

    def init_weights(self, x: torch.Tensor):
        """
        Initialization weights

        :param x: input torch tensor
        """
        if self.kernel == 'linear':
            self.weights = torch.nn.Linear(x.shape[1], 1)
        elif self.kernel == 'perceptron':
            self.weights = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns confidence probabilities of first class

        :param x: training set
        :return: probabilities
        """

        if self.weights is None:
            self.init_weights(x)

        x = x.float()
        return self.sigmoid(self.weights(x))

    def predict(self, x):
        """
        Returns binary class 0 or 1 instead of probabilities

        :param x: some tensor with shape[1] = n_features
        :return:
        """
        y_prob: torch.Tensor = self.forward(x).flatten()
        y_pred: torch.Tensor = (y_prob > self.best_threshold) * 1
        return y_pred

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs=1000, l1_lambda: float = 0,
            show_epoch: int = 0, print_function: Callable = print) -> torch.nn.Module:
        """
        Returns trained model Logistic Regression

        :param x: training set
        :param y: target value
        :param epochs: max number of sgd implements
        :param l1_lambda: l1 regularization weight
        :param show_epoch: amount of showing epochs
        :param print_function: print or streamlit.write
        :return: trained model
        """
        x = x.float()
        y = y.float()

        self.forward(x)

        print_epochs = np.unique(np.linspace(1, epochs, min(epochs, show_epoch), dtype=int))

        optimizer = torch.optim.Adam(self.parameters())
        loss = torch.nn.BCELoss()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            output = loss(self.forward(x).flatten(), y.flatten())
            if l1_lambda > 0:
                for layer in self.parameters():
                    output += l1_lambda * layer.data.sum()

            output.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch in print_epochs:
                    print_function(f'Epoch: {epoch: 5d} | CrossEntropyLoss: {output.item(): 0.5f}')

        self.best_threshold = best_threshold(x, y, self, metric='f1')

        return self


class LogisticRegressionRBF(BaseClassification):
    """

    This is a logistic regression, but before we make a basic linear prediction and apply the sigmoid, we transfer x to
    another space using radial basis functions. The dimension of this space depends on the basis matrix x (x_basis) [1]_

    .. rubric:: Radial basis functions

    #. gaussian :math:`\\displaystyle \\varphi (x, x_b)=e^{-\\Vert x - x_b \\Vert^2}`

    #. linear :math:`\\varphi (x, x_b) = \\Vert x - x_b \\Vert`

    #. multiquadratic :math:`\\displaystyle \\varphi (x, x_b)  = \\sqrt{1 + \\Vert x - x_b \\Vert^2}`

    .. rubric:: References

    .. [1] https://en.wikipedia.org/wiki/Radial_basis_function

    """

    def __init__(self, x_basis: torch.Tensor, rbf: Literal['linear', 'gaussian', 'multiquadratic'] = 'gaussian'):
        """

        :param x_basis: centers of basis functions
        :param rbf: type of rbf function. Available: ['linear', 'gaussian']
        """

        super(LogisticRegressionRBF, self).__init__()

        self.w = torch.nn.Linear(x_basis.shape[0], 1)
        self.rbf = rbf
        self.x_basis = x_basis
        self.sigmoid = torch.nn.Sigmoid()
        self.best_threshold = 0.5

    def forward(self, x: torch.Tensor = None, phi_matrix: torch.Tensor = None) -> torch.Tensor:
        """
        Returns a "probability" (confidence) of class 1

        :param x: 2D array
        :param phi_matrix: 2D array
        :return: 1D array
        """
        if phi_matrix is None:
            phi_matrix = self.make_phi_matrix(x)

        return self.sigmoid(self.w(phi_matrix))

    def make_phi_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns n x k array with calculated phi(x_i, x_basis_j). n is number of observation from x (x.shape[0])
        k is number of basis from initialization.

        .. math::
            
            \\begin{bmatrix} \\varphi(x_1, x^\\text{basis}_1) & \\varphi(x_1, x^\\text{basis}_2) & \\dots &
            \\varphi(x_1, x^\\text{basis}_k) \\\\ \\varphi(x_2, x^\\text{basis}_1) & \\varphi(x_2, x^\\text{basis}_2) &
            \\dots &  \\varphi(x_2, x^\\text{basis}_k) \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\varphi(x_n, x^\\text{basis}_1) & \\varphi(x_n, x^\\text{basis}_2) & \\dots &
            \\varphi(x_n, x^\\text{basis}_k) \\ \\end{bmatrix}

        :param x: Array k x m dimensional. k different x_i and m features
        """
        x = x.float()
        n = self.x_basis.shape[0]
        k = x.shape[0]

        repeated_input_x = torch.tile(x, (n, 1))
        repeated_basis_x = torch.tile(self.x_basis, (1, k))
        repeated_basis_x = torch.reshape(repeated_basis_x, repeated_input_x.shape)

        phi = ((repeated_input_x - repeated_basis_x) ** 2).sum(dim=1)
        phi = torch.reshape(phi, (n, k)).T

        if self.rbf == 'linear':
            phi = phi ** 0.5
            phi = phi / phi.max()

        elif self.rbf == 'gaussian':
            phi = torch.exp(-phi)

        elif self.rbf == 'multiquadratic':
            phi = (1 + phi) ** 0.5
            phi = phi / phi.max()

        return phi.float()

    def predict(self, x):
        """
        Returns binary class 0 or 1 instead of -1; 1

        :param x: some tensor with shape[1] = n_features
        :return:
        """
        y_prob: torch.Tensor = self.forward(x).flatten()
        y_pred: torch.Tensor = (y_prob > self.best_threshold) * 1
        return y_pred

    def fit(self,
            x: torch.Tensor,
            y: torch.Tensor,
            epochs=100,
            l1_lambda: float = 0,
            show_epoch: int = 0,
            print_function: Callable = print) -> torch.nn.Module:
        """
        Returns trained model Logistic Regression with RBF

        :param x: training set
        :param y: target value
        :param epochs: max number of sgd implements
        :param l1_lambda: l1 regularization weight
        :param show_epoch: amount of showing epochs
        :param print_function: e.g. print or streamlit.write
        :return: trained model
        """
        x = x.float()
        y = y.float()

        print_epochs = np.unique(np.linspace(1, epochs, min(epochs, show_epoch), dtype=int))

        phi_matrix = self.make_phi_matrix(x)
        optimizer = torch.optim.Adam(self.parameters())
        loss = torch.nn.BCELoss()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            output = loss(self.forward(x, phi_matrix).flatten(), y.flatten())
            if l1_lambda > 0.:
                for layer in self.parameters():
                    output += l1_lambda * layer.data.sum()

            output.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch in print_epochs:
                    print_function(f'Epoch: {epoch: 5d} | CrossEntropyLoss: {output.item(): 0.5f}')

        self.best_threshold = best_threshold(x, y, self, metric='f1')

        return self


class SVM(BaseClassification):
    """

    Binary classification model. Method predict: SVM.predict(x) --> original names

    Mathematical model:

    .. math::

        \\hat y = \\operatorname{sign}(x \\cdot w - b \\cdot I)


    :math:`x \\in \\mathbb{R}^{n \\times m}, \\ w \\in \\mathbb{R}^{m \\times 1}, \\ I = [1]_{n \\times 1}`


    And search of best :math:`w, b` calculates by minimization of Hinge loss

    .. math::

        {\\displaystyle \\lambda \\lVert \\mathbf {w} \\rVert ^{2}+\\left[{\\frac {1}{n}}\\sum _{i=1}^{n}\\max
        \\left(0,1-y_{i}(x_i \\cdot w - b)\\right)\\right] \\longrightarrow \\min }


    or PEGASOS algorithm

    :ivar scale: for the best training and prediction, the model will standard normalize the input x data.
                 The first time you call model, std and mean will be saved and in the future use the parameters
                 for scaling. x = (x is the average value) / std
    :ivar weights: parameters of model. Initialize after first calling

    """

    def __init__(self):
        """
        Initialization of SVM
        """
        super(SVM, self).__init__()
        self.scale = None
        self.weights = None
        self.class_names = {0: -1, 1: 1}  # the null class has the name -1, and the first class has the name 1

    def init_weights(self, x: torch.Tensor):
        """
        Initialization weights

        :param x: input torch tensor
        """
        self.weights = torch.nn.Linear(x.shape[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns x @ w + b

        .. math::
            f(x) = w_0 + w_1 \\cdot x_1 + w_2 \\cdot x_2 + \\dots + w_m \\cdot x_m

        :param x: input observations, tensor n x m (n is the number of observations that have m parameters)
        :return: regression value (yes, no classification, for binary classes call predict)
        """
        x = x.float()

        if self.weights is None:
            self.init_weights(x)

        return self.weights(x)

    def scaler(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the scaled value of x. Standard x scaling and storing settings

        :param x: torch.Tensor
        :return:
        """
        if self.scale is None:
            self.scale = (x.mean(), x.flatten().std())

        return (x - self.scale[0]) / self.scale[1]

    def fit(self,
            x: torch.Tensor,
            y: torch.Tensor,
            method: Literal['pegasos', 'sgd'] = 'sgd',
            epochs=100,
            lambda_reg: float = 0.1,
            show_epoch: int = 0,
            print_function: Callable = print):
        """
        Returns trained model SVM

        :param x: training set
        :param y: target value. binary classes
        :param method: optimization method. Available PEGASOS or sgd
        :param epochs: max number of sgd and pegasos steps 
        :param lambda_reg: l2 regularization weight
        :param show_epoch: amount of showing epochs
        :param print_function: print or streamlit.write
        :return: trained model
        """
        y = y.flatten().float().round().int()
        uniq_y = torch.unique(y)
        assert torch.unique(y).shape[0] <= 2, 'binary classification support only'
        if 1 in uniq_y:
            self.class_names = {int(uniq_y[uniq_y != 1][0]): -1, int(uniq_y[uniq_y == 1][0]): 1}
        else:
            self.class_names = dict(zip(map(int, uniq_y), (-1, 1)))

        y_replaced = torch.zeros_like(y)
        for i in range(y_replaced.shape[0]):
            y_replaced[i] = self.class_names[int(y[i])]

        y = y_replaced.float()

        if method == 'pegasos':
            return self._fit_pegasos(x, y, epochs, lambda_reg, show_epoch, print_function)
        else:
            return self._fit_sgd(x, y, epochs, lambda_reg, show_epoch, print_function)

    def _fit_sgd(self, x: torch.Tensor, y: torch.Tensor, epochs=500, l2_lambda: float = 0,
                 show_epoch: int = 0, print_function: Callable = print):
        """
        Returns trained model SVM

        :param x: training set
        :param y: target value
        :param epochs: max number of sgd implements
        :param l2_lambda: l2 regularization weight
        :param show_epoch: amount of showing epochs
        :param print_function: print or streamlit.write
        :return: trained model
        """

        x = self.scaler(x.float())
        self.forward(x)

        print_epochs = np.unique(np.linspace(1, epochs, min(epochs, show_epoch), dtype=int))

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=l2_lambda)
        loss = torch.nn.MarginRankingLoss(margin=1)  # hinge loss if x2 = 0 and margin = 1

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            output = loss(self.forward(x).flatten(), torch.tensor([0]), y.flatten())
            output.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch in print_epochs:
                    print_function(f'Epoch: {epoch: 5d} | HingeLoss: {output.item(): 0.5f}')

        return self

    def _fit_pegasos(self, x: torch.Tensor, y: torch.Tensor, epochs=20, lambda_reg: float = 0.95,
                     show_epoch: int = 0, print_function: Callable = print) -> torch.nn.Module:
        """
        Returns trained model SVM [2]_

        :param x: training set
        :param y: target value
        :param epochs: max number of sgd implements
        :param lambda_reg: regularization parameter
        :param show_epoch: amount of showing epochs
        :param print_function: print or streamlit.write
        :return: trained model

        .. rubric:: References

        .. [2] Pegasos: Primal Estimated sub-GrAdient SOlver for SVM. Shai Shalev-Shwartz; Yoram Singer; Nathan Srebro;
               Andrew Cotter

        """
        x = self.scaler(x.float())
        self.forward(x)

        print_epochs = np.unique(np.linspace(1, epochs, min(epochs, show_epoch), dtype=int))
        weights = torch.zeros(x.shape[1])
        bias = torch.zeros(1)
        t = 0
        for epoch in range(1, epochs + 1):
            t += 1
            eta = 1 / (lambda_reg * t)
            for j in torch.randint(0, x.shape[0], (min(100, x.shape[0]),)):

                if y[j] * (weights @ x[j] + bias) < 1:
                    weights = (1 - eta * lambda_reg) * weights + eta * y[j] * x[j]
                    bias = (1 - eta * lambda_reg) * bias + eta * y[j] * 1

                else:
                    weights = (1 - eta * lambda_reg) * weights
                    bias = (1 - eta * lambda_reg) * bias

                weights = min(1, 1 / lambda_reg ** 0.5 / torch.concat([weights, bias]).norm(2)) * weights

                with torch.no_grad():
                    if epoch in print_epochs:
                        print_function(f'Epoch: {epoch: 5d} | F1 score: {f_score(y, self.predict(x)): 0.5f}')

        params = list(self.weights.parameters())
        params[0].data = weights.reshape(1, -1)
        params[1].data = bias

        return self

    def predict(self, x):
        """
        Returns binary class from the first call, in training or just call

        :param x: some tensor with shape[1] = n_features
        :return:
        """
        x = self.scaler(x).float()
        y = torch.zeros(x.shape[0])
        reversed_names = {**{0: 1}, **{v: u for u, v in self.class_names.items()}}
        for i, xi in enumerate(self.weights(x).sign().flatten().int()):
            y[i] = reversed_names[int(xi)]

        return y


if __name__ == '__main__':
    # from sklearn.datasets import make_blobs
    # from metrics import accuracy

    # torch.random.manual_seed(7)
    # _x, _y = make_blobs(1000, centers=2, random_state=8)
    # _x, _y = torch.tensor(_x), torch.tensor(_y)
    # m = SVM().fit(_x, _y)
    # print(accuracy(_y, m.predict(_x)))
    # m = SVM().fit(_x, _y, method='pegasos')
    # print(accuracy(_y, m.predict(_x)))
    # m = LogisticRegression().fit(_x, _y)
    # print(accuracy(_y, m.predict(_x)))
    # m = LogisticRegressionRBF(_x[:100]).fit(_x, _y)
    # print(accuracy(_y, m.predict(_x)))

    from sklearn.datasets import make_moons

    _x, _y = make_moons(10_000, noise=.1, random_state=84)
    _x, _y = torch.tensor(_x), torch.tensor(_y)

    logistic_model_rbf = LogisticRegressionRBF(_x[:1000]).fit(_x, _y, show_epoch=10)

    print(logistic_model_rbf.metrics_tab(_x, _y))
