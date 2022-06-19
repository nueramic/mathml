from __future__ import annotations

from typing import Optional

import plotly.express as px
import plotly.graph_objs as go
import torch
from sklearn.manifold import TSNE

from .one_animation import standard_layout
from ..ml.classification import LogisticRegressionRBF
from ..ml.metrics import roc_curve_plot


def gen_classification_plot(x_tensor: torch.Tensor,
                            y_true: torch.Tensor,
                            model: Optional[torch.nn.Module] = None,
                            threshold: float = 0.5,
                            cnt_points: int = 1000,
                            k: float = 0.1,
                            title: Optional[str] = None,
                            epsilon: float = 1e-4,
                            insert_na: bool = False) -> go.Figure:
    """
    Returns a graph with a distribution and an optional line. If dim(x) = 2, then you can get model. If dim(x) > 2,
    then returns graph of TSNE from sklearn.manifold with default settings. dim(x) is not support

    .. note::

        if model os linear and have one layer, simple activation function, then visualization will faster

    .. warning::

        if the model is heavy, then you should reduce cnt_points, but the probability of missing points is higher,
        and the visualization will be rather incorrect. You can increase the gap by increasing the epsilon.

    :param x_tensor: training tensor
    :param y_true: target tensor. array with true values of binary classification
    :param model: some model that returns a torch tensor with class 1 probabilities using the call: model(x)
    :param threshold: if model(xi) >= threshold, then yi = 1
    :param cnt_points: number of points on each of the two axes when dim(x) = 2
    :param k: constant for draw on section: [x.min() - (x.max() - x.min()) * k, x.max() + (x.max() - x.min()) * k]
    :param title: title of plots
    :param epsilon: contour line points: :math:`\\{x\\in \\mathbb{R}^2 \\, | \\,
                \\text{threshold} - \\text{epsilon} \\le \\text{model}(x) \\le \\text{threshold} + \\text{epsilon}\\}`
    :param insert_na: na insertion flag when two points too far away
    :return: scatter plot go.Figure

    .. code-block:: python3

        >>> from sklearn.datasets import make_moons
        >>> torch.random.manual_seed(7)
        >>> x, y = make_moons(1000, noise=0.15, random_state=7)
        >>> x, y = torch.tensor(x), torch.tensor(y)

        >>> lr_rbf = LogisticRegressionRBF(x[:50])
        >>> lr_rbf.fit(x, y, epochs=5000)

        >>> lr_rbf.metrics_tab(x, y)

    .. code-block:: python3

        {'recall': 0.9980000257492065,
         'precision': 0.9842209219932556,
         'f1': 0.9910625822119956,
         'auc_roc': 0.9995800006320514}

    .. code-block:: python3

        >>> gen_classification_plot(x, y, model, threshold=0.5, epsilon=0.001)

    """
    colors = list(map(lambda e: str(int(e)), y_true))

    if x_tensor.shape[1] < 2:
        raise AssertionError('x.shape[1] must be >= 2')

    elif x_tensor.shape[1] == 2:
        title = '<b>Initial Distribution</b>' if title is None else title
        fig = px.scatter(x=x_tensor[:, 0], y=x_tensor[:, 1], title=title, color=colors)

        if model is not None:
            dx = x_tensor[:, 0].max() - x_tensor[:, 0].min()
            dy = x_tensor[:, 1].max() - x_tensor[:, 1].min()

            x1 = torch.tensor([x_tensor[:, 0].min() - dx * k,
                               x_tensor[:, 1].min() - dy * k])

            x2 = torch.tensor([x_tensor[:, 0].max() + dx * k,
                               x_tensor[:, 1].max() + dy * k])

            flag, grid = _make_line_linear((x1[0] + dx * k / 2, x2[0] - dx * k / 2), model, threshold)

            if not flag:
                grid = _make_line(x1, x2, model, threshold, cnt_points, epsilon, insert_na)

            line_x, line_y = grid.detach().cpu().T

            fig.add_scatter(x=line_x, y=line_y, name='sep plane', mode='lines')

    else:
        title = '<b>TSNE of Distribution</b>' if title is None else title
        tsne_x = TSNE().fit_transform(x_tensor)
        fig = px.scatter(x=tsne_x[:, 0], y=tsne_x[:, 1], title=title, color=colors)

    fig.update_layout(**standard_layout)

    return fig


def _sort_points(line: torch.Tensor, epsilon: float = 1e-3, metric: int = 2, insert_na: bool = True) -> torch.Tensor:
    """
    Returns tensor sorted by closeness between each other. if || lines[i] - closest{lines[j]} ||_metric > epsilon
    insert [nan, nan]

    :param line: tensor n x 2
    :param epsilon: maximum closeness
    :param metric: l1, l2, or some other metric
    :param insert_na: na insertion flag
    :return: sorted tensor line with probably added nan values
    """

    copy_line = [line[0, :]]
    mask = torch.tile(torch.tensor([True]), line.shape[:1])
    mask[0] = False
    for i in range(line.shape[0] - 1):
        distances = torch.norm(line - copy_line[-1], p=metric, dim=1)
        distances[torch.logical_not(mask)] = torch.inf

        min_d, argmin_d = distances.min(), distances.argmin()
        if min_d <= epsilon ** 0.3 or insert_na is False:
            copy_line.append(line[[argmin_d]])
        else:
            copy_line.append(torch.tensor([torch.nan, torch.nan]))
            copy_line.append(line[[argmin_d]])

        mask[argmin_d] = False

    line = torch.zeros(len(copy_line), 2)
    for i in range(line.shape[0]):
        line[i, :] = copy_line[i]
    return line


roc_curve_plot = roc_curve_plot


def _make_line(x1: torch.Tensor, x2: torch.Tensor, model: torch.nn.Module, threshold: float = 0.5,
               cnt_points: int = 25, epsilon: float = 1e-3, insert_na: bool = True) -> torch.Tensor:
    """
    Returns x in [x1, x2] : threshold - epsilon <= model(x) <= threshold + epsilon

    :param x1: 2-dim tensor start
    :param x2: 2-dim tensor end
    :param model: some model that returns a torch tensor with class 1 probabilities using the call: model(x)
    :param threshold: if model(xi) >= threshold, then yi = 1
    :param cnt_points: number of points on each of the two axes
    :param epsilon: contour line points: :math:`\\{x\\in \\mathbb{R}^2 \\, | \\,
                \\text{threshold} - \\text{epsilon} \\le \\text{model}(x) \\le \\text{threshold} + \\text{epsilon}\\}`
    :param insert_na: na insertion flag
    :return: scatter plot go.Figure
    """
    if torch.isnan(x1[0]) or torch.isnan(x1[1]) or torch.isnan(x2[0]) or torch.isnan(x2[1]):
        return torch.tensor([[torch.nan, torch.nan]])

    lin_settings_1 = (min(x1[0], x2[0]), max(x1[0], x2[0]), cnt_points)
    lin_settings_2 = (min(x1[1], x2[1]), max(x1[1], x2[1]), cnt_points)

    grid = torch.cartesian_prod(torch.linspace(*lin_settings_1), torch.linspace(*lin_settings_2))

    with torch.no_grad():
        grid_pred = model(grid)

    mask = (threshold - epsilon <= grid_pred) & (grid_pred <= threshold + epsilon)
    if sum(mask) > 0:
        if sum(mask) > 1000:
            grid = grid[mask.flatten(), :]
            grid = grid[torch.linspace(0, grid.shape[0], 1000, dtype=torch.int64), :]
        else:
            grid = grid[mask.flatten(), :]
        grid = _sort_points(grid, epsilon=epsilon, insert_na=insert_na)
    else:
        grid = torch.tensor([torch.nan, torch.nan])
    return grid


def _make_line_linear(bounds_x: tuple[float, float],
                      model: torch.nn.Module,
                      threshold: float = 0.5) -> [bool, tuple | None]:
    """
    Returns for a linear model or a linear model with a sigmoid activation line on the plane

    :param bounds_x: bounds for x. tuple with two numbers
    :param model: linear model. e.g. SVM, Sigmoid
    :param threshold:
    :return:
    """
    # Check linear model
    if len(list(model.parameters())) > 2:
        return False, None

    try:
        w, b = model.parameters()
        w, b = w.flatten(), b.flatten()
    except Exception as e:
        print(e, 'non-linear model. is used basic _make_line')
        return False, None

    if len(w) > 2:
        return False, None

    x = torch.linspace(*bounds_x, 100)

    try:
        if hasattr(model, 'Sigmoid'):
            y = (torch.log(torch.tensor(threshold / (1 - threshold))) - b - w[0] * x) / w[1]

        else:
            y = (threshold - b - w[0] * x) / w[1]

    except Exception as e:
        print(e, 'non-linear model. is used basic _make_line')
        return False, None

    return True, torch.stack([x, y]).T
