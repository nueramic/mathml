from __future__ import annotations

import sys
from typing import Dict, Union, Optional

import plotly.express as px
import plotly.graph_objects as go

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
import torch


def check_tensors(*tensors: torch.Tensor) -> [torch.Tensor]:
    """
    Returns flattened tensors and checks if the arrays have the same size and flare.Tensor

    :param tensors: tensors
    :return: tensors
    """
    shapes = []
    output = []
    for arr in tensors:
        if arr is not None:
            assert isinstance(arr, torch.Tensor), 'arrays must be torch Tensor'
            arr = arr.flatten()
            shapes.append(int(arr.shape[0]))

        output.append(arr)

    assert len(torch.unique(torch.tensor(shapes))) == 1, 'Tensors must have same sizes'

    return output


def tpr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Return True Positive Rate. TPR = TP / P = TP / (TP + FN). Alias is Recall

    .. note:: if P == 0, then TPR = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    tp = ((y_true == y_pred) & (y_true == 1)).sum()
    p = (y_true == 1).sum()
    return float(tp / max(p, 1))


def fpr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Return False Positive Rate. FPR = FP / N = FP / (FP + TN).

    .. note:: if N == 0, then FPR = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    fp = ((y_true != y_pred) & (y_true == 0)).sum()
    n = (y_true == 0).sum()
    return float(fp / max(n, 1))


def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Return Positive Predictive Value . PPV = TP / (TP + FP)

    .. note:: if TP + FP == 0, then PPV = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    tp = ((y_true == y_pred) & (y_true == 1)).sum()
    fp = ((y_true != y_pred) & (y_true == 0)).sum()
    return float(tp / max(tp + fp, 1))


recall = tpr


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Return accuracy. ACC = (TP + TN) / (P + N) = (TP + TN) / (TP + FP + TN + FN)

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    return float(torch.eq(y_true, y_pred).sum() / y_true.shape[0])


def f_score(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float = 1) -> float:
    """
    Return F_score. https://en.wikipedia.org/wiki/F-score

    .. math::
        F_\\beta = (1 + \\beta^2) \\cdot \\frac{\\mathrm{precision} \\cdot \\mathrm{recall}}
        {(\\beta^2 \\cdot \\mathrm{precision}) + \\mathrm{recall}}.

    .. note:: if beta ** 2 * _precision + _recall == 0, then f_score = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :param beta: is chosen such that recall is considered beta times as important as precision
    :return:
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    _precision = precision(y_true, y_pred)
    _recall = tpr(y_true, y_pred)

    numerator = (1 + beta ** 2) * (_precision * _recall)
    denominator = max(beta ** 2 * _precision + _recall, 1e-12)

    return float(numerator / denominator)


def best_threshold(x: torch.Tensor, y_true: torch.Tensor, model: torch.nn.Module,
                   metric: Literal['f1', 'by_roc'] = 'f1', step_size: float = 0.01):
    """
    Returns best threshold by metric by linear search

    :param x: training tensor
    :param y_true: target tensor. array with true values of binary classification
    :param model: some model that returns a torch tensor with class 1 probabilities using the call: model(x)
    :param metric: name of the target metric that we need to maximize. by_roc - difference between TPR and FPR
    :param step_size: step size of linear search
    :return:
    """
    y_true = y_true.flatten()
    metric = {'f1': f_score, 'by_roc': lambda y1, y2: tpr(y1, y2) - fpr(y1, y2)}[metric]
    best_t = 0
    best_metric = 0
    y_prob = model(x).flatten()

    for threshold in torch.arange(0, 1 + step_size, step_size):

        y_pred = (y_prob >= threshold) * 1
        metric_i = metric(y_true, y_pred)

        if metric_i > best_metric:
            best_metric = metric_i
            best_t = threshold

    return best_t


def roc_curve(y_true: torch.Tensor, y_prob: torch.Tensor, n_thresholds: Union[int, None] = None) -> Dict:
    """
    Return dict with points at TPR - FPR coordinates

    :param y_true: array with true values of binary classification
    :param y_prob: array of probabilities of confidence of belonging to the 1st class
    :param n_thresholds: if len(y_true) is too large, you can limit the number of threshold values
    :return: dict with values of TPR and FPR
    """
    tpr_array = []
    fpr_array = []
    check_tensors(y_true, y_prob)

    thresholds, _ = torch.sort(torch.unique(y_prob), descending=True)
    thresholds = torch.cat((torch.ones(1), thresholds, torch.zeros(1)))

    if n_thresholds is not None:
        thresholds = thresholds[torch.linspace(0, len(thresholds) - 1, n_thresholds, dtype=torch.long)]

    for threshold in thresholds:
        tpr_array.append(tpr(y_true, (y_prob >= threshold) * 1))
        fpr_array.append(fpr(y_true, (y_prob >= threshold) * 1))

    return {'TPR': tpr_array, 'FPR': fpr_array}


def auc_roc(y_true: torch.Tensor, y_prob: torch.Tensor, n_thresholds: int = 500) -> float:
    """
    Return area under curve ROC (AUC-ROC metric)

    :param y_true: array with true values of binary classification
    :param y_prob: array of probabilities of confidence of belonging to the 1st class
    :param n_thresholds: if len(y_true) is too large, you can limit the number of threshold values
    :return: float value of area under roc-curve
    """
    check_tensors(y_true, y_prob)

    tpr_array, fpr_array = roc_curve(y_true, y_prob, min(y_true.shape[0], n_thresholds)).values()
    auc = 0
    for i in range(len(fpr_array) - 1):  # Integrating
        auc += tpr_array[i + 1] * (fpr_array[i + 1] - fpr_array[i])
    return float(auc)


def binary_classification_report(y_true: torch.Tensor,
                                 y_pred: torch.Tensor,
                                 y_prob: Optional[torch.Tensor] = None,
                                 ) -> dict:
    """
    Returns dict with recall, precision, accuracy, f1, auc roc scores

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :param y_prob: array of probabilities of confidence of belonging to the 1st class
    :return: dict with 5 metrics
    """
    check_tensors(y_true, y_pred, y_prob)

    return {
        'recall': recall(y_true, y_pred),
        'precision': precision(y_true, y_pred),
        'accuracy': accuracy(y_true, y_pred),
        'f1': f_score(y_true, y_pred),
        'auc_roc': None if y_prob is None else auc_roc(y_true, y_prob)
    }


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Return R2 metric of regression

    .. math::

        \\mathbf{R}^{2} = 1 - \\frac{\\sum_{i=1}^{n}(\\hat y_i - y_i)^2}{\\sum_{i = 1}^{n}(\\hat y_i - \\overline y)^2}

    .. note::

        if std(y_true) = 0, then r2 = 0

    :param y_true: array with true values of regression
    :param y_pred: array with prediction values of regression
    :return: r2 metric in float number
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    y_true, y_pred = y_true.float(), y_pred.float()
    rss = (y_true - y_pred).norm(2) ** 2
    tss = (y_true - y_true.mean()).norm(2) ** 2
    if tss < 1e-12:
        return 0

    return float(1 - rss / tss)


def roc_curve_plot(y_true: torch.Tensor, y_prob: torch.Tensor, fill: bool = False) -> go.Figure:
    """
    Return figure with plotly.Figure ROC curve

    :param y_true: array with true values of binary classification
    :param y_prob: array of probabilities of confidence of belonging to the 1st class
    :param fill: flag for filling the area under the curve
    :return: go.Figure

    .. code-block:: python3

        >>> yt = torch.tensor([1, 1, 0, 0, 1, 0])
        >>> yp = torch.tensor([0.7, 0.6, 0.3, 0.5, 0.4, 0.4])
        >>> roc_curve_plot(yt, yp)

    """
    if fill:
        fig = px.area(roc_curve(y_true, y_prob, None if len(y_true) < 1000 else 1000), x='FPR', y='TPR',
                      title='<b>ROC curve</b>')
    else:
        fig = px.line(roc_curve(y_true, y_prob, None if len(y_true) < 1000 else 1000), x='FPR', y='TPR',
                      title='<b>ROC curve</b>')

    fig.update_layout(font={'size': 18}, autosize=False, width=700, height=600, xaxis={'range': [-0.05, 1.05]})
    fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line={'dash': 'dash'}, name='', showlegend=False)
    return fig


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Returns MSE

    .. math::

        \\mathbf{MSE}^{2} = \frac{1}{n}\\sum_{i=1}^{n}(\\hat y_i - y_i)^2}

    :param y_true: array with true values of regression
    :param y_pred: array with prediction values of regression
    :return: mse metric in float number
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    y_true, y_pred = y_true.float(), y_pred.float()
    _mse = ((y_true - y_pred) ** 2).mean()

    return float(_mse)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Returns MAE

    .. math::

        \\mathbf{MSE}^{2} = \frac{1}{n}\\sum_{i=1}^{n}|\\hat y_i - y_i|

    :param y_true: array with true values of regression
    :param y_pred: array with prediction values of regression
    :return: mae metric in float number
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    y_true, y_pred = y_true.float(), y_pred.float()
    _mae = (y_true - y_pred).abs().mean()

    return float(_mae)


def mape(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Returns MAPE

    .. math::

        \\displaystyle \\mathbf{MAPE}= \\frac{100}{n}}\\sum _{i=1}^{n}\\left|{\\frac{A_{i}-F_{i}}{A_{t}}}\\right|}

    .. note::

        All values in y_true that are less than 1e-10 in absolute value will be replaced by 1e-10

    :param y_true: array with true values of regression
    :param y_pred: array with prediction values of regression
    :return: mape metric in float number
    """
    y_true, y_pred = check_tensors(y_true, y_pred)
    y_true, y_pred = y_true.float(), y_pred.float()

    numerator = (y_true - y_pred).abs()

    y_true = y_true.abs()
    y_true[y_true < 1e-10] = 1e-10

    _mape = (numerator / y_true).mean() * 100

    return float(_mape)


def regression_report(y_true: torch.Tensor,
                      y_pred: torch.Tensor) -> dict:
    """
    Returns dict with recall, precision, accuracy, f1, auc roc scores

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return: dict with 4 metrics
    """
    check_tensors(y_true, y_pred)

    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mse': mse(y_true, y_pred),
        'mape': mape(y_true, y_pred),
    }
