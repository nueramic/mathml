from __future__ import annotations

import sys
from typing import Dict, Union, Optional

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
import numpy as np
import torch


def check_tensors(*args: torch.Tensor) -> None:
    """
    Checks if arrays is 1d or torch.Tensor

    :param args: tensors
    :return:
    """
    for arr in args:
        if arr is not None:
            assert isinstance(arr, torch.Tensor), 'arrays must be torch Tensor'
            assert len(arr.shape) == 1, 'arrays must me 1-d'


def tpr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Return True Positive Rate. TPR = TP / P = TP / (TP + FN). Alias is Recall

    .. note:: if P == 0, then TPR = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    check_tensors(y_true, y_pred)
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
    check_tensors(y_true, y_pred)
    fp = ((y_true != y_pred) & (y_true == 0)).sum()
    n = (y_true == 0).sum()
    return float(fp / max(n, 1))


def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Return Positive Predictive Value . PPV = TP / (TP + FP)

    .. note:: if TP + FN == 0, then PPV = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    check_tensors(y_true, y_pred)
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
    check_tensors(y_true, y_pred)
    return float((y_true == y_pred).sum() / y_true.shape[0])


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
    check_tensors(y_true, y_pred)
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

    for threshold in np.arange(0, 1 + step_size, step_size):
        y_prob = model(x).flatten()
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

    thresholds = np.sort(np.unique(y_prob))[::-1]
    if n_thresholds is not None:
        thresholds = thresholds[np.linspace(0, len(thresholds) - 1, n_thresholds, dtype=int)]

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
    for i in range(len(fpr_array) - 1):  # Integrating by Trapezoidal rule
        auc += (tpr_array[i] + tpr_array[i + 1]) * (fpr_array[i + 1] - fpr_array[i]) / 2
    return float(auc)


def binary_classification_report(y_true: torch.Tensor,
                                 y_pred: torch.Tensor,
                                 y_prob: Optional[torch.Tensor] = None,
                                 ) -> dict:
    """
    Returns dict with recall, precision, accuracy, f1, auc roc scores
    best threshold

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :param y_prob: array of probabilities of confidence of belonging to the 1st class
    :return: float value of area under roc-curve
    """
    check_tensors(y_true, y_pred, y_prob)

    return {
        'recall': recall(y_true, y_pred),
        'precision': precision(y_true, y_pred),
        'f1': f_score(y_true, y_pred),
        'auc_roc': None if y_prob is None else auc_roc(y_true, y_prob)
    }


if __name__ == '__main__':
    yt = torch.randint(0, 2, (10_000,))
    yp = torch.rand((10_000,))
    print(auc_roc(yt, yp))
    print(binary_classification_report(yt, yt, yp))
