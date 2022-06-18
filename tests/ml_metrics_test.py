import pytest
import torch

from src.nueramic_mathml.ml.metrics import binary_classification_report, precision, recall

binary_classification_report_tests = [
    (
        torch.tensor([1, 0, 1, 1]),  # true
        torch.tensor([1, 0, 0, 1]),  # predicted
        torch.tensor([0.9, 0.2, 0.4, 0.8]),  # probabilities
        {
            'recall': 2 / 3,
            'precision': 1,
            'f1': 0.8,
            'auc_roc': 1.0
        }  # actual report
    ),

    (
        torch.cat((torch.zeros(5_000), torch.ones(5_000))),
        torch.cat((torch.zeros(5_000), torch.ones(5_000))),
        torch.cat((torch.ones(5_000) * 0.3, torch.ones(5_000) * 0.7)),  # probabilities
        {
            'recall': 1,
            'precision': 1,
            'f1': 1,
            'auc_roc': 1.0
        }

    )
]


@pytest.mark.parametrize('y_true, y_pred, y_prob, expected', binary_classification_report_tests)
def test_bin_cr(y_true, y_pred, y_prob, expected):
    assert binary_classification_report(y_true, y_pred, y_prob) == pytest.approx(expected)


precision_tests = [
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 0, 0, 1]), 2 / 3, 1),  # basic test
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1, 1, 1]), 1, 3 / 4),
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 0, 0, 0]), 1 / 3, 1),
    (torch.tensor([1, 0, 1, 1]), torch.tensor([0, 1, 0, 0]), 0, 0),
    (torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 0, 1]), 3 / 4, 1),
    (torch.zeros(10_000), torch.ones(10_000), 0, 0)
]


@pytest.mark.parametrize('y_true, y_pred, expected_recall, expected_precision', precision_tests)
def test_precision(y_true, y_pred, expected_recall, expected_precision):
    assert precision(y_true, y_pred) == pytest.approx(expected_precision, rel=1e-5)


@pytest.mark.parametrize('y_true, y_pred, expected_recall, expected_precision', precision_tests)
def test_precision(y_true, y_pred, expected_recall, expected_precision):
    assert recall(y_true, y_pred) == pytest.approx(expected_recall, rel=1e-5)
