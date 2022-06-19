import pytest
import torch

from src.nueramic_mathml.ml.metrics import binary_classification_report, precision, recall, accuracy, \
    auc_roc, f_score, r2_score

binary_classification_report_tests = [
    (
        torch.tensor([1, 0, 1, 1]),  # true
        torch.tensor([1, 0, 0, 1]),  # predicted
        torch.tensor([0.9, 0.2, 0.4, 0.8]),  # probabilities
        {
            'recall': 2 / 3,
            'precision': 1,
            'accuracy': 3 / 4,
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
            'accuracy': 1,
            'f1': 1,
            'auc_roc': 1.0
        }

    )
]


@pytest.mark.parametrize('y_true, y_pred, y_prob, expected', binary_classification_report_tests)
def test_bin_cr(y_true, y_pred, y_prob, expected):
    assert binary_classification_report(y_true, y_pred, y_prob) == pytest.approx(expected)


precision_tests = [

    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 0, 0, 1]), 2 / 3, 1, 3 / 4),  # basic test
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1, 1, 1]), 1, 3 / 4, 3 / 4),
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 0, 0, 0]), 1 / 3, 1, 1 / 2),
    (torch.tensor([1, 0, 1, 1]), torch.tensor([0, 1, 0, 0]), 0, 0, 0),
    (torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 0, 1]), 3 / 4, 1, 3 / 4),
    (torch.zeros(10_000), torch.ones(10_000), 0, 0, 0)]


@pytest.mark.parametrize('y_true, y_pred, expected_recall, expected_precision, expected_accuracy', precision_tests)
def test_precision(y_true, y_pred, expected_recall, expected_precision, expected_accuracy):
    assert precision(y_true, y_pred) == pytest.approx(expected_precision, rel=1e-5)


@pytest.mark.parametrize('y_true, y_pred, expected_recall, expected_precision, expected_accuracy', precision_tests)
def test_recall(y_true, y_pred, expected_recall, expected_precision, expected_accuracy):
    assert recall(y_true, y_pred) == pytest.approx(expected_recall, rel=1e-5)


@pytest.mark.parametrize('y_true, y_pred, expected_recall, expected_precision, expected_accuracy', precision_tests)
def test_accuracy(y_true, y_pred, expected_recall, expected_precision, expected_accuracy):
    assert accuracy(y_true, y_pred) == pytest.approx(expected_accuracy, rel=1e-5)


auc_roc_tests = [
    (torch.tensor([1, 0, 1, 1]), torch.tensor([0.9, 0.2, 0.4, 0.8]), 1),
    (torch.cat((torch.zeros(5_000), torch.ones(5_000))),
     torch.cat((torch.ones(5_000) * 0.3, torch.ones(5_000) * 0.7)), 1)

]


@pytest.mark.parametrize('y_true, y_prob, expected', auc_roc_tests)
def test_(y_true, y_prob, expected):
    assert auc_roc(y_true, y_prob) == pytest.approx(expected)


f_tests = [
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 0, 0, 1]), 0.8),  # basic test
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1, 1, 1]), 6 / 7),
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 0, 0, 0]), 0.5),
    (torch.tensor([1, 0, 1, 1]), torch.tensor([0, 1, 0, 0]), 0),
    (torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 0, 1]), 6 / 7),
    (torch.zeros(10_000), torch.ones(10_000), 0)
]


@pytest.mark.parametrize('y_true, y_pred, expected_f', f_tests)
def test_f(y_true, y_pred, expected_f):
    assert f_score(y_true, y_pred) == pytest.approx(expected_f)


r_tests = [
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 0, 0, 1]), -1 / 3),  # basic test
    (torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1, 1, 1]), -1 / 3),
    (torch.tensor([3, -0.5, 2, 7]), torch.tensor([2.5, 0.0, 2, 8]), 0.948)
]


@pytest.mark.parametrize('y_true, y_pred, expected_r', r_tests)
def test_r(y_true, y_pred, expected_r):
    assert r2_score(y_true, y_pred) == pytest.approx(expected_r, rel=1e-3)
