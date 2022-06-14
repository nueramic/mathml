import pytest
import torch

from src.nueramic_mathml.ml import binary_classification_report

metrics_tests = [
    (
        torch.tensor([1, 0, 1, 1]),  # true
        torch.tensor([1, 0, 0, 1]),  # predicted
        torch.tensor([0.9, 0.2, 0.4, 0.8]),  # probabilities
        {
            'recall': 0.6666666865348816,
            'precision': 1.0,
            'f1': 0.8000000143051146,
            'auc_roc': 1.0
        }  # actual report
    )
]


@pytest.mark.parametrize('y_true, y_pred, y_prob, expected', metrics_tests)
def test_bin_cr(y_true, y_pred, y_prob, expected):
    assert binary_classification_report(y_true, y_pred, y_prob) == pytest.approx(expected)
