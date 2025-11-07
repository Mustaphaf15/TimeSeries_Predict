import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.evaluation.evaluators import compute_metrics

def test_compute_metrics():
    """Teste le calcul des métriques d'évaluation."""
    y_true = pd.Series([100, 110, 120])
    y_pred = pd.Series([95, 115, 125])

    metrics_to_compute = ["mean_absolute_error", "root_mean_squared_error"]

    results = compute_metrics(y_true, y_pred, metrics_to_compute)

    assert "mean_absolute_error" in results
    assert "root_mean_squared_error" in results
    assert results["mean_absolute_error"] == 5.0
    assert results["root_mean_squared_error"] == 5.0

def test_compute_metrics_unsupported():
    """Teste qu'une exception est levée pour une métrique non supportée."""
    y_true = pd.Series([1, 2, 3])
    y_pred = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        compute_metrics(y_true, y_pred, ["non_existent_metric"])
