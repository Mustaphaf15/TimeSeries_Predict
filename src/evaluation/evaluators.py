from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
import numpy as np

METRIC_MAP = {
    "mean_absolute_error": mean_absolute_error,
    "mean_squared_error": mean_squared_error,
    "root_mean_squared_error": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
    "r2_score": r2_score,
}

def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, metrics: List[str]
) -> Dict[str, float]:
    """Calcule un dictionnaire de métriques d'évaluation."""
    results = {}
    for metric_name in metrics:
        metric_func = METRIC_MAP.get(metric_name)
        if not metric_func:
            raise ValueError(f"Métrique non supportée: {metric_name}")
        results[metric_name] = metric_func(y_true, y_pred)
    return results

def evaluate_predictions(
    y_true: pd.DataFrame,
    predictions: pd.DataFrame,
    folds: List[Dict[str, any]],
    metrics: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Évalue les prédictions sur plusieurs plis et retourne les métriques par pli et agrégées.
    """
    fold_metrics_list = []

    # Align predictions with y_true to ensure they share the same index
    y_true_aligned, predictions_aligned = y_true.align(predictions, join='inner', axis=0)

    for fold in folds:
        start_date = fold['test_start']
        end_date = fold['test_end']

        # Select data for the current fold using the date range
        y_true_fold = y_true_aligned.loc[start_date:end_date, target_column]
        y_pred_fold = predictions_aligned.loc[start_date:end_date]

        if not y_true_fold.empty:
            # Calculate metrics for the fold
            fold_metrics = compute_metrics(y_true_fold, y_pred_fold, metrics)
            fold_metrics['fold_id'] = fold['fold_id']
            fold_metrics_list.append(fold_metrics)

    # Create a DataFrame with metrics for each fold
    fold_metrics_df = pd.DataFrame(fold_metrics_list).set_index('fold_id')

    # Calculate overall metrics on all predictions
    global_metrics = compute_metrics(y_true_aligned[target_column], predictions_aligned, metrics)

    return fold_metrics_df, global_metrics
