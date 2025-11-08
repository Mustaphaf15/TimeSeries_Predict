# Documentation du Module : `evaluation`

## Fichier : `evaluators.py`

### ` def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, metrics: List[str]
) -> Dict[str, float] `

```
Calcule un dictionnaire de métriques d'évaluation.
```

### ` def evaluate_predictions(
    y_true: pd.DataFrame,
    predictions: pd.DataFrame,
    folds: List[Dict[str, any]],
    metrics: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, Dict[str, float]] `

```
Évalue les prédictions sur plusieurs plis et retourne les métriques par pli et agrégées.
```
