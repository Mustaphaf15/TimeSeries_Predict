from typing import Any, Dict, List, Tuple
import pandas as pd
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter

def create_splitter(
    splitter_config: Dict[str, Any],
    forecast_horizon: int,
    data_length: int
):
    """Crée un splitter sktime à partir de la configuration."""
    splitter_type = splitter_config.get('type')
    if not splitter_type:
        raise ValueError("Le type de splitter n'est pas spécifié dans la configuration.")

    if splitter_type == "ExpandingWindowSplitter":
        initial_window = splitter_config.get('initial_window', 0.5)
        if isinstance(initial_window, float):
            initial_window = int(data_length * initial_window)

        return ExpandingWindowSplitter(
            initial_window=initial_window,
            step_length=splitter_config.get('step_length', 1),
            fh=list(range(1, forecast_horizon + 1))
        )
    elif splitter_type == "SlidingWindowSplitter":
        window_length = splitter_config.get('window_length')
        if not window_length:
            raise ValueError("`window_length` est requis pour SlidingWindowSplitter.")
        if isinstance(window_length, float):
            window_length = int(data_length * window_length)

        return SlidingWindowSplitter(
            window_length=window_length,
            step_length=splitter_config.get('step_length', 1),
            fh=list(range(1, forecast_horizon + 1))
        )
    else:
        raise ValueError(f"Splitter non supporté: {splitter_type}")

def generate_folds(data: pd.DataFrame, splitter) -> List[Dict[str, Any]]:
    """Génère les plis de validation croisée."""
    folds = []
    for i, (train_indices, test_indices) in enumerate(splitter.split(data)):
        train_start = data.index[train_indices[0]]
        train_end = data.index[train_indices[-1]]
        test_start = data.index[test_indices[0]]
        test_end = data.index[test_indices[-1]]

        folds.append({
            'fold_id': i + 1,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
    return folds

def get_train_test_indices(fold: Dict[str, Any]) -> Tuple[pd.Index, pd.Index]:
    """Récupère les index d'entraînement et de test pour un pli."""
    return fold['train_indices'], fold['test_indices']

def split_data_by_fold(data: pd.DataFrame, fold: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les données en ensembles d'entraînement et de test pour un pli."""
    train_data = data.iloc[fold['train_indices']]
    test_data = data.iloc[fold['test_indices']]
    return train_data, test_data

def get_folds_summary(folds: List[Dict[str, Any]]) -> pd.DataFrame:
    """Crée un résumé des plis générés."""
    summary_data = []
    for fold in folds:
        summary_data.append({
            'fold_id': fold['fold_id'],
            'train_size': fold['train_size'],
            'test_size': fold['test_size'],
            'train_start': fold['train_start'],
            'train_end': fold['train_end'],
            'test_start': fold['test_start'],
            'test_end': fold['test_end'],
        })
    return pd.DataFrame(summary_data)

def validate_splitter_config(config: Dict[str, Any], data_length: int, forecast_horizon: int) -> Dict[str, Any]:
    """Valide la configuration du splitter."""
    errors = []
    initial_window = config.get('initial_window', 0.5)
    if isinstance(initial_window, float):
        initial_window = int(data_length * initial_window)

    if initial_window >= data_length:
        errors.append("La fenêtre initiale est plus grande ou égale à la taille des données.")

    if initial_window + forecast_horizon > data_length:
        errors.append("Pas assez de données pour un seul pli avec l'horizon de prévision donné.")

    splitter = create_splitter(config, forecast_horizon, data_length)
    n_splits = splitter.get_n_splits()

    return {
        'is_valid': not errors,
        'errors': errors,
        'estimated_folds': n_splits
    }
