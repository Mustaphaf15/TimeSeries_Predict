from typing import Any, Dict, List, Tuple
import pandas as pd
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter
from sktime.transformations.series.fourier import FourierFeatures
from sktime.transformations.series.lag import Lag
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.boxcox import BoxCoxTransformer, LogTransformer
from sktime.transformations.compose import TransformerPipeline

# Mapping des transformers supportés
TRANSFORMER_MAP = {
    "Imputer": Imputer,
    "HampelFilter": HampelFilter,
    "FourierFeatures": FourierFeatures,
    "Lag": Lag,
    "WindowSummarizer": WindowSummarizer,
    "DateTimeFeatures": DateTimeFeatures,
    "BoxCoxTransformer": BoxCoxTransformer,
    "LogTransformer": LogTransformer,
}

def _create_single_pipeline(transformer_configs: List[Dict[str, Any]]) -> TransformerPipeline:
    """Helper to create a single sktime pipeline from a list of configs."""
    steps = []
    if not transformer_configs:
        return None # Retourner None si aucune transformation n'est définie

    for config in transformer_configs:
        transformer_class = TRANSFORMER_MAP.get(config['transform'])
        if not transformer_class:
            raise ValueError(f"Transformer non supporté: {config['transform']}")

        params = {k: v for k, v in config.items() if k not in ['type', 'transform']}
        steps.append((config['type'], transformer_class(**params)))

    return TransformerPipeline(steps)

def create_transformer_pipelines(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Crée des pipelines séparés pour target et exogènes."""
    target_pipeline = _create_single_pipeline(config['preprocessing'].get('target_transform', []))
    exog_pipeline = _create_single_pipeline(config['preprocessing'].get('exog_transform', []))

    return {
        'target_pipeline': target_pipeline,
        'exog_pipeline': exog_pipeline,
        'target_column': config['data']['target_column'],
        'exog_cat_columns': config['features'].get('exog_cat_column', []),
        'exog_num_columns': config['features'].get('exog_num_column', [])
    }

def encode_categorical_features(
    data: pd.DataFrame,
    cat_columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Encode les variables catégorielles et retourne les nouvelles colonnes."""
    if not cat_columns:
        return data, []

    df_encoded = pd.get_dummies(data, columns=cat_columns, drop_first=True)
    new_cols = [c for c in df_encoded.columns if c not in data.columns]

    return df_encoded, new_cols

def fit_transform_backtest(
    data: pd.DataFrame,
    pipelines: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit et transforme les données backtest."""
    df_encoded, new_cat_cols = encode_categorical_features(data, pipelines['exog_cat_columns'])

    fitted_pipelines = pipelines.copy()
    fitted_pipelines['encoded_cat_columns'] = new_cat_cols

    target_col = pipelines['target_column']
    exog_cols = pipelines['exog_num_columns'] + new_cat_cols

    y = df_encoded[[target_col]]
    X = df_encoded[exog_cols] if exog_cols else None

    y_transformed = pipelines['target_pipeline'].fit_transform(y) if pipelines['target_pipeline'] else y

    if X is not None and pipelines['exog_pipeline']:
        X_transformed = pipelines['exog_pipeline'].fit_transform(X)
    else:
        X_transformed = X

    if X_transformed is not None:
        transformed_data = pd.concat([y_transformed, X_transformed], axis=1)
    else:
        transformed_data = y_transformed

    return transformed_data, fitted_pipelines


def transform_production(
    data: pd.DataFrame,
    fitted_pipelines: Dict[str, Any]
) -> pd.DataFrame:
    """Transforme les données production avec pipelines déjà fittés."""
    df_encoded, _ = encode_categorical_features(data, fitted_pipelines['exog_cat_columns'])

    exog_cols = fitted_pipelines['exog_num_columns'] + fitted_pipelines['encoded_cat_columns']
    df_aligned = df_encoded.reindex(columns=exog_cols, fill_value=0)

    if not fitted_pipelines['exog_pipeline']:
         return df_aligned

    transformed_data = fitted_pipelines['exog_pipeline'].transform(df_aligned)
    return transformed_data
