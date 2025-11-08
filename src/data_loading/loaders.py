from typing import Any, Dict, Tuple
import pandas as pd
from src.configuration.config import get_entity_config

def _build_entity_id(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Construit la colonne 'entity_id' en concaténant les colonnes d'entité."""
    entity_cols = config['data']['entity_columns']
    naming_config = config['data']['entity_naming']
    separator = naming_config.get('separator', '_')
    replace_char = naming_config.get('replace_spaces_with')
    do_lowercase = naming_config.get('lowercase', False)

    temp_df = pd.DataFrame(index=df.index)
    for col in entity_cols:
        series = df.get(col, pd.Series(index=df.index, dtype=str))

        # Handle numeric columns to avoid '.0' issues and preserve NaNs
        if pd.api.types.is_numeric_dtype(series):
            series = series.astype('Int64')

        # Convert the series to string, which safely handles all types including Int64's <NA>
        series_str = series.astype(str)

        # Replace any NaN representations with an empty string and strip whitespace
        series_clean = series_str.replace('<NA>', '').str.strip()

        if do_lowercase:
            series_clean = series_clean.str.lower()
        if replace_char:
            series_clean = series_clean.str.replace(' ', replace_char, regex=False)

        temp_df[col] = series_clean

    def join_non_empty(row, sep):
        """Joins only the non-empty strings in a row."""
        return sep.join(s for s in row if s)

    df['entity_id'] = temp_df[entity_cols].apply(lambda row: join_non_empty(row, separator), axis=1)
    return df

def load_data_from_csv(config: Dict[str, Any], data_type: str = 'backtest') -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV (backtest ou production).
    """
    path = config['csv'][f'{data_type}_path']
    read_options = config['csv'].get('read_options', {})
    na_values = config['csv'].get('na_values')

    df = pd.read_csv(path, na_values=na_values, **read_options)

    date_col = config['data']['date_column']
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

    freq = config['data'].get('date_freq')
    if freq and isinstance(df.index, pd.DatetimeIndex):
        df = df.asfreq(freq)

    return df

def load_data_from_clickhouse(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Charge les données depuis ClickHouse.
    """
    raise NotImplementedError("ClickHouse data loading is not yet implemented.")

def split_backtest_production(
    data: pd.DataFrame,
    production_start_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare les données en backtest et production selon une date de coupure.
    """
    production_start_date = pd.to_datetime(production_start_date)
    backtest_data = data[data.index < production_start_date]
    production_data = data[data.index >= production_start_date]
    return backtest_data, production_data

def load_entity_data(entity_name: str, config_dir: str = "config", entity_config_name: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données backtest et production pour une entité.
    """
    config_to_load = entity_config_name if entity_config_name else entity_name
    config = get_entity_config(config_to_load, config_dir=config_dir)

    # Charger les données brutes
    if config['data']['source'] == 'csv':
        backtest_raw = load_data_from_csv(config, 'backtest')
        production_raw = load_data_from_csv(config, 'production')
    elif config['data']['source'] == 'clickhouse':
        raise NotImplementedError("ClickHouse not implemented")
    else:
        raise ValueError(f"Source de données non supportée: {config['data']['source']}")

    # Construire l'entity_id et filtrer les données pour l'entité voulue
    backtest_with_id = _build_entity_id(backtest_raw, config)
    entity_data = backtest_with_id[backtest_with_id['entity_id'] == entity_name].copy()

    # Concaténer les données de l'entité avec les données de production
    combined_data = pd.concat([entity_data, production_raw], sort=True)

    # Re-establish frequency, as concat can drop it
    freq = config['data'].get('date_freq')
    if freq and isinstance(combined_data.index, pd.DatetimeIndex):
        combined_data = combined_data.asfreq(freq)

    # Nettoyer les colonnes qui ne sont pas des features
    entity_cols_to_drop = config['data']['entity_columns'] + ['entity_id']
    combined_data = combined_data.drop(columns=[col for col in entity_cols_to_drop if col in combined_data.columns], errors='ignore')

    # Séparer en backtest et production
    start_prod = config['date']['start_prod']
    backtest_df, production_df = split_backtest_production(combined_data, start_prod)

    # Validation
    min_backtest = config['date']['validation']['min_backtest_size']
    if len(backtest_df) < min_backtest:
        raise ValueError(f"Taille du backtest ({len(backtest_df)}) insuffisante pour l'entité {entity_name}. Minimum requis: {min_backtest}")

    return backtest_df, production_df
