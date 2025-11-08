import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.preprocessing.transformers import create_transformer_pipelines, fit_transform_backtest
from src.configuration.config import get_entity_config
from sktime.transformations.compose import TransformerPipeline

CONFIG_DIR = "tests/fixtures/config"

@pytest.fixture
def sample_data():
    """Fixture pour fournir un DataFrame de test."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    return pd.DataFrame({'value': [10, pd.NA, 30]}, index=dates)

def test_create_transformer_pipelines():
    """Teste la création des pipelines de transformation."""
    config = get_entity_config("test_entity_A", config_dir=CONFIG_DIR)
    # Ajout d'un transformateur pour le test
    config['preprocessing']['target_transform'] = [{'type': 'impute', 'transform': 'Imputer', 'method': 'mean'}]

    pipelines = create_transformer_pipelines(config)
    assert isinstance(pipelines['target_pipeline'], TransformerPipeline)
    assert pipelines['target_pipeline'].steps[0][0] == 'impute'

def test_fit_transform_backtest(sample_data):
    """Teste le fit et transform sur les données de backtest."""
    config = get_entity_config("test_entity_A", config_dir=CONFIG_DIR)
    config['preprocessing']['target_transform'] = [{'type': 'impute', 'transform': 'Imputer', 'method': 'mean'}]
    config['data']['target_column'] = 'value'
    config['features'] = {}


    pipelines = create_transformer_pipelines(config)
    transformed_data, _ = fit_transform_backtest(sample_data, pipelines)

    assert isinstance(transformed_data, pd.DataFrame)
    assert not transformed_data.isnull().values.any() # Vérifie qu'il n'y a plus de NaNs
    assert transformed_data['value'][1] == 20 # 10 + 30 / 2
