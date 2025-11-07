import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.splitting.splitters import create_splitter, generate_folds
from src.configuration.config import get_entity_config
from sktime.split import ExpandingWindowSplitter

CONFIG_DIR = "tests/fixtures/config"

@pytest.fixture
def sample_data():
    """Fixture pour fournir un DataFrame de test."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    return pd.DataFrame({'value': range(10)}, index=dates)

def test_create_splitter():
    """Teste la création d'un objet splitter."""
    config = get_entity_config("test_entity_A", config_dir=CONFIG_DIR)
    config['evaluation']['splitter']['initial_window'] = 5

    splitter = create_splitter(
        splitter_config=config['evaluation']['splitter'],
        forecast_horizon=1,
        data_length=10
    )
    assert isinstance(splitter, ExpandingWindowSplitter)
    assert splitter.initial_window == 5

def test_generate_folds(sample_data):
    """Teste la génération des plis de validation."""
    config = get_entity_config("test_entity_A", config_dir=CONFIG_DIR)
    config['evaluation']['splitter']['initial_window'] = 8

    splitter = create_splitter(
        splitter_config=config['evaluation']['splitter'],
        forecast_horizon=1,
        data_length=len(sample_data)
    )
    folds = generate_folds(sample_data, splitter)

    assert isinstance(folds, list)
    assert len(folds) == 2 # 8 train / 1 test, puis 9 train / 1 test
    assert folds[0]['train_size'] == 8
    assert folds[1]['test_size'] == 1
