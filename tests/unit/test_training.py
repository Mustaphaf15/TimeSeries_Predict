import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.training.trainers import create_forecaster
from src.configuration.config import get_entity_config
from sktime.forecasting.base import BaseForecaster

CONFIG_DIR = "tests/fixtures/config"

def test_create_forecaster():
    """Teste la création d'un forecaster à partir de la configuration."""
    config = get_entity_config("test_entity_A", config_dir=CONFIG_DIR)

    forecaster = create_forecaster(config['models'], "sarimax")

    assert isinstance(forecaster, BaseForecaster)
    assert forecaster.get_params()['order'] == [0, 0, 0]

def test_create_forecaster_not_found():
    """Teste qu'une exception est levée si le modèle n'existe pas."""
    config = get_entity_config("test_entity_A", config_dir=CONFIG_DIR)

    with pytest.raises(ValueError):
        create_forecaster(config['models'], "non_existent_model")
