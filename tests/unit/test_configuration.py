import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.configuration.config import get_globals_config, get_entity_config, list_entities

CONFIG_DIR = "tests/fixtures/config"

def test_get_globals_config():
    """Teste le chargement et la fusion des configurations globales."""
    config = get_globals_config(config_dir=CONFIG_DIR)
    assert "mlflow" in config
    assert "data" in config
    assert config['data']['source'] == 'csv'

def test_get_entity_config():
    """Teste le chargement de la configuration complète d'une entité."""
    config = get_entity_config("test_entity_A", config_dir=CONFIG_DIR)
    assert config['entity_name'] == 'group_A'
    assert config['mlflow']['experiment_prefix'] == 'Test_'
    assert config['data']['target_column'] == 'value'

def test_list_entities_active_only():
    """Teste que list_entities ne retourne que les entités actives."""
    entities = list_entities(active_only=True, config_dir=CONFIG_DIR)
    assert entities == ['test_entity_A']

def test_list_entities_all():
    """Teste que list_entities retourne toutes les entités."""
    entities = list_entities(active_only=False, config_dir=CONFIG_DIR)
    assert sorted(entities) == ['test_entity_A', 'test_entity_B']

def test_get_entity_config_not_found():
    """Teste qu'une exception est levée si l'entité n'existe pas."""
    with pytest.raises(FileNotFoundError):
        get_entity_config("non_existent_entity", config_dir=CONFIG_DIR)
