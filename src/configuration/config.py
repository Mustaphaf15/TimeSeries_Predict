from typing import Any, Dict, List
import yaml
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    """Charge un fichier YAML de configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(base_config: Dict[str, Any], entity_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne la configuration de l'entité avec la configuration de base."""
    merged = base_config.copy()
    for key, value in entity_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def get_entity_config(entity_name: str, config_dir: str = "config") -> Dict[str, Any]:
    """Charge la configuration complète d'une entité (merge des 3 niveaux)."""
    config_path = Path(config_dir)
    global_config = load_config(config_path / "global_config.yaml")
    data_config = load_config(config_path / "data_config.yaml")
    entity_config_path = config_path / "entities" / f"{entity_name}.yaml"
    if not entity_config_path.exists():
        raise FileNotFoundError(f"Configuration file for entity '{entity_name}' not found at {entity_config_path}")
    entity_config = load_config(entity_config_path)

    # Merge configs: entity overrides data, which overrides global
    merged_config = merge_configs(global_config, data_config)
    final_config = merge_configs(merged_config, entity_config)

    return final_config

def list_entities(active_only: bool = True, config_dir: str = "config") -> List[str]:
    """
    Scanne le dossier des entités et retourne une liste de noms d'entités.
    """
    entities_path = Path(config_dir) / "entities"
    all_entities = [p.stem for p in entities_path.glob("*.yaml") if not p.name.startswith('_')]

    if not active_only:
        return sorted(all_entities)

    active_entities = []
    for entity_name in all_entities:
        try:
            config = get_entity_config(entity_name, config_dir)
            if config.get('active', False):
                active_entities.append(entity_name)
        except Exception as e:
            print(f"Warning: Could not load or validate config for entity '{entity_name}': {e}")

    return sorted(active_entities)
