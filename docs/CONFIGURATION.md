# Documentation du Module : `configuration`

## Fichier : `config.py`

### ` def load_config(config_path: str) -> Dict[str, Any] `

```
Charge un fichier YAML de configuration.
```

### ` def merge_configs(base_config: Dict[str, Any], entity_config: Dict[str, Any]) -> Dict[str, Any] `

```
Fusionne la configuration de l'entité avec la configuration de base.
```

### ` def get_globals_config(config_dir: str = "config") -> Dict[str, Any] `

```
Charge et fusionne les configurations globales (global et data).
```

### ` def get_entity_config(entity_name: str, config_dir: str = "config") -> Dict[str, Any] `

```
Charge la configuration complète d'une entité (merge des 3 niveaux).
```

### ` def list_entities(active_only: bool = True, config_dir: str = "config") -> List[str] `

```
Scanne le dossier des entités et retourne une liste de noms d'entités.
```
