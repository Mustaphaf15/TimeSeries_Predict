# TimeSeries Predict Framework

## 1. Vue d'ensemble

**TimeSeries Predict** est un framework de prévision de séries temporelles modulaire, configurable et scalable, conçu pour gérer l'entraînement, l'évaluation et le déploiement de modèles dans un environnement de production.

Il est entièrement construit sur un paradigme de **programmation fonctionnelle** et utilise `sktime` comme bibliothèque de base pour toutes les opérations liées aux séries temporelless.

## 2. Fonctionnalités Clés

- **Architecture 100% Fonctionnelle** : Pas de classes. Le code est organisé en fonctions pures, composables et faciles à tester.
- **Basé sur `sktime`** : Utilisation de l'écosystème `sktime` pour les transformateurs, les modèles, le splitting temporel et le tuning.
- **Configuration Centralisée** : Toute la logique est pilotée par des fichiers de configuration YAML, suivant une hiérarchie à trois niveaux (global, data, entité).
- **Parallélisation** : Capacité à traiter plusieurs entités en parallèle grâce à `joblib`.
- **Générateur de Notebooks EDA** : Un module intégré pour générer automatiquement des notebooks d'analyse exploratoire (EDA) pour chaque entité, avec des recommandations de pipeline de preprocessing.
- **Suivi des Expériences** : Intégration avec MLflow via un module de tracking dédié pour journaliser les paramètres, les métriques et les modèles.
- **Tests Unitaires Complets** : Une suite de tests robuste utilisant `pytest` pour assurer la fiabilité du code.
- **Documentation Automatisée** : Le projet inclut un répertoire `docs/` avec une documentation Markdown générée à partir des docstrings du code source.

## 3. Architecture

### 3.1. Paradigme Fonctionnel

Le framework évite la programmation orientée objet. Chaque composant est une fonction avec une responsabilité unique. Les workflows complexes sont créés en composant ces fonctions, ce qui rend le code plus simple, plus testable et plus facile à maintenir.

### 3.2. Configuration à Trois Niveaux

1.  **`config/global_config.yaml`** : Paramètres généraux du système (MLflow, logging, exécution parallèle).
2.  **`config/data_config.yaml`** : Configuration de la source de données (CSV ou ClickHouse) et des colonnes communes.
3.  **`config/entities/*.yaml`** : Configuration spécifique à chaque entité (série temporelle), incluant les features, les dates, le pipeline de preprocessing et les modèles à tester.

## 4. Structure du Projet

```
.
├── config/                  # Fichiers de configuration YAML
├── data/                    # Données brutes (CSV)
├── docs/                    # Documentation auto-générée des modules
├── src/                     # Code source
│   ├── configuration/
│   ├── data_loading/
│   ├── evaluation/
│   ├── notebook_generator/
│   ├── prediction/
│   ├── preprocessing/
│   ├── splitting/
│   ├── tracking/
│   ├── training/
│   └── tuning/
├── tests/                   # Tests unitaires et fixtures
├── main.py                  # Point d'entrée pour l'exécution du pipeline
└── README.md
```

## 5. Installation

1.  **Clonez le dépôt :**
    ```bash
    git clone <repository_url>
    cd timeseries-predict
    ```

2.  **Installez les dépendances :**
    Il est recommandé d'utiliser un environnement virtuel.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Note: Un fichier `requirements.txt` devrait être créé pour lister les dépendances comme `pandas`, `sktime`, `pyyaml`, `mlflow`, `pytest`, `joblib`, etc.)*

## 6. Utilisation

### 6.1. Exécuter le Pipeline de Prévision

Le script `main.py` est le point d'entrée pour entraîner les modèles et générer des prédictions.

- **Pour une seule entité :**
  ```bash
  python3 main.py --entity test_entity_A
  ```

- **Pour toutes les entités actives (en parallèle) :**
  ```bash
  python3 main.py --all-active
  ```

### 6.2. Générer un Notebook d'Analyse (EDA)

Le module `notebook_generator` peut créer un rapport d'analyse pour une ou plusieurs entités.

- **Pour une seule entité :**
  ```bash
  python3 -m src.notebook_generator.generator test_entity_A --output-dir notebooks
  ```

- **Pour toutes les entités actives :**
  ```bash
  python3 -m src.notebook_generator.generator --all --output-dir notebooks
  ```
Le notebook généré sera sauvegardé dans le répertoire `notebooks/`.

## 7. Tests

Pour lancer la suite de tests unitaires, utilisez `pytest` depuis la racine du projet :
```bash
pytest
```

## 8. Documentation

Une documentation détaillée de chaque module, générée à partir des docstrings du code, est disponible dans le répertoire [`docs/`](./docs/).
