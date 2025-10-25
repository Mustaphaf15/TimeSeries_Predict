# 🧠 TimeSeries Predict – Application modulable de prévision de séries temporelles

## 🎯 Objectif du projet
TimeSeries Predict est une application flexible et modulaire permettant d'entraîner, évaluer et déployer des modèles de prévision de séries temporelles à partir d’un simple fichier de configuration YAML ou JSON.

L’application permet de définir dynamiquement :
- Le prétraitement des données
- Les modèles de prévision à utiliser
- Les paramètres d’entraînement et d’évaluation

---

## 🧩 Architecture et principes clés
L’application repose sur quatre blocs principaux :
1. Configuration dynamique
2. Pipeline de prétraitement configurable
3. Entraînement et sélection de modèles
4. Évaluation et sauvegarde des résultats

Exemple de configuration :
```yaml
data:
  path: data/input.csv
  target_col: ventes
  datetime_col: date
  freq: D

preprocessing:
  - type: imputation
    method: linear
  - type: scaling
    method: standard

models:
  - name: sarimax
    params:
      order: [2, 1, 2]
      seasonal_order: [1, 1, 1, 7]
  - name: prophet
    params:
      changepoint_prior_scale: 0.1

evaluation:
  splitter: expanding_window
  metric: mape
```

---

## ⚙️ Fonctionnalités principales
- Chargement automatique des données (CSV, Parquet, SQL)
- Prétraitement configurable (imputation, normalisation, encodage temporel, etc.)
- Support multi-modèles : ARIMA, SARIMAX, Prophet, XGBoost, LSTM, etc.
- Recherche d’hyperparamètres (GridSearchCV, RandomizedSearchCV, Optuna)
- Évaluation avec plusieurs métriques (RMSE, MAE, MAPE, etc.)
- Sauvegarde automatique des résultats et des modèles entraînés

---

## 🧱 Structure du projet
```
time_series_predict/
├── config/
│   └── config.yaml
├── data/
│   ├── input.csv
│   └── output/
├── src/
│   ├── preprocessing/
│   │   └── preprocessors.py
│   ├── models/
│   │   ├── forecasters.py
│   │   └── sarimax_regressor.py
│   ├── pipelines/
│   │   └── training_pipeline.py
│   ├── utils/
│   │   ├── logging_utils.py
│   │   └── config_utils.py
│   └── evaluation/
│       └── metrics.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Avantages
- **Modularité totale** : ajout de modèles ou preprocessors sans modifier le cœur du code.
- **Reproductibilité** : toutes les expériences sont définies via un fichier de configuration.
- **Extensibilité** : support des frameworks sktime, statsmodels, prophet, pytorch, tensorflow.
- **Automatisation complète** du prétraitement à la sauvegarde des résultats.

Exécution :
```bash
python main.py --config config/config.yaml
```

---

## 🔮 Évolutions futures
- Interface web (Streamlit ou FastAPI)
- Versionnement des modèles (MLflow)
- API REST pour la prédiction en temps réel
- Détection du data drift (Evidently)
- Intégration AutoML pour séries temporelles
