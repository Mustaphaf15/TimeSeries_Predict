import argparse
import pandas as pd
from src.configuration.config import get_entity_config
from src.data_loading.loaders import load_entity_data
from src.preprocessing.transformers import (
    create_transformer_pipelines,
    fit_transform_backtest,
)
from src.splitting.splitters import create_splitter, generate_folds, get_folds_summary
from src.training.trainers import create_forecaster, train_and_predict_on_folds
from src.evaluation.evaluators import evaluate_predictions
from src.logging import loggers as mlflow_log

def main(entity_name: str, model_name: str):
    """
    Workflow principal pour l'entraînement et l'évaluation d'un modèle.
    """
    # --- 1. Configuration & Logging ---
    print(f"--- Démarrage du workflow pour l'entité: {entity_name} ---")
    config = get_entity_config(entity_name)
    mlflow_log.init_mlflow_experiment(f"{config['mlflow']['experiment_prefix']}{entity_name}")
    mlflow_log.log_params(config)

    # --- 2. Chargement des données ---
    print("\n--- Étape 1: Chargement des données ---")
    backtest_df, _ = load_entity_data(entity_name)
    mlflow_log.log_dataset(backtest_df, "backtest_raw")
    print(f"Données backtest chargées: {backtest_df.shape}")

    # --- 3. Prétraitement ---
    print("\n--- Étape 2: Prétraitement des données ---")
    pipelines = create_transformer_pipelines(config)
    backtest_transformed, fitted_pipelines = fit_transform_backtest(backtest_df, pipelines)
    print(f"Données backtest transformées: {backtest_transformed.shape}")

    # --- 4. Découpage (Splitting) ---
    print("\n--- Étape 3: Découpage en plis (Folds) ---")
    horizon_config = config['evaluation']['forecast_horizon']
    forecast_horizon = horizon_config['steps'] if horizon_config['is_relative'] else len(production_df)

    splitter = create_splitter(
        splitter_config=config['evaluation']['splitter'],
        forecast_horizon=forecast_horizon,
        data_length=len(backtest_transformed)
    )
    folds = generate_folds(backtest_transformed, splitter)
    print(f"{len(folds)} plis générés.")
    print(get_folds_summary(folds))

    # --- 5. Entraînement & Prédiction ---
    print(f"\n--- Étape 4: Entraînement du modèle '{model_name}' ---")
    forecaster = create_forecaster(config['models'], model_name)

    target_col = config['data']['target_column']
    exog_cols = [c for c in backtest_transformed.columns if c != target_col]

    final_forecaster, predictions = train_and_predict_on_folds(
        forecaster,
        backtest_transformed,
        folds,
        target_col,
        exog_cols if exog_cols else None
    )
    print("Entraînement et prédictions sur les plis terminés.")
    mlflow_log.log_forecaster(final_forecaster, f"models/{model_name}")

    # --- 6. Évaluation ---
    print("\n--- Étape 5: Évaluation des performances ---")
    # Utiliser les données transformées pour y_true pour éviter les NaNs
    y_true_transformed = backtest_transformed[[target_col]].loc[predictions.index]

    fold_metrics, global_metrics = evaluate_predictions(
        y_true=y_true_transformed,
        predictions=predictions,
        folds=folds,
        metrics=config['evaluation']['metrics'],
        target_column=target_col
    )

    print("\n--- Métriques Globales ---")
    for metric, value in global_metrics.items():
        print(f"  {metric}: {value:.4f}")
    mlflow_log.log_metrics(global_metrics)

    print("\n--- Métriques par Pli ---")
    print(fold_metrics)

    print(f"\n--- Workflow terminé pour {entity_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer le pipeline de prévision de séries temporelles.")
    parser.add_argument("entity", type=str, help="Le nom de l'entité à traiter (ex: 75_Paris_Sud).")
    parser.add_argument("--model", type=str, default="sarimax", help="Le nom du modèle à entraîner (ex: sarimax, fbprophet).")

    args = parser.parse_args()
    main(args.entity, args.model)
