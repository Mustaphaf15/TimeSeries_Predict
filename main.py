import argparse
import pandas as pd
from joblib import Parallel, delayed
from src.configuration.config import get_entity_config, list_entities, get_globals_config
from src.data_loading.loaders import load_entity_data
from src.preprocessing.transformers import (
    create_transformer_pipelines,
    fit_transform_backtest,
    transform_production,
)
from src.splitting.splitters import create_splitter, generate_folds, get_folds_summary
from src.training.trainers import create_forecaster, train_and_predict_on_folds, train_on_fold
from src.evaluation.evaluators import evaluate_predictions
from src.tuning.tuners import create_tuner, tune_hyperparameters
from src.prediction.predictors import generate_predictions
from src.tracking import mlflow_manager, artifacts_logger

def run_model_workflow(config: dict, model_name: str, backtest_transformed: pd.DataFrame, folds: list, y_true: pd.DataFrame, splitter):
    """
    Exécute le workflow pour un modèle donné (avec ou sans tuning).
    """
    print(f"\n--- Démarrage du workflow pour le modèle: {model_name} ---")

    target_col = config['data']['target_column']
    exog_cols = [c for c in backtest_transformed.columns if c != target_col]

    with mlflow_manager.start_run(config['entity_name'], model_name, config.get('tags', {})):
        mlflow_manager.log_params(config)

        if config['models'].get('auto_hyperparameter_tuning', False):
            print(f"\n--- Étape 4a: Tuning d'hyperparamètres pour '{model_name}' ---")
            forecaster_base = create_forecaster(config['models'], model_name)

            tuner = create_tuner(
                search_method=config['models']['search_method'],
                forecaster=forecaster_base,
                param_grid=config['models']['predictors'][model_name],
                splitter=splitter,
                scoring=config['models']['scoring'],
                n_iter=config['models'].get('n_iter', 10)
            )

            best_forecaster, best_params = tune_hyperparameters(
                tuner, backtest_transformed, target_col, exog_cols if exog_cols else None
            )
            print(f"Meilleurs paramètres trouvés: {best_params}")
            mlflow_manager.log_params({f"best_params": best_params})

            final_forecaster_for_training = best_forecaster
        else:
            final_forecaster_for_training = create_forecaster(config['models'], model_name)

        print(f"\n--- Étape 4b: Entraînement du modèle '{model_name}' ---")
        _, predictions = train_and_predict_on_folds(
            final_forecaster_for_training,
            backtest_transformed,
            folds,
            target_col,
            exog_cols if exog_cols else None
        )

        print("\n--- Étape 5: Évaluation des performances ---")
        y_true_transformed = backtest_transformed[[target_col]].loc[predictions.index]

        _, global_metrics = evaluate_predictions(
            y_true=y_true_transformed,
            predictions=predictions,
            folds=folds,
            metrics=config['evaluation']['metrics'],
            target_column=target_col
        )

        print("\n--- Métriques Globales ---")
        for metric, value in global_metrics.items():
            print(f"  {metric}: {value:.4f}")
        mlflow_manager.log_metrics(global_metrics)

        return final_forecaster_for_training, global_metrics


def run_entity_workflow(entity_name: str, model_name: str = None):
    """
    Exécute le workflow complet pour une seule entité.
    """
    config = get_entity_config(entity_name)
    mlflow_manager.init_mlflow_experiment(f"{config['mlflow']['experiment_prefix']}{entity_name}")

    backtest_df, production_df = load_entity_data(entity_name)

    pipelines = create_transformer_pipelines(config)
    backtest_transformed, fitted_pipelines = fit_transform_backtest(backtest_df, pipelines)

    horizon_config = config['evaluation']['forecast_horizon']
    forecast_horizon = horizon_config.get('steps', 1)

    splitter = create_splitter(config['evaluation']['splitter'], forecast_horizon, len(backtest_transformed))
    folds = generate_folds(backtest_transformed, splitter)

    models_to_run = [model_name] if model_name else list(config['models']['predictors'].keys())

    best_model_name = None
    best_model_forecaster = None
    min_metric = float('inf')

    for model in models_to_run:
        try:
            forecaster, metrics = run_model_workflow(config, model, backtest_transformed, folds, backtest_df, splitter)

            scoring_metric = config['models']['scoring']
            if metrics[scoring_metric] < min_metric:
                min_metric = metrics[scoring_metric]
                best_model_name = model
                best_model_forecaster = forecaster

        except Exception as e:
            print(f"ERREUR: Le workflow pour le modèle '{model}' a échoué: {e}")

    if best_model_name:
        print(f"\n--- Étape 6: Prédiction en production avec le meilleur modèle: {best_model_name} ---")

        final_model = train_on_fold(
            best_model_forecaster,
            backtest_transformed,
            config['data']['target_column'],
            [c for c in backtest_transformed.columns if c != config['data']['target_column']]
        )
        artifacts_logger.log_model(final_model, entity_name, best_model_name, config)

        production_transformed = transform_production(production_df, fitted_pipelines)

        prod_predictions = generate_predictions(final_model, production_transformed, forecast_horizon)

        print("\n--- Prédictions pour la production ---")
        print(prod_predictions)
        artifacts_logger.log_predictions(prod_predictions, entity_name, best_model_name, config)

    print(f"\n--- Workflow terminé pour l'entité {entity_name} ---")


def main(entity_arg: str, model_arg: str):
    """
    Point d'entrée principal. Gère la parallélisation et les arguments 'all'.
    """
    globals_config = get_globals_config()
    mlflow_manager.init_mlflow(globals_config['mlflow']['tracking_uri'])

    parallel_config = globals_config.get('execution', {}).get('parallel', {})

    entities_to_process = list_entities(active_only=True) if entity_arg == "all" else [entity_arg]

    model_to_run = model_arg if model_arg != "all" else None

    if parallel_config.get('enabled', False) and len(entities_to_process) > 1:
        Parallel(n_jobs=parallel_config.get('n_jobs', -1), backend=parallel_config.get('backend', 'loky'))(
            delayed(run_entity_workflow)(entity, model_to_run) for entity in entities_to_process
        )
    else:
        for entity in entities_to_process:
            run_entity_workflow(entity, model_to_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer le pipeline de prévision de séries temporelles.")
    parser.add_argument("entity", type=str, help="Le nom de l'entité à traiter ou 'all' pour toutes les entités actives.")
    parser.add_argument("--model", type=str, default="all", help="Le nom du modèle à entraîner, ou 'all' pour tous les modèles de la config.")

    args = parser.parse_args()
    main(args.entity, args.model)
