from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import load_df, store_df, save_pickle, logger, MODEL_DIR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier
import optuna
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from prometheus_client import Gauge
import pandas as pd
import os

# FIXED: Use different metric names to avoid conflicts with monitoring metrics
PROM_GAUGES = {
    "accuracy": Gauge("training_model_accuracy", "Training Model Accuracy", ["dag_id"]),
    "precision": Gauge("training_model_precision", "Training Model Precision", ["dag_id"]),
    "recall": Gauge("training_model_recall", "Training Model Recall", ["dag_id"]),
    "f1": Gauge("training_model_f1", "Training Model F1 Score", ["dag_id"]),
    "roc_auc": Gauge("training_model_roc_auc", "Training Model ROC-AUC", ["dag_id"])
}

# MLflow setup
MLFLOW_DB_PATH = os.getenv("MLFLOW_DB_PATH", "/home/anish/airflow/dags/mlflow.db")
mlflow.set_tracking_uri(f"sqlite:////{MLFLOW_DB_PATH}")
mlflow.set_experiment("Framingham")

default_args = {
    'owner': 'Anish',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    "model_training",
    default_args=default_args,
    description="Tunes and trains CatBoost model for Framingham dataset",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

def hyperparameter_tuning(**kwargs):
    logger.info("=== STARTING HYPERPARAM TUNING ===")
    
    try:
        X_train = load_df("X_train")
        y_train = load_df("y_train").values.ravel()

        def objective(trial):
            params = {
                'iterations': 100,
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'verbose': 0,
                'random_state': 42
            }
            model = CatBoostClassifier(**params, class_weights=[1, 3000/900])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=1).mean()
            
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("roc_auc", float(score))
            
            return score

        with mlflow.start_run(run_name="catboost_optuna"):
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
            best_params = study.best_params
            
            # Store best parameters
            store_df("best_catboost_params", pd.DataFrame([best_params]))
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric("best_roc_auc", float(study.best_value))
            
            logger.info("=== HYPERPARAM TUNING COMPLETED, Best ROC-AUC=%s ===", study.best_value)
            
    except Exception as e:
        logger.error(f"❌ Hyperparameter tuning failed: {e}")
        raise

def final_model_training(**kwargs):
    logger.info("=== STARTING FINAL MODEL TRAINING ===")
    
    try:
        # Load best parameters and data
        best_params_df = load_df("best_catboost_params")
        best_params = best_params_df.iloc[0].to_dict()
        X_train = load_df("X_train")
        y_train = load_df("y_train").values.ravel()
        X_test = load_df("X_test")
        y_test = load_df("y_test").values.ravel()

        with mlflow.start_run(run_name="final_catboost_model"):
            # Train final model
            model = CatBoostClassifier(**best_params, class_weights=[1, 3000/900])
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
            }
            
            # Update Prometheus metrics (with training_ prefix to avoid conflicts)
            try:
                for metric_name, value in metrics.items():
                    PROM_GAUGES[metric_name].labels(dag_id=kwargs['dag'].dag_id).set(value)
                logger.info("✅ Training metrics updated in Prometheus")
            except Exception as e:
                logger.warning(f"⚠️ Failed to update Prometheus metrics: {e}")
            
            # Log to MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_params(best_params)
            
            # Save model to MLflow
            try:
                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model, 
                    "catboost_best_model", 
                    signature=signature, 
                    input_example=X_train.iloc[:1]
                )
            except Exception as e:
                logger.warning(f"Could not infer signature for final model: {e}")
                mlflow.sklearn.log_model(
                    model, 
                    "catboost_best_model", 
                    input_example=X_train.iloc[:1]
                )
            
            # Save model locally
            save_pickle(model, os.path.join(MODEL_DIR, 'best_catboost_model.pkl'))
            
            # Print classification report
            logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
            logger.info(f"✅ Model Metrics: {metrics}")

        logger.info("=== FINAL MODEL TRAINING COMPLETED ===")
        
    except Exception as e:
        logger.error(f"❌ Final model training failed: {e}")
        raise

# Task definitions
tune_task = PythonOperator(
    task_id="hyperparameter_tuning",
    python_callable=hyperparameter_tuning,
    dag=dag,
)

final_train_task = PythonOperator(
    task_id="final_model_training",
    python_callable=final_model_training,
    provide_context=True,
    dag=dag,
)

# Task dependency
tune_task >> final_train_task