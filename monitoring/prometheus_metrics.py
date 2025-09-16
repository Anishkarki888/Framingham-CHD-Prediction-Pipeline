from prometheus_client import start_http_server, Gauge, Counter
import time
import logging
import threading
import pickle
import os
import sqlite3
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DB = os.path.join(BASE_DIR, "monitoring/mlflow/mlflow.db")
USER_PREDICTIONS = os.path.join(BASE_DIR, "monitoring/data/user_predictions.pkl")
DATA_DRIFT_PKL = os.path.join(BASE_DIR, "streamlit_app/data/data_drift_results.pkl")
CONCEPT_DRIFT_PKL = os.path.join(BASE_DIR, "streamlit_app/data/concept_drift_results.pkl")

# ------------------------------
# Prometheus Metrics
# ------------------------------
model_accuracy = Gauge('framingham_model_accuracy', 'Model accuracy score')
model_precision = Gauge('framingham_model_precision', 'Model precision score')
model_recall = Gauge('framingham_model_recall', 'Model recall score')
model_f1_score = Gauge('framingham_model_f1_score', 'Model F1 score')
model_auc_score = Gauge('framingham_model_auc_score', 'Model AUC score')

total_predictions = Counter('framingham_predictions_total', 'Total number of predictions made')
positive_predictions = Counter('framingham_positive_predictions_total', 'Total positive predictions')

data_drift_score = Gauge('framingham_data_drift_score', 'Data drift score (0-1)')
concept_drift_score = Gauge('framingham_concept_drift_score', 'Concept drift score (0-1)')

model_status = Gauge('framingham_model_status', 'Model health status (1=healthy, 0=unhealthy)')

METRICS_PORT = 8001

# ------------------------------
# Fetch Functions
# ------------------------------

def fetch_mlflow_metrics():
    """Fetch latest model metrics from MLflow DB."""
    if not os.path.exists(MLFLOW_DB):
        logger.warning("MLflow DB not found")
        return {}
    try:
        conn = sqlite3.connect(MLFLOW_DB)
        df = pd.read_sql_query("SELECT metric_name, value FROM metrics ORDER BY timestamp DESC", conn)
        conn.close()
        metrics_dict = {}
        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            filtered = df[df['metric_name'] == metric]
            if not filtered.empty:
                metrics_dict[metric] = filtered.iloc[0]['value']
        return metrics_dict
    except Exception as e:
        logger.error(f"Error fetching MLflow metrics: {e}")
        return {}

def fetch_user_predictions():
    """Fetch total and positive predictions from user_predictions.pkl."""
    total, positive = 0, 0
    if os.path.exists(USER_PREDICTIONS):
        try:
            with open(USER_PREDICTIONS, "rb") as f:
                preds = pickle.load(f)
            total = len(preds)
            positive = sum(1 for p in preds if p.get("prediction", 0) == 1)
        except Exception as e:
            logger.error(f"Error reading user predictions: {e}")
    return total, positive

def fetch_data_drift():
    """Fetch numeric data drift score from PKL."""
    if os.path.exists(DATA_DRIFT_PKL):
        try:
            with open(DATA_DRIFT_PKL, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading data drift PKL: {e}")
    return 0.0

def fetch_concept_drift():
    """Fetch numeric concept drift score from PKL."""
    if os.path.exists(CONCEPT_DRIFT_PKL):
        try:
            with open(CONCEPT_DRIFT_PKL, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading concept drift PKL: {e}")
    return 0.0

# ------------------------------
# Update Metrics
# ------------------------------
def update_metrics():
    """Update Prometheus metrics with real values."""
    try:
        # MLflow metrics
        mlflow_metrics = fetch_mlflow_metrics()
        model_accuracy.set(mlflow_metrics.get("accuracy", 0))
        model_precision.set(mlflow_metrics.get("precision", 0))
        model_recall.set(mlflow_metrics.get("recall", 0))
        model_f1_score.set(mlflow_metrics.get("f1_score", 0))
        model_auc_score.set(mlflow_metrics.get("roc_auc", 0))

        # User predictions
        total, positive = fetch_user_predictions()
        total_predictions._value.set(total)
        positive_predictions._value.set(positive)

        # Drift metrics
        data_drift_score.set(fetch_data_drift())
        concept_drift_score.set(fetch_concept_drift())

        # Model health
        model_status.set(1 if mlflow_metrics.get("roc_auc", 0) > 0.6 else 0)

        logger.info("✅ Prometheus metrics updated")
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")

def metrics_updater():
    while True:
        update_metrics()
        time.sleep(15)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    threading.Thread(target=metrics_updater, daemon=True).start()
    start_http_server(METRICS_PORT)
    logger.info(f"✅ Prometheus metrics server running on :{METRICS_PORT}")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


'''
airflow/dags/framingham_dags/
├─ main_pipeline_dag.py         # Orchestrator DAG
├─ 00_start_services_dag.py     # Start services
├─ 01_data_ingest_dag.py        # Data ingestion
├─ 02_raw_validation_dag.py     # Raw data validation
├─ 03_star_schema_dag.py        # Star schema creation
├─ 04_preprocessing_dag.py      # Preprocessing
├─ 05_processed_validation_dag.py  # Post-process validation
├─ 06_model_training_dag.py     # Tuning + training
├─ 07_model_deploy_dag.py       # Deployment



# Email configuration (replace with your SMTP details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "anish_24152356@sunway.edu.np"  
SMTP_PASSWORD = "sunwaY@123"  
RECIPIENT_EMAIL = "anishkarki989@gmail.com"  

'''