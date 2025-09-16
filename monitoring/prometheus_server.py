from prometheus_client import start_http_server, Gauge, Counter
import time
import logging
import threading
import pickle
import os
import sqlite3
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")
CONCEPT_DRIFT_RESULTS = os.path.join(BASE_DIR, "concept_drift_results.pkl")
USER_PREDICTIONS = os.path.join(BASE_DIR, "streamlit_app/data/user_predictions.pkl")
DATA_DRIFT_REPORT = os.path.join(BASE_DIR, "streamlit_app/reports/data_drift_report.pkl")  # optional if saved

# ------------------------------
# Metrics
# ------------------------------
model_accuracy = Gauge('framingham_model_accuracy', 'Model accuracy score')
model_precision = Gauge('framingham_model_precision', 'Model precision score')
model_recall = Gauge('framingham_model_recall', 'Model recall score')
model_f1_score = Gauge('framingham_model_f1_score', 'Model F1 score')
model_auc_score = Gauge('framingham_model_auc_score', 'Model AUC score')

total_predictions = Counter('framingham_predictions_total', 'Total number of predictions made')
positive_predictions = Counter('framingham_positive_predictions_total', 'Total positive predictions')

data_drift_age = Gauge('framingham_data_drift_age', 'Data drift score for age feature')
data_drift_cholesterol = Gauge('framingham_data_drift_cholesterol', 'Data drift score for cholesterol feature')
data_drift_bp = Gauge('framingham_data_drift_blood_pressure', 'Data drift score for blood pressure feature')

concept_drift_score = Gauge('framingham_concept_drift_score', 'Concept drift detection score')

model_status = Gauge('framingham_model_status', 'Model health status (1=healthy, 0=unhealthy)')

METRICS_PORT = 8001

# ------------------------------
# Functions to fetch real metrics
# ------------------------------

def fetch_mlflow_metrics():
    """Fetch latest model metrics from MLflow DB."""
    if not os.path.exists(MLFLOW_DB):
        logger.warning("MLflow DB not found")
        return None

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
        logger.error(f"Error fetching metrics from MLflow: {e}")
        return None


def fetch_user_predictions():
    """Count total and positive predictions from pickle."""
    total, positive = 0, 0
    if os.path.exists(USER_PREDICTIONS):
        with open(USER_PREDICTIONS, "rb") as f:
            preds = pickle.load(f)
        total = len(preds)
        positive = sum(1 for p in preds if p.get("prediction", 0) == 1)
    return total, positive


def fetch_concept_drift():
    """Fetch latest concept drift AUC."""
    if os.path.exists(CONCEPT_DRIFT_RESULTS):
        with open(CONCEPT_DRIFT_RESULTS, "rb") as f:
            return pickle.load(f)
    return 0.0


def fetch_data_drift():
    """Fetch latest data drift scores (example for age, cholesterol, bp)."""
    # Replace this if you save drift scores separately; using placeholder 0.0 if not
    return {"age": 0.0, "cholesterol": 0.0, "bp": 0.0}


# ------------------------------
# Update metrics function
# ------------------------------
def update_metrics():
    """Update Prometheus metrics with real values."""
    try:
        mlflow_metrics = fetch_mlflow_metrics()
        if mlflow_metrics:
            model_accuracy.set(mlflow_metrics.get("accuracy", 0))
            model_precision.set(mlflow_metrics.get("precision", 0))
            model_recall.set(mlflow_metrics.get("recall", 0))
            model_f1_score.set(mlflow_metrics.get("f1_score", 0))
            model_auc_score.set(mlflow_metrics.get("roc_auc", 0))

        total, positive = fetch_user_predictions()
        total_predictions._value.set(total)
        positive_predictions._value.set(positive)

        drift = fetch_data_drift()
        data_drift_age.set(drift.get("age", 0))
        data_drift_cholesterol.set(drift.get("cholesterol", 0))
        data_drift_bp.set(drift.get("bp", 0))

        concept_drift_score.set(fetch_concept_drift())

        # Model health: healthy if AUC > 0.6
        model_status.set(1 if mlflow_metrics and mlflow_metrics.get("roc_auc", 0) > 0.6 else 0)

        logger.info("✅ Metrics updated")
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")


def metrics_updater():
    """Background thread to continuously update metrics."""
    while True:
        update_metrics()
        time.sleep(15)


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # Start updater thread
    threading.Thread(target=metrics_updater, daemon=True).start()

    # Start Prometheus metrics server
    start_http_server(METRICS_PORT)
    logger.info(f"✅ Prometheus metrics server running on :{METRICS_PORT}")

    # Keep server alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
