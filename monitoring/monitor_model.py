import sys
import os
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import start_http_server

# ----------------------------
# Base directory setup
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils import load_pickle
from monitoring.prometheus_metrics import (
    model_accuracy,
    model_precision,
    model_recall,
    model_f1,
    model_roc_auc,
    data_drift_detected,
    concept_drift_detected
)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Paths
# ----------------------------
MONITOR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MONITOR_DIR, "data")
MODELS_DIR = os.path.join(MONITOR_DIR, "models")
REPORTS_DIR = os.path.join(MONITOR_DIR, "reports")
MLFLOW_DB = "/home/anish/airflow/dags/mlflow.db"
PIPELINE_PATH = os.path.join(MODELS_DIR, "final_pipeline.pkl")
USER_PREDICTIONS_PATH = os.path.join(DATA_DIR, "user_predictions.pkl")

# ----------------------------
# MLflow connection
# ----------------------------
def connect_mlflow():
    try:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
        mlflow.set_experiment("Framingham")
        logger.info("✅ MLflow connected")
    except Exception as e:
        logger.error(f"❌ MLflow connection failed: {e}")

# ----------------------------
# Utilities
# ----------------------------
def ensure_dataframe(data, name="dataset"):
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
        logger.info(f"ℹ️ Converted {name} from ndarray to DataFrame")
    return data

def check_files_exist(files):
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        logger.error(f"❌ Missing files: {missing}")
        return False
    return True

# ----------------------------
# Monitoring functions
# ----------------------------
def monitor_model():
    """Monitor model performance and update metrics"""
    logger.info("🔍 Starting model monitoring...")
    
    critical_files = [PIPELINE_PATH,
                      os.path.join(DATA_DIR, "X_test.pkl"),
                      os.path.join(DATA_DIR, "y_test.pkl")]
    if not check_files_exist(critical_files):
        return False

    try:
        connect_mlflow()
        pipeline = load_pickle(PIPELINE_PATH, "ML Pipeline")
        model = pipeline.get("model")
        if model is None:
            raise ValueError("No 'model' found in pipeline")

        X_test = ensure_dataframe(load_pickle(os.path.join(DATA_DIR, "X_test.pkl"), "X_test"))
        y_test = np.array(load_pickle(os.path.join(DATA_DIR, "y_test.pkl"), "y_test")).ravel()

        preds = model.predict(X_test)
        try:
            pred_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            pred_proba = np.zeros_like(preds, dtype=float)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, pred_proba)) if pred_proba.sum() != 0 else 0.0
        }
        logger.info(f"✅ Model metrics: {metrics}")

        # MLflow logging
        try:
            with mlflow.start_run(run_name="model_monitoring"):
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"⚠️ MLflow logging failed: {e}")

        # Update Prometheus gauges
        try:
            model_accuracy.set(metrics["accuracy"])
            model_precision.set(metrics["precision"])
            model_recall.set(metrics["recall"])
            model_f1.set(metrics["f1"])
            model_roc_auc.set(metrics["roc_auc"])
            logger.info("✅ Prometheus metrics updated")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update Prometheus metrics: {e}")

        return True
    except Exception as e:
        logger.error(f"❌ Model monitoring failed: {e}")
        return False

def monitor_data_drift():
    """Monitor data drift and update metrics"""
    logger.info("🔍 Starting data drift monitoring...")
    
    if not os.path.exists(USER_PREDICTIONS_PATH):
        logger.info("No user predictions yet, skipping data drift")
        try:
            data_drift_detected.set(0)
        except Exception:
            pass
        return True

    try:
        X_ref = ensure_dataframe(load_pickle(os.path.join(DATA_DIR, "X_test.pkl"), "X_test"))
        user_preds = load_pickle(USER_PREDICTIONS_PATH, "User Predictions") or []
        if not user_preds:
            data_drift_detected.set(0)
            return True

        user_data = pd.DataFrame([p["input"] for p in user_preds])
        if user_data.empty:
            data_drift_detected.set(0)
            return True

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=X_ref, current_data=user_data)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_file = os.path.join(REPORTS_DIR, f"data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        report.save_html(report_file)
        logger.info(f"✅ Data drift report saved: {report_file}")

        # Check for drift in report
        drift_detected = False
        try:
            report_dict = report.as_dict()
            for metric in report_dict.get('metrics', []):
                if 'result' in metric and metric['result'].get('drift_detected', False):
                    drift_detected = True
                    break
        except Exception as e:
            logger.warning(f"⚠️ Could not parse drift report: {e}")
            drift_detected = False

        try:
            data_drift_detected.set(1 if drift_detected else 0)
            logger.info(f"✅ Data drift status: {'Detected' if drift_detected else 'Not detected'}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update drift metric: {e}")

        try:
            mlflow.log_metric("data_drift_detected", 1 if drift_detected else 0)
        except Exception as e:
            logger.warning(f"⚠️ MLflow logging failed: {e}")

        return True
    except Exception as e:
        logger.error(f"❌ Data drift monitoring failed: {e}")
        try:
            data_drift_detected.set(0)
        except Exception:
            pass
        return False

def monitor_concept_drift():
    """Monitor concept drift and update metrics"""
    logger.info("🔍 Starting concept drift monitoring...")
    
    if not os.path.exists(USER_PREDICTIONS_PATH):
        logger.info("No user predictions yet, skipping concept drift")
        try:
            concept_drift_detected.set(0)
        except Exception:
            pass
        return True

    try:
        pipeline = load_pickle(PIPELINE_PATH, "ML Pipeline")
        model = pipeline.get("model")
        if model is None:
            raise ValueError("No 'model' found in pipeline")

        user_preds = load_pickle(USER_PREDICTIONS_PATH, "User Predictions") or []
        if not user_preds:
            concept_drift_detected.set(0)
            return True

        user_data = pd.DataFrame([p["input"] for p in user_preds])
        user_labels = np.array([p["class"] for p in user_preds])
        if user_data.empty or len(user_labels) == 0:
            concept_drift_detected.set(0)
            return True

        preds = model.predict(user_data)
        try:
            pred_proba = model.predict_proba(user_data)[:, 1]
        except Exception:
            pred_proba = np.zeros_like(preds, dtype=float)

        current_roc_auc = float(roc_auc_score(user_labels, pred_proba)) if pred_proba.sum() != 0 else 0.0

        ROC_THRESHOLD = 0.5
        drift_detected = current_roc_auc < ROC_THRESHOLD
        
        try:
            concept_drift_detected.set(1 if drift_detected else 0)
            logger.info(f"✅ Concept drift status: {'Detected' if drift_detected else 'Not detected'} (ROC-AUC: {current_roc_auc:.3f})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update concept drift metric: {e}")

        try:
            mlflow.log_metric("concept_drift_detected", 1 if drift_detected else 0)
        except Exception as e:
            logger.warning(f"⚠️ MLflow logging failed: {e}")

        return True
    except Exception as e:
        logger.error(f"❌ Concept drift monitoring failed: {e}")
        try:
            concept_drift_detected.set(0)
        except Exception:
            pass
        return False

# ----------------------------
# Main loop to start Prometheus server and monitor continuously
# ----------------------------
if __name__ == "__main__":
    # Start Prometheus server once
    start_http_server(8001)
    logger.info("✅ Prometheus server running on :8001")

    while True:
        monitor_model()
        monitor_data_drift()
        monitor_concept_drift()
        time.sleep(60)