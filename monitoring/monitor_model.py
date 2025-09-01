# monitor.py
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import Gauge, start_http_server

# ---------------- Prometheus Gauges ----------------
accuracy_gauge = Gauge("model_accuracy", "Model Accuracy")
precision_gauge = Gauge("model_precision", "Model Precision")
recall_gauge = Gauge("model_recall", "Model Recall")
f1_gauge = Gauge("model_f1", "Model F1 Score")
roc_auc_gauge = Gauge("model_roc_auc", "Model ROC-AUC")

METRICS_FILE = "/home/anish/airflow/dags/models/streamlit_app/metrics.pkl"

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def monitor_model():
    # ---------------- MLflow setup ----------------
    mlflow.set_tracking_uri("sqlite:////home/anish/airflow/dags/mlflow.db")
    mlflow.set_experiment("Framingham")

    # ---------------- Load trained pipeline ----------------
    pipeline_path = '/home/anish/airflow/dags/models/final_pipeline.pkl'
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    model = pipeline.get("model")
    if model is None:
        raise ValueError("No 'model' found in final_pipeline.pkl")

    # ---------------- Load datasets ----------------
    X_train = load_pickle("/home/anish/airflow/dags/data/X_train.pkl")
    X_test = load_pickle("/home/anish/airflow/dags/data/X_test.pkl")
    y_train = load_pickle("/home/anish/airflow/dags/data/y_train.pkl")
    y_test = load_pickle("/home/anish/airflow/dags/data/y_test.pkl")

    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    # ---------------- Predictions ----------------
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    try:
        pred_proba_test = model.predict_proba(X_test)[:, 1]
    except Exception:
        pred_proba_test = np.zeros_like(preds_test, dtype=float)

    # ---------------- Compute metrics ----------------
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds_test)),
        "precision": float(precision_score(y_test, preds_test, zero_division=0)),
        "recall": float(recall_score(y_test, preds_test, zero_division=0)),
        "f1": float(f1_score(y_test, preds_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, pred_proba_test)) if pred_proba_test.sum() != 0 else 0.0
    }

    # ---------------- Log metrics to MLflow ----------------
    with mlflow.start_run(run_name="model_monitoring"):
        mlflow.log_metrics(metrics)
        print(f"Logged metrics to MLflow: {metrics}")

    # ---------------- Save metrics for Prometheus ----------------
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "wb") as f:
        pickle.dump(metrics, f)
    print(f"✅ Metrics saved for Prometheus at {METRICS_FILE}")

    # ---------------- Push metrics to Prometheus gauges ----------------
    accuracy_gauge.set(metrics["accuracy"])
    precision_gauge.set(metrics["precision"])
    recall_gauge.set(metrics["recall"])
    f1_gauge.set(metrics["f1"])
    roc_auc_gauge.set(metrics["roc_auc"])

    # ---------------- Evidently Report ----------------
    try:
        X_train_with_target = X_train.copy()
        X_train_with_target["target"] = y_train
        X_train_with_target["prediction"] = preds_train

        X_test_with_target = X_test.copy()
        X_test_with_target["target"] = y_test
        X_test_with_target["prediction"] = preds_test

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=X_train_with_target, current_data=X_test_with_target)

        os.makedirs('/home/anish/airflow/dags/reports', exist_ok=True)
        report_path = f"/home/anish/airflow/dags/reports/evidently_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(report_path)
        print(f"✅ Evidently report saved at {report_path}")
    except Exception as e:
        print(f"⚠️ Failed to generate Evidently report: {e}")

    # ---------------- Performance warning ----------------
    if metrics["roc_auc"] < 0.70:
        print(f"⚠️ Warning: ROC-AUC below threshold (0.70) – {metrics['roc_auc']:.4f}")
    else:
        print("✅ Model performance is satisfactory")

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8001)
    print("🚀 Prometheus metrics server running on :8001")

    # Run monitoring every X seconds (like cronjob / service)
    import time
    while True:
        try:
            monitor_model()
        except Exception as e:
            print("❌ Monitoring error:", e)
        time.sleep(60)  # Run every 1 minute
