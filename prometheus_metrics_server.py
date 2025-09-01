from prometheus_client import start_http_server, Gauge
import time

# Define Gauges
PROM_GAUGES = {
    "accuracy": Gauge("model_accuracy", "Model Accuracy"),
    "precision": Gauge("model_precision", "Model Precision"),
    "recall": Gauge("model_recall", "Model Recall"),
    "f1": Gauge("model_f1", "Model F1 Score"),
    "roc_auc": Gauge("model_roc_auc", "Model ROC-AUC"),
}

start_http_server(8001)
print("✅ Prometheus metrics server started on port 8001")

# Dummy loop (replace with reading metrics from file/db/redis)
while True:
    # Example values — replace with real DAG metrics
    PROM_GAUGES["accuracy"].set(0.78)
    PROM_GAUGES["precision"].set(0.25)
    PROM_GAUGES["recall"].set(0.24)
    PROM_GAUGES["f1"].set(0.248)
    PROM_GAUGES["roc_auc"].set(0.63)
    time.sleep(15)
