# prometheus_metrics.py
from prometheus_client import Gauge
import logging

logger = logging.getLogger(__name__)

# Create metrics with error handling
try:
    model_accuracy = Gauge("model_accuracy", "Model Accuracy")
    model_precision = Gauge("model_precision", "Model Precision")
    model_recall = Gauge("model_recall", "Model Recall")
    model_f1 = Gauge("model_f1", "Model F1 Score")
    model_roc_auc = Gauge("model_roc_auc", "Model ROC-AUC")
    data_drift_detected = Gauge("data_drift_detected", "Data Drift Detected (1=Yes, 0=No)")
    concept_drift_detected = Gauge("concept_drift_detected", "Concept Drift Detected (1=Yes, 0=No)")
    logger.info("✅ Prometheus metrics initialized")
except ValueError as e:
    if "Duplicated timeseries" in str(e):
        logger.warning("⚠️ Metrics already exist, using existing ones")
        # Metrics already exist, continue
        pass
    else:
        logger.error(f"❌ Failed to create metrics: {e}")
        raise
except Exception as e:
    logger.error(f"❌ Unexpected error creating metrics: {e}")
    raise



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