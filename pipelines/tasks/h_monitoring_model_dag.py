# h_monitoring_model_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# ----------------------------
# Allow imports from monitoring folder
# ----------------------------
sys.path.append("/home/anish/airflow/dags/monitoring")

# Import monitoring functions
def run_monitor_model():
    """Wrapper function to run model monitoring"""
    from monitor_model import monitor_model
    return monitor_model()

def run_monitor_data_drift():
    """Wrapper function to run data drift monitoring"""
    from monitor_model import monitor_data_drift
    return monitor_data_drift()

def run_monitor_concept_drift():
    """Wrapper function to run concept drift monitoring"""
    from monitor_model import monitor_concept_drift
    return monitor_concept_drift()

# ----------------------------
# Default arguments
# ----------------------------
default_args = {
    'owner': 'anish',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

# ----------------------------
# DAG definition
# ----------------------------
with DAG(
    dag_id='monitoring_pipeline',
    default_args=default_args,
    description='ML Model Monitoring Pipeline',
    start_date=datetime(2025, 9, 1),
    schedule_interval='@hourly',
    catchup=False,
    tags=['monitoring', 'ml'],
    max_active_runs=1,
    is_paused_upon_creation=False
) as dag:
    
    # 1️⃣ Model performance monitoring
    monitor_model_task = PythonOperator(
        task_id='monitor_model_task',
        python_callable=run_monitor_model,
        doc_md="Monitor model performance metrics (accuracy, precision, recall, f1, roc_auc)"
    )
    
    # 2️⃣ Data drift monitoring
    monitor_data_drift_task = PythonOperator(
        task_id='monitor_data_drift_task',
        python_callable=run_monitor_data_drift,
        doc_md="Monitor for data drift using Evidently"
    )
    
    # 3️⃣ Concept drift monitoring
    monitor_concept_drift_task = PythonOperator(
        task_id='monitor_concept_drift_task',
        python_callable=run_monitor_concept_drift,
        doc_md="Monitor for concept drift based on model performance"
    )

    # ----------------------------
    # Task dependencies
    # ----------------------------
    monitor_model_task >> monitor_data_drift_task >> monitor_concept_drift_task