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

# ----------------------------
# Wrapper functions for DAG tasks
# ----------------------------
def run_monitor_model():
    """Launch the main user input Streamlit app for monitoring"""
    from monitor_model import run_streamlit_app, APP_PATH, APP_PORT
    run_streamlit_app(APP_PATH, APP_PORT)

def run_monitor_data_drift():
    """Launch the data drift monitoring Streamlit app"""
    from monitor_model import run_streamlit_app, DATA_DRIFT_PATH, DATA_DRIFT_PORT
    run_streamlit_app(DATA_DRIFT_PATH, DATA_DRIFT_PORT)

def run_monitor_concept_drift():
    """Launch the concept drift monitoring Streamlit app"""
    from monitor_model import run_streamlit_app, CONCEPT_DRIFT_PATH, CONCEPT_DRIFT_PORT
    run_streamlit_app(CONCEPT_DRIFT_PATH, CONCEPT_DRIFT_PORT)

# ----------------------------
# Default DAG arguments
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
    

    monitor_model_task = PythonOperator(
        task_id="monitor_model_task",
        python_callable=run_monitor_model,  # <- your wrapper launching Streamlit
        dag=dag
    )

    monitor_data_drift_task = PythonOperator(
        task_id="monitor_data_drift_task",
        python_callable=run_monitor_data_drift,
        dag=dag
    )

    monitor_concept_drift_task = PythonOperator(
        task_id="monitor_concept_drift_task",
        python_callable=run_monitor_concept_drift,
        dag=dag
    )


    # ----------------------------
    # Task dependencies
    # ----------------------------
    monitor_model_task >> monitor_data_drift_task >> monitor_concept_drift_task