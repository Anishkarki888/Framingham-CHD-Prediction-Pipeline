from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import os

BASE_PATH = "/home/anish/airflow/dags/data"
CONCEPT_DRIFT_THRESHOLD = 0.5

def check_retrain_conditions():
    """Check if retraining is needed based on new data or concept drift"""
    retrain_needed = False
    
    # Check new patient data (example: presence of new CSV)
    if os.path.exists(os.path.join(BASE_PATH, "new_patient_data.csv")):
        retrain_needed = True
    
    # Check concept drift value
    concept_drift_file = os.path.join(BASE_PATH, "concept_drift_value.txt")
    if os.path.exists(concept_drift_file):
        with open(concept_drift_file, 'r') as f:
            value = float(f.read().strip())
            if value < CONCEPT_DRIFT_THRESHOLD:
                retrain_needed = True

    return retrain_needed

default_args = {
    'owner': 'anish',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='retrain_pipeline',
    default_args=default_args,
    description='Hybrid Auto-Retrain DAG',
    start_date=datetime(2025, 9, 1),
    schedule_interval='@weekly',  # Scheduled retrain (hybrid)
    catchup=False,
    max_active_runs=1,
) as dag:

    check_retrain_task = PythonOperator(
        task_id="check_retrain_conditions",
        python_callable=check_retrain_conditions
    )

    trigger_retrain_task = TriggerDagRunOperator(
        task_id="trigger_main_pipeline",
        trigger_dag_id="framingham_mlops_pipeline",
        wait_for_completion=False
    )

    check_retrain_task >> trigger_retrain_task
