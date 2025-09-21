import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

# ----------------------------
# Allow imports from pipelines folder
# ----------------------------
sys.path.append("/home/anish/airflow/dags/pipelines")

# ----------------------------
# Import Python functions from task files
# ----------------------------
from tasks.a_data_ingest_dag import data_ingest
from tasks.b_raw_validation_dag import validate_raw_data
from tasks.c_preprocessing_dag import preprocess_data
from tasks.d_processed_validation_dag import validate_processed_data
from pipelines.tasks.e_model_training_dag import hyperparameter_tuning, final_model_training
from pipelines.tasks.f_model_deploy_dag import deploy_model

# ----------------------------
# Default arguments
# ----------------------------
default_args = {
    'owner': 'Anish',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ----------------------------
# DAG definition
# ----------------------------
dag = DAG(
    "framingham_mlops_pipeline",
    default_args=default_args,
    description="Orchestrates Framingham MLOps pipeline",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

# ----------------------------
# Tasks
# ----------------------------

# Start services via shell script
start_services = BashOperator(
    task_id="start_services",
    bash_command="./start_services.sh",
    cwd="/home/anish/airflow/dags/pipelines",
    dag=dag,
)

# Check if data file exists
check_data_file = FileSensor(
    task_id="check_data_file",
    filepath="/home/anish/framingham.csv",
    fs_conn_id="fs_default",
    poke_interval=30,
    timeout=300,
    dag=dag,
)

# Data ingestion
ingest_task = PythonOperator(
    task_id="data_ingest",
    python_callable=data_ingest,
    dag=dag
)

# Raw data validation
raw_validate_task = PythonOperator(
    task_id="validate_raw_data",
    python_callable=validate_raw_data,
    dag=dag
)

# Preprocess data
preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag
)

# Processed data validation
post_validate_task = PythonOperator(
    task_id="validate_processed_data",
    python_callable=validate_processed_data,
    dag=dag
)

# Hyperparameter tuning
tune_task = PythonOperator(
    task_id="hyperparameter_tuning",
    python_callable=hyperparameter_tuning,
    dag=dag
)

# Final model training
final_train_task = PythonOperator(
    task_id="final_model_training",
    python_callable=final_model_training,
    dag=dag
)

# Model deployment
deploy_task = PythonOperator(
    task_id="deploy_model",
    python_callable=deploy_model,
    dag=dag
)

# Cleanup services placeholder
cleanup_services = BashOperator(
    task_id="cleanup_services",
    bash_command='echo "Cleanup placeholder (services left running)."',
    trigger_rule='all_done',
    dag=dag
)

# ----------------------------
# Task dependencies
# ----------------------------
start_services >> check_data_file
check_data_file >> ingest_task >> raw_validate_task
raw_validate_task >> preprocess_task >> post_validate_task
post_validate_task >> tune_task >> final_train_task >> deploy_task
deploy_task >> cleanup_services
