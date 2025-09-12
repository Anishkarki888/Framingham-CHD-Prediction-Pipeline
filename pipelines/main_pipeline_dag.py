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
from tasks.c_star_schema_dag import build_star_schema
from tasks.d_preprocessing_dag import preprocess_data
from tasks.e_processed_validation_dag import validate_processed_data
from tasks.f_model_training_dag import hyperparameter_tuning, final_model_training
from tasks.g_model_deploy_dag import deploy_model

# ----------------------------
# Monitoring wrapper functions (NO DIRECT IMPORTS)
# ----------------------------
def run_monitor_model():
    """Wrapper function to run model monitoring"""
    import sys
    sys.path.append("/home/anish/airflow/dags/monitoring")
    from monitor_model import monitor_model
    return monitor_model()

def run_monitor_data_drift():
    """Wrapper function to run data drift monitoring"""
    import sys
    sys.path.append("/home/anish/airflow/dags/monitoring")
    from monitor_model import monitor_data_drift
    return monitor_data_drift()

def run_monitor_concept_drift():
    """Wrapper function to run concept drift monitoring"""
    import sys
    sys.path.append("/home/anish/airflow/dags/monitoring")
    from monitor_model import monitor_concept_drift
    return monitor_concept_drift()

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
    description="Orchestrates Framingham MLOps pipeline with monitoring",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

# ----------------------------
# Tasks
# ----------------------------

# 1️⃣ Start services via shell script
start_services = BashOperator(
    task_id="start_services",
    bash_command="./start_services.sh",
    cwd="/home/anish/airflow/dags/pipelines",
    dag=dag,
)

# 2️⃣ Check if data file exists
check_data_file = FileSensor(
    task_id="check_data_file",
    filepath="/home/anish/framingham.csv",
    fs_conn_id="fs_default",
    poke_interval=30,
    timeout=300,
    dag=dag,
)

# 3️⃣ Data ingestion
ingest_task = PythonOperator(
    task_id="data_ingest",
    python_callable=data_ingest,
    dag=dag
)

# 4️⃣ Raw data validation
raw_validate_task = PythonOperator(
    task_id="validate_raw_data",
    python_callable=validate_raw_data,
    dag=dag
)

# 5️⃣ Build star schema
star_schema_task = PythonOperator(
    task_id="build_star_schema",
    python_callable=build_star_schema,
    dag=dag
)

# 6️⃣ Preprocess data
preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag
)

# 7️⃣ Processed data validation
post_validate_task = PythonOperator(
    task_id="validate_processed_data",
    python_callable=validate_processed_data,
    dag=dag
)

# 8️⃣ Hyperparameter tuning
tune_task = PythonOperator(
    task_id="hyperparameter_tuning",
    python_callable=hyperparameter_tuning,
    dag=dag
)

# 9️⃣ Final model training
final_train_task = PythonOperator(
    task_id="final_model_training",
    python_callable=final_model_training,
    dag=dag
)

# 🔟 Model deployment
deploy_task = PythonOperator(
    task_id="deploy_model",
    python_callable=deploy_model,
    dag=dag
)

# 1️⃣1️⃣ Monitoring tasks (using wrapper functions)
monitor_model_task = PythonOperator(
    task_id="monitor_model_task",
    python_callable=run_monitor_model,  # Using wrapper function
    dag=dag
)

monitor_data_drift_task = PythonOperator(
    task_id="monitor_data_drift_task",
    python_callable=run_monitor_data_drift,  # Using wrapper function
    dag=dag
)

monitor_concept_drift_task = PythonOperator(
    task_id="monitor_concept_drift_task",
    python_callable=run_monitor_concept_drift,  # Using wrapper function
    dag=dag
)

# 1️⃣2️⃣ Cleanup services placeholder
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
check_data_file >> ingest_task >> raw_validate_task >> star_schema_task
star_schema_task >> preprocess_task >> post_validate_task
post_validate_task >> tune_task >> final_train_task >> deploy_task
deploy_task >> monitor_model_task >> monitor_data_drift_task >> monitor_concept_drift_task >> cleanup_services