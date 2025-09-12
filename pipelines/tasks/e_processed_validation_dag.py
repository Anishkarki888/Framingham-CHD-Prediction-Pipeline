from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import load_df, logger
import great_expectations as ge
from prometheus_client import Gauge

# Prometheus metrics
PROM_GAUGES = {
    "validation_success": Gauge("data_validation_success", "Whether data validation passed (1=success, 0=failure)", ["dag_id"])
}

default_args = {
    'owner': 'Anish',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    "processed_validation",
    default_args=default_args,
    description="Validates preprocessed Framingham dataset",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

def validate_processed_data(**kwargs):
    logger.info("=== VALIDATING PROCESSED DATA ===")
    df = load_df("framingham_clean")
    ge_df = ge.dataset.PandasDataset(df)

    for col in df.columns:
        ge_df.expect_column_values_to_not_be_null(col)

    if "age" in df.columns:
        ge_df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
    if "TenYearCHD" in df.columns:
        ge_df.expect_column_values_to_be_in_set("TenYearCHD", [0, 1])
    if "male" in df.columns:
        ge_df.expect_column_values_to_be_in_set("male", [0, 1])
    if "education" in df.columns:
        ge_df.expect_column_values_to_be_between("education", min_value=1, max_value=4)

    result = ge_df.validate()
    PROM_GAUGES["validation_success"].labels(dag_id=kwargs['dag'].dag_id).set(1 if result["success"] else 0)

    if not result["success"]:
        raise ValueError("Processed validation failed")

    logger.info("=== PROCESSED DATA VALIDATION PASSED ===")

post_validate_task = PythonOperator(
    task_id="validate_processed_data",
    python_callable=validate_processed_data,
    provide_context=True,
    dag=dag,
)