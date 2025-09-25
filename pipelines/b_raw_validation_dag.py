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
    "raw_validation",
    default_args=default_args,
    description="Validates raw Framingham dataset",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

def validate_raw_data(**kwargs):
    logger.info("=== VALIDATING RAW DATA ===")
    df = load_df("framingham_raw")
    ge_df = ge.dataset.PandasDataset(df)

    required_cols = [
        "age", "male", "education", "currentSmoker", "cigsPerDay", "BPMeds",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in RAW: {col}")
        ge_df.expect_column_to_exist(col)

    for col in ["age", "male", "TenYearCHD"]:
        ge_df.expect_column_values_to_not_be_null(col)

    result = ge_df.validate()
    PROM_GAUGES["validation_success"].labels(dag_id=kwargs['dag'].dag_id).set(1 if result["success"] else 0)

    if not result["success"]:
        raise ValueError(f"RAW validation failed: {result}")

    logger.info("=== RAW VALIDATION PASSED ===")

raw_validate_task = PythonOperator(
    task_id="validate_raw_data",
    python_callable=validate_raw_data,
    provide_context=True,
    dag=dag,
)