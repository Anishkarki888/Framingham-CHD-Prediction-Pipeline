from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import load_df, store_df, logger, make_redis_client, redis_conn
import great_expectations as ge
import time

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
    description="Validates raw Framingham dataset (Redis first, fallback to local pickle)",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

def validate_raw_data(**kwargs):
    global redis_conn
    make_redis_client()
    
    # Try Redis first with retries
    df = None
    retries = 5
    for i in range(retries):
        try:
            if redis_conn and redis_conn.exists("framingham_raw"):
                df = load_df("framingham_raw")
                logger.info("Loaded 'framingham_raw' from Redis")
                break
        except Exception as e:
            logger.warning(f"Redis check failed: {e}")
        logger.warning(f"'framingham_raw' not in Redis yet, retrying ({i+1}/{retries})...")
        time.sleep(3)
    
    # If Redis failed, fallback to local pickle
    if df is None:
        try:
            df = load_df("framingham_raw")
            logger.info("Redis unavailable. Loaded 'framingham_raw' from local pickle")
        except FileNotFoundError:
            raise ValueError("framingham_raw' not found in Redis or local pickle. Ensure ingestion completed.")

    # Great Expectations validation
    ge_df = ge.dataset.PandasDataset(df)

    # Critical columns
    critical_cols = ["age", "male", "TenYearCHD"]
    missing_critical = [col for col in critical_cols if col not in df.columns]
    if missing_critical:
        raise ValueError(f"Critical columns missing: {missing_critical}")
    
    for col in critical_cols:
        ge_df.expect_column_values_to_not_be_null(col)

    # Optional columns
    optional_cols = [
        "education", "currentSmoker", "cigsPerDay", "BPMeds",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    ]
    for col in optional_cols:
        if col in df.columns:
            ge_df.expect_column_to_exist(col)

    # Run validation
    result = ge_df.validate()
    if result["success"]:
        logger.info("RAW VALIDATION PASSED")
    else:
        logger.warning("RAW VALIDATION COMPLETED WITH WARNINGS")
        logger.warning(result)

validate_raw_data_task = PythonOperator(
    task_id="validate_raw_data",
    python_callable=validate_raw_data,
    dag=dag
)
