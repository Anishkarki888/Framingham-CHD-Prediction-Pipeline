from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import get_engine_with_retry, store_df, logger, make_redis_client, redis_conn
import pandas as pd
import os
import time

# -------------------------------
# Default DAG args
# -------------------------------
default_args = {
    'owner': 'Anish',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    "data_ingest",
    default_args=default_args,
    description="Ingests Framingham dataset into MariaDB and Redis",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

# -------------------------------
# Task function
# -------------------------------
def data_ingest(**kwargs):
    logger.info("=== STARTING DATA INGEST (ELT: Load RAW) ===")
    
    # Initialize Redis client
    make_redis_client()
    r = redis_conn()
    if not r:
        raise ValueError("❌ Redis client could not be initialized")

    # Find dataset
    paths = ['/home/anish/airflow/dags/framingham.csv']
    df = None
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            logger.info("Loaded dataset from %s, shape=%s", p, df.shape)
            break
    
    if df is None:
        raise FileNotFoundError("framingham.csv not found in any specified path")
    
    # Add unique patient ID
    df['patient_id'] = range(1, len(df) + 1)
    
    # Persist to Redis + local
    store_df("framingham_raw", df)

    # Confirm Redis key exists before finishing
    retries = 5
    for i in range(retries):
        if r.exists("framingham_raw"):
            logger.info("✅ 'framingham_raw' successfully stored in Redis")
            break
        logger.warning(f"'framingham_raw' not yet in Redis, retrying... ({i+1}/{retries})")
        time.sleep(2)
    else:
        raise ValueError("❌ Failed to store 'framingham_raw' in Redis after retries")

    # Write to MariaDB staging
    try:
        engine = get_engine_with_retry()
        with engine.begin() as conn:
            df.to_sql('staging_framingham', conn, if_exists='replace', index=False)
        logger.info("RAW data written to MariaDB (staging_framingham)")
    except Exception as e:
        logger.warning("Failed to write staging to MariaDB: %s", e)
    
    logger.info("=== DATA INGEST COMPLETED ===")

# -------------------------------
# DAG task
# -------------------------------
ingest_task = PythonOperator(
    task_id="data_ingest",
    python_callable=data_ingest,
    dag=dag,
)
