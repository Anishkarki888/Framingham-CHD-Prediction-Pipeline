from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import get_engine_with_retry, store_df, logger
import pandas as pd

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

def data_ingest(**kwargs):
    logger.info("=== STARTING DATA INGEST (ELT: Load RAW) ===")
    paths = [
        '/home/anish/framingham.csv',
        './framingham.csv',
        '/tmp/framingham.csv',
        '/home/anish/airflow/dags/framingham.csv'
    ]
    df = None
    for p in paths:
        try:
            df = pd.read_csv(p)
            logger.info("Loaded dataset from %s", p)
            break
        except FileNotFoundError:
            continue
    if df is None:
        raise FileNotFoundError("framingham.csv not found in any specified path")

    df['patient_id'] = range(1, len(df) + 1)

    # Persist raw data for downstream tasks
    store_df("framingham_raw", df)

    # Write to MariaDB staging table
    try:
        engine = get_engine_with_retry()
        with engine.begin() as conn:
            df.to_sql('staging_framingham', conn, if_exists='replace', index=False)
        logger.info("RAW data written to MariaDB (staging_framingham)")
    except Exception as e:
        logger.warning("Failed to write staging to MariaDB: %s", e)

    logger.info("=== DATA INGEST COMPLETED ===")

ingest_task = PythonOperator(
    task_id="data_ingest",
    python_callable=data_ingest,
    dag=dag,
)