from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import load_df, store_df, get_engine_with_retry, logger, make_redis_client
import pandas as pd
import os

default_args = {
    'owner': 'Anish',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    "star_schema",
    default_args=default_args,
    description="Builds star schema for Framingham dataset in MariaDB",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

def build_star_schema(**kwargs):
    logger.info("=== STARTING STAR SCHEMA BUILD ===")
    
    # Initialize Redis client
    make_redis_client()
    
    # Load data with multiple fallback methods
    df = None
    
    # Method 1: Try load_df function
    try:
        df = load_df("framingham_raw")
        if df is not None:
            logger.info(f"✅ Loaded data via load_df: {df.shape}")
    except Exception as e:
        logger.warning(f"load_df failed: {e}, trying direct pickle load")
    
    # Method 2: Direct pickle load if load_df failed
    if df is None:
        try:
            pickle_path = "/home/anish/airflow/dags/monitoring/data/framingham_raw.pkl"
            if os.path.exists(pickle_path):
                df = pd.read_pickle(pickle_path)
                logger.info(f"✅ Loaded data via direct pickle: {df.shape}")
            else:
                logger.error(f"Pickle file doesn't exist: {pickle_path}")
        except Exception as e:
            logger.error(f"Direct pickle load failed: {e}")
    
    # Method 3: Load from original CSV as last resort
    if df is None:
        try:
            csv_path = "/home/anish/airflow/dags/framingham.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['patient_id'] = range(1, len(df) + 1)  # Add patient_id like in ingestion
                logger.info(f"✅ Loaded data from CSV fallback: {df.shape}")
            else:
                logger.error(f"CSV file doesn't exist: {csv_path}")
        except Exception as e:
            logger.error(f"CSV fallback failed: {e}")
    
    if df is None:
        raise ValueError("Could not load framingham_raw data from any source")
    
    # Proceed with star schema creation
    df = df.copy()
    
    # Ensure patient_id is int
    if 'patient_id' not in df.columns:
        df['patient_id'] = range(1, len(df) + 1)
    
    df["patient_id_int"] = df["patient_id"].astype(int)
    
    # Create dimension and fact tables
    logger.info("Creating dimension tables...")
    
    dim_demo = df[["patient_id_int", "male", "age", "education"]].drop_duplicates().rename(columns={"patient_id_int": "patient_id"})
    dim_life = df[["patient_id_int", "currentSmoker", "cigsPerDay"]].drop_duplicates().rename(columns={"patient_id_int": "patient_id", "currentSmoker": "current_smoker"})
    
    # Handle optional columns that might not exist
    medical_cols = ["patient_id_int", "BPMeds"]
    if "prevalentStroke" in df.columns:
        medical_cols.append("prevalentStroke")
    if "prevalentHyp" in df.columns:
        medical_cols.append("prevalentHyp")
    if "diabetes" in df.columns:
        medical_cols.append("diabetes")
    
    dim_med = df[medical_cols].drop_duplicates().rename(columns={"patient_id_int": "patient_id"})
    
    test_cols = ["patient_id_int", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
    # Only include columns that actually exist
    test_cols = [col for col in test_cols if col in df.columns]
    dim_test = df[test_cols].drop_duplicates().rename(columns={"patient_id_int": "patient_id"})
    
    fact_chd = df[["patient_id", "TenYearCHD"]].copy()
    
    logger.info(f"Created tables - Demo: {dim_demo.shape}, Life: {dim_life.shape}, Med: {dim_med.shape}, Test: {dim_test.shape}, Fact: {fact_chd.shape}")
    
    # Write to MariaDB
    try:
        engine = get_engine_with_retry()
        with engine.begin() as conn:
            dim_demo.to_sql("dim_demographics", con=conn, if_exists="replace", index=False)
            dim_life.to_sql("dim_lifestyle", con=conn, if_exists="replace", index=False)
            dim_med.to_sql("dim_medical", con=conn, if_exists="replace", index=False)
            dim_test.to_sql("dim_tests", con=conn, if_exists="replace", index=False)
            fact_chd.to_sql("fact_chd", con=conn, if_exists="replace", index=False)
            df.to_sql("framingham_raw_full", con=conn, if_exists="replace", index=False)
        logger.info("✅ Wrote star schema tables to MariaDB")
    except Exception as e:
        logger.warning(f"Could not write star schema to MariaDB (continuing): {e}")
    
    # Store for downstream tasks
    try:
        store_df("framingham_star", df)
        logger.info("✅ Stored framingham_star for downstream tasks")
    except Exception as e:
        logger.error(f"Failed to store framingham_star: {e}")
        raise
    
    logger.info("=== STAR SCHEMA BUILD COMPLETED ===")

star_schema_task = PythonOperator(
    task_id="build_star_schema",
    python_callable=build_star_schema,
    dag=dag,
)