# db_setup.py
from utils import load_df, store_df, get_engine_with_retry, logger, make_redis_client
import pandas as pd
import os

def build_star_schema():
    logger.info("=== STARTING STAR SCHEMA BUILD ===")

    # Initialize Redis client
    make_redis_client()

    # Try loading data from Redis first
    df = None
    try:
        df = load_df("framingham_raw")
        if df is not None:
            logger.info(f"✅ Loaded data from Redis: {df.shape}")
    except Exception as e:
        logger.warning(f"Redis load failed: {e}, trying local pickle/CSV")

    # Fallbacks if Redis is empty
    if df is None:
        pickle_path = "/home/anish/airflow/dags/monitoring/data/framingham_raw.pkl"
        csv_path = "/home/anish/airflow/dags/framingham.csv"

        if os.path.exists(pickle_path):
            df = pd.read_pickle(pickle_path)
            logger.info(f"✅ Loaded data from pickle fallback: {df.shape}")
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['patient_id'] = range(1, len(df) + 1)
            logger.info(f"✅ Loaded data from CSV fallback: {df.shape}")
        else:
            raise ValueError("❌ Could not load framingham_raw data from any source")

    # Add integer patient_id if missing
    if 'patient_id' not in df.columns:
        df['patient_id'] = range(1, len(df) + 1)
    df["patient_id_int"] = df["patient_id"].astype(int)

    # Create dimension and fact tables
    dim_demo = df[["patient_id_int", "male", "age", "education"]].drop_duplicates().rename(columns={"patient_id_int": "patient_id"})
    dim_life = df[["patient_id_int", "currentSmoker", "cigsPerDay"]].drop_duplicates().rename(columns={"patient_id_int": "patient_id", "currentSmoker": "current_smoker"})

    medical_cols = ["patient_id_int", "BPMeds"]
    for col in ["prevalentStroke", "prevalentHyp", "diabetes"]:
        if col in df.columns:
            medical_cols.append(col)
    dim_med = df[medical_cols].drop_duplicates().rename(columns={"patient_id_int": "patient_id"})

    test_cols = [c for c in ["patient_id_int", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"] if c in df.columns]
    dim_test = df[test_cols].drop_duplicates().rename(columns={"patient_id_int": "patient_id"})

    fact_chd = df[["patient_id", "TenYearCHD"]].copy()

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
        logger.warning(f"Could not write star schema to MariaDB: {e}")

    # Store star schema in Redis for downstream tasks
    store_df("framingham_star", df)
    logger.info("✅ Stored framingham_star in Redis for downstream tasks")

    logger.info("=== STAR SCHEMA BUILD COMPLETED ===")

if __name__ == "__main__":
    build_star_schema()
