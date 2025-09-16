from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import (
    load_df, store_df, save_pickle,
    logger, MODEL_DIR, make_redis_client
)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import os

default_args = {
    "owner": "Anish",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "preprocessing_redis",
    default_args=default_args,
    description="Preprocess Framingham dataset and store cleaned data in Redis",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False,
)


def preprocess_data(**kwargs):
    logger.info("STARTING DATA PREPROCESSING")

    # Initialize Redis client
    make_redis_client()

    # Load raw data
    df = load_df("framingham_raw", name="raw dataframe")

  
    # Data cleaning and preprocessing
    df = df.dropna(subset=["age", "male", "TenYearCHD"])

    numeric_cols = [
        "education", "cigsPerDay", "BPMeds",
        "totChol", "sysBP", "diaBP",
        "BMI", "heartRate", "glucose"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "education" in df.columns:
        df["education"] = df["education"].round().astype(int).clip(1, 4)

    df = df[(df["age"] >= 0) & (df["age"] <= 120)]

    for col in ["totChol", "sysBP", "glucose"]:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)

    # Scale numeric columns
    scale_cols = [c for c in numeric_cols if c in df.columns and c != "education"]
    scaler = RobustScaler()
    if scale_cols:
        df[scale_cols] = scaler.fit_transform(df[scale_cols])


    # Train/Test split
    X = df.drop(["TenYearCHD", "patient_id"], axis=1, errors="ignore")
    y = df["TenYearCHD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

 
    # Store processed data in Redis + local
    store_df("framingham_clean", df)
    store_df("X_train", X_train)
    store_df("X_test", X_test)
    store_df("y_train", y_train)  
    store_df("y_test", y_test)

    # Save scaler
    save_pickle(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    logger.info(f"PREPROCESSING COMPLETED | Cleaned dataset shape: {df.shape}")


preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)
