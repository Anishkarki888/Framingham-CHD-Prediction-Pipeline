from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import load_df, store_df, save_pickle, logger, MODEL_DIR
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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
    "preprocessing",
    default_args=default_args,
    description="Preprocesses Framingham dataset for ML",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

def preprocess_data(**kwargs):
    logger.info("=== STARTING PREPROCESSING (ML) ===")
    df = load_df("framingham_raw")

    # Clean and impute
    df = df.dropna(subset=["age", "male", "TenYearCHD"])
    numeric_cols = ["education", "cigsPerDay", "BPMeds", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
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

    # Split and balance
    X = df.drop(["TenYearCHD", "patient_id"], axis=1, errors='ignore')
    y = df["TenYearCHD"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Store processed data
    store_df("framingham_clean", df)
    store_df("X_train", pd.DataFrame(X_train_sm, columns=X_train.columns))
    store_df("X_test", pd.DataFrame(X_test, columns=X_test.columns))
    store_df("y_train", pd.DataFrame(y_train_sm, columns=["TenYearCHD"]))
    store_df("y_test", pd.DataFrame(y_test, columns=["TenYearCHD"]))
    save_pickle(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    logger.info("=== PREPROCESSING COMPLETED | Cleaned dataset shape: %s ===", df.shape)

preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)