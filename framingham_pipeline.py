import os
import logging
import pickle
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import redis
from sqlalchemy import create_engine
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import optuna
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from prometheus_client import Gauge, start_http_server
from airflow import DAG
from airflow.operators.python import PythonOperator
import great_expectations as ge

# ---------------- Logging setup ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airflow.task")

# ---------------- Metrics server ----------------
METRICS_PORT = 8001
try:
    start_http_server(METRICS_PORT)
    logger.info(f"✅ Prometheus metrics server started on :{METRICS_PORT}")
except Exception as e:
    logger.warning(f"Could not start Prometheus metrics server on :{METRICS_PORT} -> {e}")

# Global Gauges
PROM_GAUGES = {
    "accuracy": Gauge("model_accuracy", "Model Accuracy", ["dag_id"]),
    "precision": Gauge("model_precision", "Model Precision", ["dag_id"]),
    "recall": Gauge("model_recall", "Model Recall", ["dag_id"]),
    "f1": Gauge("model_f1", "Model F1 Score", ["dag_id"]),
    "roc_auc": Gauge("model_roc_auc", "Model ROC-AUC", ["dag_id"]),
    "validation_success": Gauge("data_validation_success", "Whether data validation passed (1=success, 0=failure)", ["dag_id"])
}

# ---------------- DB & Redis ----------------
password = quote_plus("Pa55W0rd123#")
DB_URL = f"mysql+pymysql://root:{password}@127.0.0.1:3308/Framingham"

def get_engine_with_retry(retries=5, delay=5):
    for i in range(retries):
        try:
            engine = create_engine(DB_URL)
            with engine.connect() as conn:
                conn.execute("SELECT 1;")
            logger.info("MariaDB connection successful")
            return engine
        except Exception as e:
            logger.warning(f"MariaDB not ready, retrying {i+1}/{retries} in {delay}s... ({e})")
            time.sleep(delay)
    raise ConnectionError("Cannot connect to MariaDB after multiple retries")

try:
    redis_conn = redis.Redis(host='127.0.0.1', port=6379, socket_connect_timeout=5)
    redis_conn.ping()
    logger.info("Redis connection successful")
except Exception:
    logger.warning("Redis not available, using local storage")
    redis_conn = None

# ---------------- MLflow ----------------
mlflow.set_tracking_uri("sqlite:////home/anish/airflow/dags/mlflow.db")
mlflow.set_experiment("Framingham")

# ---------------- Helper functions ----------------
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle: {path}")

def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    return pickle.load(open(path, 'rb'))

def store_df(key, df):
    os.makedirs('/home/anish/airflow/dags/data', exist_ok=True)
    local_path = f'/home/anish/airflow/dags/data/{key}.pkl'
    save_pickle(df, local_path)

    if redis_conn:
        try:
            table = pa.Table.from_pandas(df)
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf)
            redis_conn.set(key, buf.getvalue().to_pybytes())
            logger.info(f"Stored {key} in Redis")
        except Exception as e:
            logger.warning(f"Failed to store {key} in Redis: {e}")

def load_df(key):
    local_path = f'/home/anish/airflow/dags/data/{key}.pkl'
    if redis_conn:
        try:
            retrieved = redis_conn.get(key)
            if retrieved is not None:
                df = pq.read_table(pa.BufferReader(retrieved)).to_pandas()
                logger.info(f"Loaded {key} from Redis")
                return df
            else:
                logger.info(f"{key} not found in Redis, loading from pickle")
        except Exception as e:
            logger.warning(f"Failed to load {key} from Redis: {e}, falling back to pickle")

    if os.path.exists(local_path):
        df = pd.read_pickle(local_path)
        logger.info(f"Loaded {key} from pickle")
        return df
    else:
        raise FileNotFoundError(f"{key} not found in Redis or pickle at {local_path}")

# ---------------- Tasks ----------------
def data_ingest():
    logger.info("=== STARTING DATA INGEST ===")
    paths = ['/home/anish/framingham.csv', './framingham.csv', '/tmp/framingham.csv', '/home/anish/airflow/dags/framingham.csv']
    df = None
    for p in paths:
        try:
            df = pd.read_csv(p)
            logger.info(f"Loaded dataset from {p}")
            break
        except FileNotFoundError:
            continue
    if df is None:
        raise FileNotFoundError("framingham.csv not found in any specified path")

    df['patient_id'] = range(1, len(df)+1)
    store_df("framingham_raw", df)

    try:
        engine = get_engine_with_retry()
        with engine.connect() as conn:
            df.to_sql('staging_framingham', conn, if_exists='replace', index=False)
        logger.info("Data written to MariaDB")
    except Exception as e:
        logger.warning(f"Failed to write to MariaDB: {e}")

    logger.info("=== DATA INGEST COMPLETED ===")


def preprocess_data():
    logger.info("=== STARTING PREPROCESSING ===")
    
    # Load raw dataset
    df = load_df("framingham_raw")
    
    # 1. Drop rows with missing critical columns
    critical_cols = ["age", "male", "TenYearCHD"]
    df = df.dropna(subset=critical_cols)
    
    # 2. Fill missing numeric values with median (excluding age)
    numeric_cols = ["education", "cigsPerDay", "BPMeds", "totChol",
                    "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Round and clip education after imputation
    if "education" in df.columns:
        df["education"] = df["education"].round().astype(int)
        df["education"] = df["education"].clip(1, 4) 
    
    # 3. Remove invalid ages
    df = df[(df["age"] >= 0) & (df["age"] <= 120)]
    
    # 4. Clip extreme values for certain numeric columns
    for col in ["totChol", "sysBP", "glucose"]:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)
    
    # 5. Scale numeric features excluding 'age'
    scale_cols = [c for c in numeric_cols if c in df.columns and c != "education"]
    scaler = RobustScaler()
    if scale_cols:
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    # 6. Train/test split
    X = df.drop(["TenYearCHD", "patient_id"], axis=1, errors='ignore')
    y = df["TenYearCHD"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 7. Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # 8. Store processed datasets
    store_df("framingham_clean", df)
    store_df("X_train", pd.DataFrame(X_train_sm, columns=X_train.columns))
    store_df("X_test", pd.DataFrame(X_test, columns=X_test.columns))
    store_df("y_train", pd.DataFrame(y_train_sm, columns=["TenYearCHD"]))
    store_df("y_test", pd.DataFrame(y_test, columns=["TenYearCHD"]))
    
    # 9. Save scaler
    save_pickle(scaler, "/home/anish/airflow/dags/models/scaler.pkl")
    
    logger.info(f"=== PREPROCESSING COMPLETED | Cleaned dataset shape: {df.shape} ===")



def validate_data(**kwargs):
    logger.info("=== STARTING DATA VALIDATION ===")
    try:
        df = load_df("framingham_clean")

        if df.empty:
            PROM_GAUGES["validation_success"].labels(dag_id=kwargs['dag'].dag_id).set(0)
            raise ValueError("Preprocessed dataset is empty")

        ge_df = ge.dataset.PandasDataset(df)

        # Expect no nulls
        for col in df.columns:
            ge_df.expect_column_values_to_not_be_null(col)

        # Column-specific expectations
        if "age" in df.columns:
            # Keep as-is, check realistic range only
            ge_df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
        if "TenYearCHD" in df.columns:
            ge_df.expect_column_values_to_be_in_set("TenYearCHD", [0, 1])
        if "male" in df.columns:
            ge_df.expect_column_values_to_be_in_set("male", [0, 1])
        if "education" in df.columns:
            ge_df.expect_column_values_to_be_between("education", min_value=1, max_value=4)

        # Run validation
        validation_result = ge_df.validate()

        success_flag = 1 if validation_result["success"] else 0
        PROM_GAUGES["validation_success"].labels(dag_id=kwargs['dag'].dag_id).set(success_flag)

        if not validation_result["success"]:
            raise ValueError(f"Data validation failed: {validation_result}")

        logger.info("=== DATA VALIDATION COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        PROM_GAUGES["validation_success"].labels(dag_id=kwargs['dag'].dag_id).set(0)
        logger.error(f"Validation error: {e}")
        raise



def train_and_evaluate(**kwargs):
    logger.info("=== STARTING TRAINING ===")
    X_train = load_df("X_train")
    y_train = load_df("y_train").values.ravel()
    X_test = load_df("X_test")
    y_test = load_df("y_test").values.ravel()
    model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42, class_weights=[1, 3000/900])
    with mlflow.start_run(run_name="catboost_default"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        for metric_name, value in metrics.items():
            PROM_GAUGES[metric_name].labels(dag_id=kwargs['dag'].dag_id).set(value)
        mlflow.log_metrics(metrics)
        params = {'iterations': 500, 'learning_rate': 0.1, 'depth': 6, 'random_state': 42, 'class_weights': [1, 3000/900]}
        mlflow.log_params(params)
        try:
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "catboost_model", signature=signature, input_example=X_train.iloc[:1])
        except Exception as e:
            logger.warning(f"Could not infer signature: {e}")
            mlflow.sklearn.log_model(model, "catboost_model", input_example=X_train.iloc[:1])
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        save_pickle(model, '/home/anish/airflow/dags/models/catboost_model.pkl')
    logger.info("=== TRAINING COMPLETED ===")

def hyperparameter_tuning():
    logger.info("=== STARTING HYPERPARAM TUNING ===")
    X_train = load_df("X_train")
    y_train = load_df("y_train").values.ravel()
    def objective(trial):
        params = {
            'iterations': 100,
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': 0,
            'random_state': 42
        }
        model = CatBoostClassifier(**params, class_weights=[1, 3000/900])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=1).mean()
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("roc_auc", float(score))
        return score
    with mlflow.start_run(run_name="catboost_optuna"):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        store_df("best_catboost_params", pd.DataFrame([best_params]))
        mlflow.log_params(best_params)
        mlflow.log_metric("best_roc_auc", float(study.best_value))
        logger.info(f"=== HYPERPARAM TUNING COMPLETED, Best ROC-AUC={study.best_value:.4f} ===")

def final_model_training(**kwargs):
    logger.info("=== STARTING FINAL MODEL TRAINING ===")
    best_params_df = load_df("best_catboost_params")
    best_params = best_params_df.iloc[0].to_dict()
    X_train = load_df("X_train")
    y_train = load_df("y_train").values.ravel()
    X_test = load_df("X_test")
    y_test = load_df("y_test").values.ravel()
    with mlflow.start_run(run_name="final_catboost_model"):
        model = CatBoostClassifier(**best_params, class_weights=[1, 3000/900])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        for metric_name, value in metrics.items():
            PROM_GAUGES[metric_name].labels(dag_id=kwargs['dag'].dag_id).set(value)
        mlflow.log_metrics(metrics)
        mlflow.log_params(best_params)
        try:
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "catboost_best_model", signature=signature, input_example=X_train.iloc[:1])
        except Exception as e:
            logger.warning(f"Could not infer signature for final model: {e}")
            mlflow.sklearn.log_model(model, "catboost_best_model", input_example=X_train.iloc[:1])
        save_pickle(model, '/home/anish/airflow/dags/models/best_catboost_model.pkl')
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logger.info("=== FINAL MODEL TRAINING COMPLETED ===")

def deploy_model():
    logger.info("=== STARTING DEPLOYMENT ===")
    os.makedirs('/home/anish/airflow/dags/models', exist_ok=True)
    model = load_pickle('/home/anish/airflow/dags/models/best_catboost_model.pkl')
    scaler = load_pickle('/home/anish/airflow/dags/models/scaler.pkl') if os.path.exists('/home/anish/airflow/dags/models/scaler.pkl') else None
    pipeline = {'model': model, 'scaler': scaler}
    save_pickle(pipeline, '/home/anish/airflow/dags/models/final_pipeline.pkl')
    if redis_conn:
        try:
            redis_conn.set("deployed_model_status", "deployed")
        except Exception as e:
            logger.warning(f"Redis write failed: {e}")
    else:
        with open('/home/anish/airflow/dags/data/deployment_status.txt', 'w') as f:
            f.write("deployed")
    logger.info("=== DEPLOYMENT COMPLETED ===")

# ---------------- DAG definition ----------------
default_args = {
    'owner': 'Anish',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    "framingham_pipeline",
    default_args=default_args,
    description="Framingham MLOps pipeline",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

# ---------------- Operators ----------------
ingest_task = PythonOperator(task_id="data_ingest", python_callable=data_ingest, dag=dag)
preprocess_task = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data, dag=dag)
validate_task = PythonOperator(task_id="validate_data", python_callable=validate_data, provide_context=True, dag=dag)
train_task = PythonOperator(task_id="train_and_evaluate", python_callable=train_and_evaluate, provide_context=True, dag=dag)
tune_task = PythonOperator(task_id="hyperparameter_tuning", python_callable=hyperparameter_tuning, dag=dag)
final_train_task = PythonOperator(task_id="final_model_training", python_callable=final_model_training, provide_context=True, dag=dag)
deploy_task = PythonOperator(task_id="deploy_model", python_callable=deploy_model, dag=dag)

# ---------------- Dependencies ----------------
ingest_task >> preprocess_task >> validate_task >> train_task >> tune_task >> final_train_task >> deploy_task
