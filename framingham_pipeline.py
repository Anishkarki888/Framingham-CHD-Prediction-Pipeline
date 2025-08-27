#!/usr/bin/env python3
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





# ---------------- Logging setup ----------------
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airflow.task")


# ---------------- Metrics server ----------------
METRICS_PORT = 8001  # or whatever you used
try:
    start_http_server(METRICS_PORT)
    logger.info(f"✅ Prometheus metrics server started on :{METRICS_PORT}")
except Exception as e:
    logger.warning(f"Could not start Prometheus metrics server on :{METRICS_PORT} -> {e}")


# Global Gauges (keys must match metrics computed later)
PROM_GAUGES = {
    "accuracy": Gauge("model_accuracy", "Model Accuracy", ["dag_id"]),
    "precision": Gauge("model_precision", "Model Precision", ["dag_id"]),
    "recall": Gauge("model_recall", "Model Recall", ["dag_id"]),
    "f1": Gauge("model_f1", "Model F1 Score", ["dag_id"]),
    "roc_auc": Gauge("model_roc_auc", "Model ROC-AUC", ["dag_id"]),
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
    return pickle.load(open(path, 'rb'))

def store_df(key, df):
    """
    Stores df to Redis (pyarrow parquet bytes) if available, otherwise to local pickle.
    Key is a short name (no extension).
    """
    if redis_conn:
        table = pa.Table.from_pandas(df)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)
        redis_conn.set(key, buf.getvalue().to_pybytes())
    else:
        os.makedirs('/home/anish/airflow/dags/data', exist_ok=True)
        df.to_pickle(f'/home/anish/airflow/dags/data/{key}.pkl')

def load_df(key):
    """
    Loads a dataframe from Redis or local storage. Raises informative errors.
    """
    try:
        if redis_conn:
            retrieved = redis_conn.get(key)
            if retrieved is None:
                raise ValueError(f"No data for {key} in Redis")
            return pq.read_table(pa.BufferReader(retrieved)).to_pandas()
        else:
            return pd.read_pickle(f'/home/anish/airflow/dags/data/{key}.pkl')
    except Exception as e:
        logger.warning(f"Failed loading {key} from Redis: {e}, fallback to local")
        return pd.read_pickle(f'/home/anish/airflow/dags/data/{key}.pkl')

# ---------------- Tasks ----------------
def data_ingest():
    logger.info("=== STARTING DATA INGEST ===")
    paths = ['/home/anish/framingham.csv','./framingham.csv','/tmp/framingham.csv','/opt/airflow/dags/framingham.csv']
    df = None
    for p in paths:
        try:
            df = pd.read_csv(p)
            logger.info(f"Loaded dataset from {p}")
            break
        except FileNotFoundError:
            continue
    if df is None:
        raise FileNotFoundError("framingham.csv not found")

    df['patient_id'] = range(1, len(df)+1)
    store_df("framingham_raw", df)
    save_pickle(df, '/home/anish/airflow/dags/data/framingham_raw.pkl')

    # Try writing to MariaDB (best-effort)
    try:
        engine = get_engine_with_retry()
        with engine.connect() as conn:
            df.to_sql('staging_framingham', conn, if_exists='replace', index=False)
            # modular writes to smaller tables for convenience
            if set(['patient_id','male','age','education']).issubset(df.columns):
                df[['patient_id','male','age','education']].to_sql('demographics', conn, if_exists='replace', index=False)
            if set(['patient_id','currentSmoker','cigsPerDay']).issubset(df.columns):
                df[['patient_id','currentSmoker','cigsPerDay']].to_sql('lifestyle', conn, if_exists='replace', index=False)
            if set(['patient_id','BPMeds','prevalentStroke','prevalentHyp','diabetes']).issubset(df.columns):
                df[['patient_id','BPMeds','prevalentStroke','prevalentHyp','diabetes']].to_sql('medical_history', conn, if_exists='replace', index=False)
            if set(['patient_id','totChol','sysBP','diaBP','BMI','heartRate','glucose','TenYearCHD']).issubset(df.columns):
                df[['patient_id','totChol','sysBP','diaBP','BMI','heartRate','glucose','TenYearCHD']].to_sql('patient_records', conn, if_exists='replace', index=False)
        logger.info("Data written to MariaDB")
    except Exception as e:
        logger.warning(f"Failed to write to MariaDB: {e}")

    logger.info("=== DATA INGEST COMPLETED ===")

def preprocess_data():
    logger.info("=== STARTING PREPROCESSING ===")
    df = load_df("framingham_raw")
    # basic cleaning
    df = df.dropna()
    for col in ['totChol','sysBP','glucose']:
        if col in df.columns:
            df[col] = df[col].clip(upper=df[col].quantile(0.99))

    numeric_cols = ['age','education','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    scaler = RobustScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop(["TenYearCHD","patient_id"], axis=1, errors='ignore')
    y = df["TenYearCHD"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # handle imbalance
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # store
    store_df("X_train", pd.DataFrame(X_train_sm, columns=X_train.columns))
    store_df("X_test", pd.DataFrame(X_test, columns=X_test.columns))
    store_df("y_train", pd.DataFrame(y_train_sm, columns=["TenYearCHD"]))
    store_df("y_test", pd.DataFrame(y_test.values, columns=["TenYearCHD"]))

    save_pickle(scaler, '/home/anish/airflow/dags/models/scaler.pkl')
    logger.info("=== PREPROCESSING COMPLETED ===")

def train_and_evaluate():
    logger.info("=== STARTING TRAINING ===")
    X_train = load_df("X_train")
    y_train = load_df("y_train").values.ravel()
    X_test = load_df("X_test")
    y_test = load_df("y_test").values.ravel()

    # CatBoost with class weights heuristics you used
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

        mlflow.log_metrics(metrics)
        params = {'iterations': 500,'learning_rate': 0.1,'depth': 6,'random_state': 42,'class_weights': [1, 3000/900]}
        mlflow.log_params(params)

        # signature - input_example must be a DataFrame; using a single-row example
        try:
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "catboost_model", signature=signature, input_example=X_train.iloc[:1])
        except Exception as e:
            # fallback to logging model without signature if inference fails
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
        # store best params as df for consistency with your load/save functions
        store_df("best_catboost_params", pd.DataFrame([best_params]))
        mlflow.log_params(best_params)
        mlflow.log_metric("best_roc_auc", float(study.best_value))
        logger.info(f"=== HYPERPARAM TUNING COMPLETED, Best ROC-AUC={study.best_value:.4f} ===")

def final_model_training():
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
        with open('/home/anish/airflow/dags/data/deployment_status.txt','w') as f:
            f.write("deployed")
    logger.info("=== DEPLOYMENT COMPLETED ===")

def monitor_model(**kwargs):
    """
    Airflow task: monitors the model by computing metrics, logging to MLflow,
    pushing Prometheus metrics, and saving an Evidently report.
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import mlflow
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from airflow.utils.log.logging_mixin import LoggingMixin

    logger = LoggingMixin().log

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:////home/anish/airflow/dags/mlflow.db")
    mlflow.set_experiment("Framingham")

    # Load trained pipeline
    pipeline_path = '/home/anish/airflow/dags/models/final_pipeline.pkl'
    if not os.path.exists(pipeline_path):
        logger.error(f"Pipeline file not found: {pipeline_path}")
        raise FileNotFoundError(pipeline_path)

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    model = pipeline.get("model")
    scaler = pipeline.get("scaler")

    if model is None:
        logger.error("No 'model' found in final_pipeline.pkl")
        raise ValueError("Missing model in final_pipeline.pkl")

    # Load datasets from Redis/parquet
    X_train = load_df("X_train")
    X_test = load_df("X_test")
    y_train_df = load_df("y_train")
    y_test_df = load_df("y_test")

    y_train = y_train_df.values.ravel() if hasattr(y_train_df, "values") else np.array(y_train_df)
    y_test = y_test_df.values.ravel() if hasattr(y_test_df, "values") else np.array(y_test_df)

    # Make predictions
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    try:
        pred_proba_test = model.predict_proba(X_test)[:, 1]
    except Exception:
        pred_proba_test = np.zeros_like(preds_test, dtype=float)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds_test)),
        "precision": float(precision_score(y_test, preds_test, zero_division=0)),
        "recall": float(recall_score(y_test, preds_test, zero_division=0)),
        "f1": float(f1_score(y_test, preds_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, pred_proba_test)) if pred_proba_test.sum() != 0 else 0.0
    }

    # Log metrics to MLflow
    with mlflow.start_run(run_name="model_monitoring"):
        mlflow.log_metrics(metrics)
        logger.info(f"Monitoring Metrics: {metrics}")
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

    # Push metrics to Prometheus gauges if defined
    for name, value in metrics.items():
        if name in PROM_GAUGES:
            try:
                PROM_GAUGES[name].labels(dag_id="framingham_pipeline").set(value)
            except Exception as e:
                logger.warning(f"Failed to set Prometheus metric {name}: {e}")
        logger.info(f"Set Prometheus metric {name}: {value}")

    # Prepare Evidently input frames
    X_train_with_target = X_train.copy()
    X_train_with_target["target"] = y_train
    X_train_with_target["prediction"] = preds_train

    X_test_with_target = X_test.copy()
    X_test_with_target["target"] = y_test
    X_test_with_target["prediction"] = preds_test

    # Generate Evidently report (v0.4+ compatible)
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=X_train_with_target, current_data=X_test_with_target)

        os.makedirs('/home/anish/airflow/dags/reports', exist_ok=True)
        report_path = f"/home/anish/airflow/dags/reports/evidently_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(report_path)
        logger.info(f"✅ Evidently report saved at {report_path}")
    except Exception as e:
        logger.warning(f"Failed to generate Evidently report: {e}")

    # Warn if ROC-AUC below threshold but do not fail DAG
    threshold = 0.70
    if metrics.get("roc_auc", 0.0) < threshold:
        logger.warning(f"⚠️ Model performance below threshold ({threshold}) – ROC-AUC={metrics['roc_auc']:.4f}")
    else:
        logger.info("✅ Model performance is satisfactory")

    logger.info("=== MONITORING COMPLETED ===")


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
    start_date=datetime(2025,8,24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

# ---------------- Operators ----------------
ingest_task = PythonOperator(task_id="data_ingest", python_callable=data_ingest, dag=dag)
preprocess_task = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data, dag=dag)
train_task = PythonOperator(task_id="train_and_evaluate", python_callable=train_and_evaluate, dag=dag)
tune_task = PythonOperator(task_id="hyperparameter_tuning", python_callable=hyperparameter_tuning, dag=dag)
final_train_task = PythonOperator(task_id="final_model_training", python_callable=final_model_training, dag=dag)
deploy_task = PythonOperator(task_id="deploy_model", python_callable=deploy_model, dag=dag)
monitor_task = PythonOperator(task_id="monitor_model", python_callable=monitor_model, dag=dag)

# ---------------- Dependencies ----------------
ingest_task >> preprocess_task >> train_task >> tune_task >> final_train_task >> deploy_task >> monitor_task
