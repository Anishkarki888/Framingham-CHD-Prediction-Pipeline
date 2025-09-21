from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import load_pickle, save_pickle, redis_conn, logger, MODEL_DIR, DATA_DIR
import os

default_args = {
    'owner': 'Anish',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    "model_deploy",
    default_args=default_args,
    description="Deploys trained CatBoost model for Framingham pipeline",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

def deploy_model(**kwargs):
    logger.info("=== STARTING DEPLOYMENT ===")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'best_catboost_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    model = load_pickle(model_path)
    scaler = load_pickle(scaler_path) if os.path.exists(scaler_path) else None
    pipeline = {'model': model, 'scaler': scaler}
    save_pickle(pipeline, os.path.join(MODEL_DIR, 'final_pipeline.pkl'))

    if redis_conn:
        try:
            redis_conn.set("deployed_model_status", b"deployed")
            logger.info("Deployment status updated in Redis")
        except Exception as e:
            logger.warning("Redis write failed: %s", e)
    else:
        with open(os.path.join(DATA_DIR, "deployment_status.txt"), "w") as f:
            f.write("deployed")
        logger.info("Deployment status written to local file")

    logger.info("=== DEPLOYMENT COMPLETED ===")

deploy_task = PythonOperator(
    task_id="deploy_model",
    python_callable=deploy_model,
    dag=dag,
)