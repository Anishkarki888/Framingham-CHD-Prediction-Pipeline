from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.python import PythonOperator

def export_heart_disease_to_csv():
    try:
        engine = create_engine("mysql+pymysql://root:anish%40god@localhost:3308/Cardiology")
        df = pd.read_sql("SELECT * FROM HeartDisease", con=engine)
        df.to_csv('/home/anish/heart_cleveland.csv', index=False)
        print("HeartDisease data exported successfully!")
    except Exception as e:
        print(f"Error exporting HeartDisease data: {e}")
        raise
    finally:
        if 'engine' in locals():
            engine.dispose()

default_args = {
    "owner": "anishkarki",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "start_date": datetime(2025, 7, 28),  # Use recent past date to trigger immediately
}

with DAG(
    "export_heart_disease_dag",
    default_args=default_args,
    description="Export HeartDisease table to CSV daily",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["example"],
) as dag:

    export_task = PythonOperator(
        task_id="export_heart_disease",
        python_callable=export_heart_disease_to_csv
    )
