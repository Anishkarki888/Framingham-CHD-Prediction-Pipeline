from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Python functions
def read_iris_csv():
    try:
        # Read the Iris CSV file
        df_iris = pd.read_csv("/home/anish/iris.csv", names=['SepLen', 'SepWid', 'PetLen', 'PetWid', 'Species'])
        # Connect to MySQL and write to Iris database
        eng_conn = create_engine("mysql+pymysql://root:anish%40god@localhost:3308/Iris")
        df_iris.to_sql("Iris", con=eng_conn, if_exists="replace", index=False)
        print("Successfully loaded Iris data into MySQL")
    except Exception as e:
        print(f"Error in read_iris_csv: {e}")
        raise
    finally:
        if 'eng_conn' in locals():
            eng_conn.dispose()

def write_iris_csv():
    try:
        # Read from MySQL and write to duplicateiris.csv
        eng_conn = create_engine("mysql+pymysql://root:anish%40god@localhost:3308/Iris")
        df_iris_tmp = pd.read_sql("SELECT * FROM Iris", con=eng_conn)
        df_iris_tmp.to_csv('/home/anish/duplicateiris.csv', index=False)
        print("Successfully wrote Iris data to duplicateiris.csv")
    except Exception as e:
        print(f"Error in write_iris_csv: {e}")
        raise
    finally:
        if 'eng_conn' in locals():
            eng_conn.dispose()

# Default arguments
default_args = {
    "owner": "anishkarki",
    "depends_on_past": False,
    "email": ["anishkarki989@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,  # Allow one retry
    "retry_delay": timedelta(minutes=2)  # Reduced for faster testing
}

# DAG definition
with DAG(
    "example_2",
    default_args=default_args,
    description="A simple example DAG for Iris dataset",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 7, 28),  # Past date for immediate testing
    catchup=False,
    tags=["example"],
) as dag:
    task1 = PythonOperator(
        task_id="get_csv",
        python_callable=read_iris_csv
    )
    task2 = BashOperator(
        task_id="sleep_for_5",
        depends_on_past=False,
        bash_command="sleep 5",
        retries=1,
    )
    task3 = PythonOperator(
        task_id="write_csv",
        python_callable=write_iris_csv
    )
    task1 >> task2 >> task3
