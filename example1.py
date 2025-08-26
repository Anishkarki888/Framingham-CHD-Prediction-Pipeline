from datetime import datetime, timedelta
from textwrap import dedent
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "Anish",
    "depends_on_past": False,
    "email": ["anish.karki@mail.bcu.ac.uk"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    "example1",
    default_args=default_args,
    description="A simple example DAG",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 7, 24),
    catchup=False,
    tags=["example"],
) as dag:
    task1 = BashOperator(
        task_id="date_to_file",
        bash_command="date > ~/currentdate.txt"
    )
    task2 = BashOperator(
        task_id="sleep_for_5",
        depends_on_past=False,
        bash_command="sleep 5",
        retries=3,
    )
    task3 = BashOperator(
        task_id="join_passwd_and_currentdate",
        bash_command="cat /etc/passwd ~/currentdate.txt >> ~/exampledag1result.txt"
    )
    task1 >> [task2, task3]
