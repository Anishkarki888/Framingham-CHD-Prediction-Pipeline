#!/bin/bash
echo "🚀 Starting all services..."

# Activate environment
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myfirstenvironment || true
else
  source /home/anish/myfirstenvironment/bin/activate || true
fi

# MariaDB
docker volume create mariadb_data >/dev/null || true
if ! docker ps -q -f name=final_mariadb_v3 >/dev/null 2>&1; then
    if docker ps -aq -f name=final_mariadb_v3 >/dev/null 2>&1; then
        docker start final_mariadb_v3 || true
    else
        docker run -d --name final_mariadb_v3 -e MYSQL_ROOT_PASSWORD='Pa55W0rd123#' -e MYSQL_DATABASE=Framingham -v mariadb_data:/var/lib/mysql -p 3308:3306 mariadb:10.6
    fi
fi

# Redis
if ! docker ps -q -f name=redis_framingham >/dev/null 2>&1; then
    docker run -d --name redis_framingham -p 6379:6379 redis:7-alpine || true
fi

# MLflow
nohup mlflow ui --backend-store-uri sqlite:////home/anish/airflow/dags/monitoring/mlflow/mlflow.db --host 0.0.0.0 --port 5050 >/home/anish/airflow/dags/monitoring/mlflow/mlflow_ui.log 2>&1 &

# Streamlit
APP=/home/anish/airflow/dags/monitoring/streamlit_app/app.py
if [ -f "$APP" ]; then
  nohup streamlit run "$APP" --server.port 8501 --server.address 0.0.0.0 >/home/anish/airflow/dags/monitoring/streamlit.log 2>&1 &
fi

echo "✅ All services attempted"
