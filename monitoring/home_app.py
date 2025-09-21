import subprocess
import threading
import webbrowser
import streamlit as st
import os
import time
import socket
import psutil
import pickle
from datetime import datetime

# ------------------------------
# Paths and Ports
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMLIT_DIR = os.path.join(BASE_DIR, "streamlit_app")
APP_PATH = os.path.join(STREAMLIT_DIR, "app.py")
DATA_DRIFT_PATH = os.path.join(STREAMLIT_DIR, "data_drift.py")
CONCEPT_DRIFT_PATH = os.path.join(STREAMLIT_DIR, "concept_drift.py")
NEW_DATA_FILE = os.path.join(BASE_DIR, "data/new_patient_data.csv")

MLFLOW_DB = os.path.join(BASE_DIR, "mlflow/mlflow.db")
AIRFLOW_ENV = "myfirstenvironment"
MLFLOW_PORT = 5000
APP_PORT = 8503
DATA_DRIFT_PORT = 8504
CONCEPT_DRIFT_PORT = 8505
CONCEPT_DRIFT_THRESHOLD = 0.5

DATA_DRIFT_PKL = os.path.join(STREAMLIT_DIR, "data/data_drift_results.pkl")
CONCEPT_DRIFT_PKL = os.path.join(STREAMLIT_DIR, "data/concept_drift_results.pkl")

# ------------------------------
# Utility Functions
# ------------------------------
def wait_for_port(port, host="localhost", timeout=30):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            time.sleep(1)
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Port {port} not open after {timeout}s")

def is_port_in_use(port):
    return any(conn.laddr.port == port for conn in psutil.net_connections())

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def run_streamlit_app(path, port, save_pkl=None, pkl_data=None):
    """Run Streamlit app and optionally save pkl data"""
    def target():
        if save_pkl and pkl_data:
            save_pickle(pkl_data, save_pkl)
        if not is_port_in_use(port):
            subprocess.Popen([
                os.sys.executable, "-m", "streamlit", "run", path,
                "--server.port", str(port),
                "--server.headless=true"
            ], cwd=os.path.dirname(path))
            try:
                wait_for_port(port, timeout=60)
            except TimeoutError as e:
                st.error(str(e))
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=target, daemon=True).start()

def run_mlflow():
    """Run MLflow UI"""
    def target():
        if not is_port_in_use(MLFLOW_PORT):
            subprocess.Popen([
                os.sys.executable, "-m", "mlflow", "ui",
                "--backend-store-uri", f"sqlite:///{MLFLOW_DB}",
                "--port", str(MLFLOW_PORT)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                wait_for_port(MLFLOW_PORT, timeout=60)
            except TimeoutError as e:
                st.error(f"MLflow failed to start: {e}")
        webbrowser.open(f"http://localhost:{MLFLOW_PORT}")
    threading.Thread(target=target, daemon=True).start()

# ------------------------------
# Airflow Functions
# ------------------------------
def trigger_airflow_dag(dag_id="framingham_mlops_pipeline"):
    """Trigger Airflow DAG"""
    result = subprocess.run([
        "conda", "run", "-n", AIRFLOW_ENV,
        "airflow", "dags", "trigger", dag_id
    ], capture_output=True, text=True)

    if result.returncode == 0:
        st.success(f"DAG '{dag_id}' triggered successfully!")
    else:
        st.error(f"Failed to trigger DAG: {result.stderr}")

# ------------------------------
# Streamlit Dashboard
# ------------------------------
st.set_page_config(page_title="Framingham MLOps Dashboard", layout="wide")
st.title("🏥 Framingham MLOps Dashboard")
st.markdown("### Quick Launch")

col1, col2 = st.columns(2)

# Sample placeholder data for pkl saving
sample_data_drift = {"age": 0.1, "cholesterol": 0.2, "bp": 0.05}
sample_concept_drift = 0.87

with col1:
    if st.button("🔄 Trigger Main DAG"):
        trigger_airflow_dag()
    if st.button("🧑 User Input App"):
        run_streamlit_app(APP_PATH, APP_PORT)
    if st.button("📊 Data Drift Dashboard"):
        run_streamlit_app(DATA_DRIFT_PATH, DATA_DRIFT_PORT, save_pkl=DATA_DRIFT_PKL, pkl_data=sample_data_drift)
    if st.button("📈 Concept Drift Dashboard"):
        run_streamlit_app(CONCEPT_DRIFT_PATH, CONCEPT_DRIFT_PORT, save_pkl=CONCEPT_DRIFT_PKL, pkl_data=sample_concept_drift)

with col2:
    if st.button("🖥️ MLflow UI"):
        run_mlflow()

st.info(f"""
Ports:
- User App: {APP_PORT}
- Data Drift: {DATA_DRIFT_PORT}
- Concept Drift: {CONCEPT_DRIFT_PORT}
- MLflow: {MLFLOW_PORT}
""")
