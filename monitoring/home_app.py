import streamlit as st
import subprocess
import os
import psutil
import json
import webbrowser
import time
import socket

# ======================
# Config
# ======================
PID_FILE = "service_pids.json"
AIRFLOW_ENV = "myfirstenvironment"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STREAMLIT_DIR = os.path.join(BASE_DIR, "streamlit_app")
APP_PATH = os.path.join(STREAMLIT_DIR, "app.py")
DATA_DRIFT_PATH = os.path.join(STREAMLIT_DIR, "data_drift.py")
CONCEPT_DRIFT_PATH = os.path.join(STREAMLIT_DIR, "concept_drift.py")
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow/mlflow.db")

PORTS = {
    "Airflow Webserver": 8080,
    "User App": 8503,
    "Data Drift": 8504,
    "Concept Drift": 8505,
    "MLflow": 5000,
}

# ======================
# Load PID state
# ======================
if os.path.exists(PID_FILE):
    with open(PID_FILE, "r") as f:
        saved_pids = json.load(f)
else:
    saved_pids = {}

if "processes" not in st.session_state:
    st.session_state.processes = {}

for service_name, pid in saved_pids.items():
    if psutil.pid_exists(pid):
        st.session_state.processes[service_name] = pid


def save_pids():
    with open(PID_FILE, "w") as f:
        json.dump(st.session_state.processes, f)


# ======================
# Utilities
# ======================
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


def run_service(service_name, command, cwd=None, env_name=AIRFLOW_ENV, open_browser=False, port=None):
    try:
        full_command = f"conda run -n {env_name} {command}"
        process = subprocess.Popen(full_command, shell=True, cwd=cwd)
        st.session_state.processes[service_name] = process.pid
        save_pids()

        if port:
            try:
                wait_for_port(port, timeout=60)
            except TimeoutError as e:
                st.error(str(e))

        if open_browser and port:
            webbrowser.open(f"http://localhost:{port}")

        st.success(f"{service_name} started")
    except Exception as e:
        st.error(f"Error starting {service_name}: {e}")


def stop_service(service_name):
    if service_name not in st.session_state.processes:
        return
    try:
        pid = st.session_state.processes[service_name]
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        del st.session_state.processes[service_name]
        save_pids()
        st.success(f"Stopped {service_name}")
    except Exception as e:
        st.error(f"Error stopping {service_name}: {e}")


def is_running(service_name):
    if service_name not in st.session_state.processes:
        return False
    pid = st.session_state.processes[service_name]
    return psutil.pid_exists(pid)


def trigger_dag_cli(dag_id):
    """Trigger Airflow DAG using CLI (avoids REST API auth issues)"""
    try:
        subprocess.run(
            f"conda run -n {AIRFLOW_ENV} airflow dags trigger {dag_id}",
            shell=True,
            check=True,
        )
        st.success(f"Triggered DAG: {dag_id}")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to trigger DAG {dag_id}: {e}")


# ======================
# Services
# ======================
SERVICES = {
    "Airflow Scheduler": {
        "command": "airflow scheduler",
        "dir": None,
        "port": None,
        "browser": False,
    },
    "Airflow Webserver": {
        "command": f"airflow webserver --port {PORTS['Airflow Webserver']}",
        "dir": None,
        "port": PORTS["Airflow Webserver"],
        "browser": True,
    },
    "User App": {
        "command": f"streamlit run {APP_PATH} --server.port {PORTS['User App']}",
        "dir": os.path.dirname(APP_PATH),
        "port": PORTS["User App"],
        "browser": True,
    },
    "Data Drift": {
        "command": f"streamlit run {DATA_DRIFT_PATH} --server.port {PORTS['Data Drift']}",
        "dir": os.path.dirname(DATA_DRIFT_PATH),
        "port": PORTS["Data Drift"],
        "browser": True,
    },
    "Concept Drift": {
        "command": f"streamlit run {CONCEPT_DRIFT_PATH} --server.port {PORTS['Concept Drift']}",
        "dir": os.path.dirname(CONCEPT_DRIFT_PATH),
        "port": PORTS["Concept Drift"],
        "browser": True,
    },
    "MLflow": {
        "command": f"mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB} --port {PORTS['MLflow']}",
        "dir": None,
        "port": PORTS["MLflow"],
        "browser": True,
    },
}

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="MLOps Project - Anish Karki", layout="wide")

# ---- Heading ----
st.markdown(
    """
    <div style="text-align: center; background-color:#E3F2FD;
                padding:25px; border-radius:12px; margin-bottom:20px;">
        <h1 style="color:#0D47A1;">❤️ MLOps Project</h1>
        <h3 style="color:#1565C0;">Code Painted by <b>Anish Karki</b></h3>
        <p style="color:#0D47A1;">Orchestrating ML pipeline, deployment and monitoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---- Service Controls ----
cols = st.columns(3)

for i, (service_name, config) in enumerate(SERVICES.items()):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div style="background-color:#BBDEFB;padding:20px;border-radius:12px;margin-bottom:15px;">
                <h4 style="color:#0D47A1;">⭐ {service_name}</h4>
            """,
            unsafe_allow_html=True,
        )

        run_key = f"run_{service_name.replace(' ', '_')}"
        stop_key = f"stop_{service_name.replace(' ', '_')}"

        if is_running(service_name):
            st.success("Running ✅")
            if st.button("Stop", key=stop_key):
                stop_service(service_name)
        else:
            st.info("Stopped ❌")
            if st.button("Start", key=run_key):
                run_service(
                    service_name,
                    config["command"],
                    cwd=config["dir"],
                    port=config["port"],
                    open_browser=config["browser"],
                )

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---- Trigger Airflow DAGs ----
st.subheader("🚀 Trigger Airflow DAGs")
dag_id = st.text_input("Enter DAG ID to trigger:")
if st.button("Trigger DAG"):
    if dag_id.strip():
        trigger_dag_cli(dag_id.strip())
    else:
        st.warning("Please enter a DAG ID")
