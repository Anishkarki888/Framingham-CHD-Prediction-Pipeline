# monitor_model.py
import subprocess
import threading
import webbrowser
import os
import time
import streamlit as st

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMLIT_DIR = os.path.join(BASE_DIR, "streamlit_app")

# Main monitoring apps
APP_PATH = os.path.join(STREAMLIT_DIR, "app.py")  # User input dashboard
DATA_DRIFT_PATH = os.path.join(STREAMLIT_DIR, "data_drift.py")
CONCEPT_DRIFT_PATH = os.path.join(STREAMLIT_DIR, "concept_drift.py")

# MLflow DB
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")

# ------------------------------
# Ports
# ------------------------------
APP_PORT = 8503
DATA_DRIFT_PORT = 8504
CONCEPT_DRIFT_PORT = 8505
METRICS_PORT = 8001
PROMETHEUS_UI_PORT = 9090
GRAFANA_PORT = 3000
MLFLOW_PORT = 5000

# ------------------------------
# Utility functions
# ------------------------------
def run_streamlit_app(path: str, port: int, extra_env=None):
    """
    Run a Streamlit app in a separate thread, non-blocking.
    Opens app in default web browser automatically.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Streamlit app not found: {path}")

    def target():
        env = {**os.environ}
        if extra_env:
            env.update(extra_env)

        subprocess.Popen(
            [os.sys.executable, "-m", "streamlit", "run", path, "--server.port", str(port), "--server.headless=true"],
            cwd=os.path.dirname(path),
            env=env
        )
        time.sleep(2)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=target, daemon=True).start()


def run_mlflow():
    """Launch MLflow UI in a separate thread."""
    def target():
        subprocess.Popen([
            os.sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", f"sqlite:///{MLFLOW_DB}",
            "--port", str(MLFLOW_PORT)
        ])
        time.sleep(2)
        webbrowser.open(f"http://localhost:{MLFLOW_PORT}")

    threading.Thread(target=target, daemon=True).start()


def run_prometheus_metrics_server():
    """Run Prometheus Python metrics server in a separate thread."""
    try:
        from prometheus_client import start_http_server

        def target():
            start_http_server(METRICS_PORT)
            time.sleep(2)
            webbrowser.open(f"http://localhost:{METRICS_PORT}")

        threading.Thread(target=target, daemon=True).start()
    except Exception as e:
        st.error(f"Failed to start metrics server: {e}")


def start_docker_container(container_name: str, url: str):
    """Start Docker container if not running and open its web UI."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        ).stdout.strip()

        if container_name not in result:
            subprocess.Popen(["docker", "start", container_name])
            time.sleep(5)

        webbrowser.open(url)
    except Exception as e:
        st.error(f"Failed to start {container_name}: {e}")


# ------------------------------
# Streamlit UI for manual launch (optional)
# ------------------------------
st.set_page_config(page_title="Framingham MLOps Dashboard", layout="wide")
st.title("🏥 Framingham MLOps Dashboard")
st.markdown("### Quick Launch")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧑 User Input App"):
        run_streamlit_app(APP_PATH, APP_PORT, extra_env={"RUN_PROMETHEUS": "false"})
    if st.button("📊 Data Drift Dashboard"):
        run_streamlit_app(DATA_DRIFT_PATH, DATA_DRIFT_PORT)
    if st.button("📈 Concept Drift Dashboard"):
        run_streamlit_app(CONCEPT_DRIFT_PATH, CONCEPT_DRIFT_PORT)

with col2:
    if st.button("⚡ Python Prometheus Metrics"):
        run_prometheus_metrics_server()
    if st.button("🌐 Prometheus Web UI (Docker)"):
        start_docker_container("prometheus_container", f"http://localhost:{PROMETHEUS_UI_PORT}/targets")
    if st.button("📊 Grafana Dashboard (Docker)"):
        start_docker_container("grafana_container", f"http://localhost:{GRAFANA_PORT}")

with col3:
    if st.button("🖥️ MLflow UI"):
        run_mlflow()

st.info(f"""
Ports:
- User App: {APP_PORT}
- Data Drift: {DATA_DRIFT_PORT}
- Concept Drift: {CONCEPT_DRIFT_PORT}
- Python Prometheus Metrics: {METRICS_PORT}
- Prometheus Web UI: {PROMETHEUS_UI_PORT}/targets
- Grafana: {GRAFANA_PORT}
- MLflow: {MLFLOW_PORT}
""")
