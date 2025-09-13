import subprocess
import threading
import webbrowser
import streamlit as st
import os
import time
import socket
import psutil

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMLIT_DIR = os.path.join(BASE_DIR, "streamlit_app")

APP_PATH = os.path.join(STREAMLIT_DIR, "app.py")
DATA_DRIFT_PATH = os.path.join(STREAMLIT_DIR, "data_drift.py")
CONCEPT_DRIFT_PATH = os.path.join(STREAMLIT_DIR, "concept_drift.py")
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow", "mlflow.db")

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
def wait_for_port(port, host="localhost", timeout=30):
    """Wait until a port is open."""
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
    """Check if a port is already in use."""
    return any(conn.laddr.port == port for conn in psutil.net_connections())

def run_streamlit_app(path, port):
    """Run any Streamlit app in a separate thread."""
    def target():
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
    """Run MLflow UI safely."""
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

def run_prometheus_metrics_server():
    """Run Python Prometheus metrics server."""
    try:
        from prometheus_client import start_http_server
        def target():
            if not is_port_in_use(METRICS_PORT):
                start_http_server(METRICS_PORT)
                try:
                    wait_for_port(METRICS_PORT, timeout=30)
                except TimeoutError as e:
                    st.error(str(e))
            webbrowser.open(f"http://localhost:{METRICS_PORT}")
        threading.Thread(target=target, daemon=True).start()
    except Exception as e:
        st.error(f"Failed to start metrics server: {e}")

def start_docker_container(container_name, url):
    """Start Docker container if not running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True, text=True
        ).stdout.strip()
        if container_name not in result:
            subprocess.Popen(["docker", "start", container_name])
            time.sleep(5)
        webbrowser.open(url)
    except Exception as e:
        st.error(f"Failed to start {container_name}: {e}")

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Framingham MLOps Dashboard", layout="wide")
st.title("🏥 Framingham MLOps Dashboard")
st.markdown("### Quick Launch")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧑 User Input App"):
        run_streamlit_app(APP_PATH, APP_PORT)
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
