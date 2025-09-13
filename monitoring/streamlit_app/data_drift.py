import sys
import os
import streamlit as st
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab


# Add dags folder to path for utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))       
DAGS_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))        
sys.path.insert(0, DAGS_DIR)

from utils import load_pickle

# Streamlit UI
st.title("Framingham Data Drift Monitor")

BASE_PATH = os.path.join(os.path.dirname(__file__), "../data")

# Load reference and current datasets
reference = load_pickle(os.path.join(BASE_PATH, "X_train.pkl"), "X_train")
current = load_pickle(os.path.join(BASE_PATH, "X_test.pkl"), "X_test")

# Create Evidently dashboard
dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(reference, current)

# Save report
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "../reports")
os.makedirs(REPORTS_PATH, exist_ok=True)
report_file = os.path.join(REPORTS_PATH, "data_drift_report.html")
dashboard.save(report_file)

st.write(f"Data Drift report saved to {report_file}")
st.components.v1.html(open(report_file, "r").read(), height=800, scrolling=True)
