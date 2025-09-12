import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Add dags folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../monitoring/streamlit_app
DAGS_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))      # .../dags
sys.path.insert(0, DAGS_DIR)

from utils import load_pickle

st.title("📉 Concept Drift Monitor")

BASE_PATH = os.path.join(os.path.dirname(__file__), "../data")
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "../models/final_pipeline.pkl")
USER_PREDICTIONS_PATH = os.path.join(BASE_PATH, "user_predictions.pkl")

# ------------------------------
# Load reference dataset
# ------------------------------
reference = load_pickle(os.path.join(BASE_PATH, "X_train.pkl"), "X_train")

# ------------------------------
# Load user predictions / current dataset
# ------------------------------
if os.path.exists(USER_PREDICTIONS_PATH):
    user_preds = load_pickle(USER_PREDICTIONS_PATH, "User Predictions") or []
    if user_preds:
        current_data = pd.DataFrame([p["input"] for p in user_preds])
        current_labels = np.array([p["class"] for p in user_preds])
    else:
        st.warning("No user predictions available yet.")
        current_data = pd.DataFrame()
        current_labels = np.array([])
else:
    st.warning("User predictions file not found.")
    current_data = pd.DataFrame()
    current_labels = np.array([])

# ------------------------------
# Check if we have data to compare
# ------------------------------
if reference.empty or current_data.empty:
    st.info("Not enough data to compute concept drift.")
else:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset  # Using DataDriftPreset for simplicity

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current_data)

    REPORTS_PATH = os.path.join(os.path.dirname(__file__), "../reports")
    os.makedirs(REPORTS_PATH, exist_ok=True)
    report_file = os.path.join(REPORTS_PATH, "concept_drift_report.html")
    report.save_html(report_file)

    st.success(f"✅ Concept drift report saved: {report_file}")
    st.components.v1.html(open(report_file, "r").read(), height=800, scrolling=True)
