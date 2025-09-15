import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
import pickle

# Add dags folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))     
DAGS_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))     
sys.path.insert(0, DAGS_DIR)

from utils import load_pickle

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "anish_24152356@sunway.edu.np"
SMTP_PASSWORD = "sunwaY@123"
RECIPIENT_EMAIL = "anishkarki989@gmail.com"

def send_email(subject: str, body: str):
    """Send an email alert."""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = RECIPIENT_EMAIL

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, RECIPIENT_EMAIL, msg.as_string())

# Streamlit UI
st.title("Concept Drift Monitor")

BASE_PATH = os.path.join(os.path.dirname(__file__), "../data")
USER_PREDICTIONS_PATH = os.path.join(BASE_PATH, "user_predictions.pkl")

# Load reference dataset
reference = load_pickle(os.path.join(BASE_PATH, "X_train.pkl"), "X_train")

# -----------------------
# Persistent stack setup
# -----------------------
# Load existing predictions
if os.path.exists(USER_PREDICTIONS_PATH):
    all_preds = load_pickle(USER_PREDICTIONS_PATH, "User Predictions") or []
else:
    all_preds = []

# -----------------------
# Add new user input here
# -----------------------
# For example purposes, replace with your input collection
# new_input = {
#     "input": user_input_features,  
#     "class": user_input_class,     
#     "prediction": model_prediction 
# }
# if "input" in new_input and "class" in new_input:
#     all_preds.append(new_input)

# Keep only the last N entries (stack behavior)
N = 10
all_preds = all_preds[-N:]

# Save back to pickle (persistent)
with open(USER_PREDICTIONS_PATH, "wb") as f:
    pickle.dump(all_preds, f)

# Filter out invalid entries
user_preds = [p for p in all_preds if "input" in p and "class" in p]

# Prepare current data
if user_preds:
    current_data = pd.DataFrame([p["input"] for p in user_preds])
    current_labels = np.array([p["class"] for p in user_preds])
    predictions = np.array([p.get("prediction", 0) for p in user_preds])
else:
    st.warning("No valid user predictions available.")
    current_data = pd.DataFrame()
    current_labels = np.array([])
    predictions = np.array([])

# -----------------------
# Run Concept Drift Report
# -----------------------
if reference.empty or current_data.empty:
    st.info("Not enough data to compute concept drift.")
else:
    from evidently.report import Report
    from evidently.metric_preset import ClassificationPreset, DataDriftPreset  

    # Check if we have predictions to decide on report type
    has_predictions = len(predictions) > 0 and predictions.sum() > 0
    
    if has_predictions:
        # Add target and prediction columns to BOTH datasets for ClassificationPreset
        reference_with_cols = reference.copy()
        reference_with_cols["target"] = np.zeros(len(reference))
        reference_with_cols["prediction"] = np.zeros(len(reference))  # Add dummy predictions
        
        current_with_cols = current_data.copy()
        current_with_cols["target"] = current_labels
        current_with_cols["prediction"] = predictions
        
        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=reference_with_cols, current_data=current_with_cols)
        
    else:
        # For DataDriftPreset, only add target columns
        reference_with_cols = reference.copy()
        reference_with_cols["target"] = np.zeros(len(reference))
        
        current_with_cols = current_data.copy()
        current_with_cols["target"] = current_labels
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_with_cols, current_data=current_with_cols)

    # Save report
    REPORTS_PATH = os.path.join(os.path.dirname(__file__), "../reports")
    os.makedirs(REPORTS_PATH, exist_ok=True)
    report_file = os.path.join(REPORTS_PATH, "concept_drift_report.html")
    report.save_html(report_file)

    st.info(f"Concept drift report saved: {report_file}")
    st.components.v1.html(open(report_file, "r").read(), height=800, scrolling=True)

    # Calculate AUC and send email if predictions exist
    if has_predictions:
        from sklearn.metrics import roc_auc_score
        
        try:
            auc_score = roc_auc_score(current_labels, predictions)
            st.info(f"AUC score: {auc_score:.3f}")
            if auc_score < 0.5:
                send_email(
                    subject="Alert: Low Model Performance Detected",
                    body=f"The current AUC score is {auc_score:.3f}, which is below the safe threshold of 0.5."
                )
        except ValueError as e:
            st.warning(f"Could not calculate AUC score: {e}")
    else:
        st.info("No predictions available for AUC calculation.")