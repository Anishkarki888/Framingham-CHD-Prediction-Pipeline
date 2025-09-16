import sys
import os
import streamlit as st
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
import tempfile
import json
import logging
from pydantic import BaseModel, Field
from prometheus_client import Gauge, start_http_server

# ---------------- Add parent DAG folder to path ----------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils import load_pickle, save_pickle

# ---------------- Logging setup ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models/final_pipeline.pkl")
LATEST_METRICS_PATH = os.path.join(BASE_DIR, "models/latest_metrics.json")

# ---------------- Prometheus setup ----------------
PROM_PORT = 8002
try:
    start_http_server(PROM_PORT)
    logger.info(f"✅ Prometheus metrics server started on :{PROM_PORT}")
except Exception as e:
    logger.warning(f"Could not start Prometheus metrics server on :{PROM_PORT} -> {e}")

# ---------------- Pydantic model for input validation ----------------
class PatientInput(BaseModel):
    male: int = Field(..., ge=0, le=1)
    age: int = Field(..., ge=18, le=100)
    education: int = Field(..., ge=1, le=4)
    currentSmoker: int = Field(..., ge=0, le=1)
    cigsPerDay: int = Field(..., ge=0, le=70)
    BPMeds: int = Field(..., ge=0, le=1)
    prevalentStroke: int = Field(..., ge=0, le=1)
    prevalentHyp: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    totChol: float = Field(..., ge=100.0, le=400.0)
    sysBP: float = Field(..., ge=80.0, le=250.0)
    diaBP: float = Field(..., ge=50.0, le=150.0)
    BMI: float = Field(..., ge=15.0, le=50.0)
    heartRate: int = Field(..., ge=40, le=150)
    glucose: float = Field(..., ge=40.0, le=400.0)

# ---------------- Fetch latest model metrics ----------------
def get_latest_metrics():
    try:
        if os.path.exists(LATEST_METRICS_PATH):
            with open(LATEST_METRICS_PATH, "r") as f:
                metrics = json.load(f)
            accuracy = metrics.get("accuracy", None)
            precision = metrics.get("precision", None)
            return accuracy, precision
        else:
            logger.warning(f"{LATEST_METRICS_PATH} not found. Metrics unavailable.")
            return None, None
    except Exception as e:
        logger.error(f"Failed to load latest metrics: {str(e)}")
        return None, None

accuracy, precision = get_latest_metrics()

# ---------------- Load trained pipeline ----------------
try:
    pipeline = load_pickle(MODEL_PATH, "ML Pipeline")
    model = pipeline['model']
    scaler = pipeline['scaler']
    logger.info(f"✅ Loaded pipeline from {MODEL_PATH}")
except Exception as e:
    st.error(f"Failed to load model pipeline: {str(e)}")
    logger.error(f"Failed to load pipeline: {str(e)}")
    st.stop()

# ---------------- Streamlit page setup ----------------
st.set_page_config(
    page_title="Framingham Heart Risk Calculator",
    page_icon="❤️",
    layout="centered"
)

st.title("🫀 Framingham Heart Risk Calculator")
st.write("""
This tool uses the **Framingham dataset** and a trained **CatBoost classification model**  
to estimate your **10-year risk of developing heart disease**.  
👉 Enter your details below to get an instant prediction.
""")

# ---------------- Username input ----------------
username = st.text_input("Enter your name (unique for each session):", max_chars=50)
if not username:
    st.warning("Please enter your name to proceed with prediction.")
    st.stop()

# ---------------- Input form ----------------
with st.form("health_form"):
    st.subheader("👤 Demographics")
    age = st.number_input("Age (years)", 18, 100, 45)
    male = st.radio("Sex", ["Female", "Male"])
    
    education_map = {1: "Some High School", 2: "High School/GED", 3: "Some College", 4: "College"}
    education = st.selectbox(
        "Education Level",
        options=list(education_map.keys()),
        format_func=lambda x: education_map[x]
    )

    st.subheader("🚬 Lifestyle")
    smoker = st.radio("Do you currently smoke?", ["No", "Yes"])
    cigs_per_day = st.slider("Cigarettes per day (if smoker)", 0, 70, 0)

    st.subheader("💉 Medical History")
    bp_meds = st.radio("On BP medication?", ["No", "Yes"])
    stroke = st.radio("History of stroke?", ["No", "Yes"])
    hypertension = st.radio("Prevalent hypertension?", ["No", "Yes"])
    diabetes = st.radio("Diabetes?", ["No", "Yes"])

    st.subheader("🩺 Health Measurements")
    tot_chol = st.number_input("Total Cholesterol (mg/dL)", 100.0, 400.0, 200.0)
    sys_bp = st.number_input("Systolic BP (mmHg)", 80.0, 250.0, 120.0)
    dia_bp = st.number_input("Diastolic BP (mmHg)", 50.0, 150.0, 80.0)
    bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 70)
    glucose = st.number_input("Glucose (mg/dL)", 40.0, 400.0, 90.0)

    submitted = st.form_submit_button(f"🔍 Calculate Risk for {username}")

# ---------------- Prediction ----------------
if submitted:
    st.subheader("ℹ️ Model Information")
    if accuracy is not None and precision is not None:
        st.write(f"**Algorithm:** CatBoost Classifier  \n**Accuracy:** {accuracy:.4f}  \n**Precision:** {precision:.4f}")
    else:
        st.warning("Unable to fetch model metrics from latest_metrics.json.")
        st.write("**Algorithm:** CatBoost Classifier  \n**Accuracy:** N/A  \n**Precision:** N/A")

    try:
        input_dict = {
            "male": 1 if male == "Male" else 0,
            "age": age,
            "education": education,
            "currentSmoker": 1 if smoker == "Yes" else 0,
            "cigsPerDay": cigs_per_day,
            "BPMeds": 1 if bp_meds == "Yes" else 0,
            "prevalentStroke": 1 if stroke == "Yes" else 0,
            "prevalentHyp": 1 if hypertension == "Yes" else 0,
            "diabetes": 1 if diabetes == "Yes" else 0,
            "totChol": tot_chol,
            "sysBP": sys_bp,
            "diaBP": dia_bp,
            "BMI": bmi,
            "heartRate": heart_rate,
            "glucose": glucose
        }

        patient = PatientInput(**input_dict)
        if patient.currentSmoker == 0 and patient.cigsPerDay > 0:
            st.error("Non-smokers must have 0 cigarettes per day.")
        else:
            df_input = pd.DataFrame([input_dict])
            scaler_cols = ["cigsPerDay", "BPMeds", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
            df_input[scaler_cols] = scaler.transform(df_input[scaler_cols])

            # ---------------- MLflow logging ----------------
            mlflow.set_experiment("Framingham")
            with mlflow.start_run(run_name=f"user_prediction_{username}") as run:
                for k, v in input_dict.items():
                    mlflow.log_param(k, v)

                prob = model.predict_proba(df_input)[:,1][0]
                pred_class = model.predict(df_input)[0]

                mlflow.log_metric(f"predicted_risk_probability_{username}", prob)
                mlflow.log_metric(f"predicted_risk_class_{username}", pred_class)

                Gauge(f"predicted_risk_probability_{username}", "Prediction probability", ["app"]).labels(app="framingham_app").set(prob)
                Gauge(f"predicted_risk_class_{username}", "Prediction class", ["app"]).labels(app="framingham_app").set(pred_class)

                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
                    json.dump(input_dict, tmp, indent=2)
                    tmp_path = tmp.name
                mlflow.log_artifact(tmp_path, "input_data")
                os.remove(tmp_path)

                run_id = run.info.run_id

            prob_percent = round(prob * 100, 2)
            risk_label = "Low Risk ✅" if pred_class == 0 else "High Risk ⚠️"
            message = (
                f"{username}, your model prediction is **{pred_class} ({risk_label})**. "
                f"Estimated 10-year risk: **{prob_percent}%**. "
                f"{'Keep maintaining your healthy lifestyle!' if pred_class == 0 else 'Please consult a healthcare professional.'}"
            )

            st.subheader("📊 Prediction Result")
            st.metric(label=f"{username}'s Estimated 10-Year CHD Risk", value=f"{prob_percent}%")
            st.success(risk_label)
            st.info(message)
            st.write(f"MLflow Run ID: {run_id}")

    except Exception as e:
        st.error(f"Invalid input: {str(e)}")
        logger.error(f"Validation failed: {str(e)}")
