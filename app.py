import streamlit as st
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
import tempfile
import json
import logging
import os
from mlflow.tracking import MlflowClient

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Framingham"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Function to fetch model metrics from MLflow
def get_model_metrics():
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            logger.error(f"Experiment '{EXPERIMENT_NAME}' not found")
            return None, None
        
        # Search for runs in "Framingham" experiment, trying multiple run names
        run_names = ["final_catboost_model", "catboost_optuna", "catboost_default"]
        for run_name in run_names:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"attributes.run_name = '{run_name}'",
                order_by=["metrics.roc_auc DESC"],
                max_results=1
            )
            if runs:
                run = runs[0]
                metrics = run.data.metrics
                accuracy = metrics.get("accuracy", None)
                precision = metrics.get("precision", None)
                if accuracy is not None and precision is not None:
                    logger.info(f"Fetched metrics from run '{run_name}': Accuracy={accuracy:.4f}, Precision={precision:.4f}")
                    return accuracy, precision
        logger.error("No runs with accuracy and precision found")
        return None, None
    except Exception as e:
        logger.error(f"Failed to fetch metrics from MLflow: {str(e)}")
        return None, None

# Load trained pipeline
with open("/home/anish/airflow/dags/models/final_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

model = pipeline['model']
scaler = pipeline['scaler']

# Fetch model metrics
accuracy, precision = get_model_metrics()

# Page Setup
st.set_page_config(
    page_title="Framingham Heart Risk Calculator",
    page_icon="❤️",
    layout="centered"
)

st.title("🫀 Framingham Heart Risk Calculator")
st.write(
    """
    This tool uses the **Framingham dataset** and a trained **CatBoost classification model**  
    to estimate your **10-year risk of developing heart disease**.  
    👉 Enter your details below to get an instant prediction.  
    """
)

# Input Form
with st.form("health_form"):
    st.subheader("👤 Demographics")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, help="Enter age in years (18-100)")
    male = st.radio("Sex", ["Female", "Male"])
    
    education_map = {
        1: "Some High School",
        2: "High School/GED",
        3: "Some College",
        4: "College"
    }
    education = st.selectbox(
        "Education Level",
        options=list(education_map.keys()),
        format_func=lambda x: education_map[x],
        help="Select your highest education level"
    )

    st.subheader("🚬 Lifestyle")
    smoker = st.radio("Do you currently smoke?", ["No", "Yes"])
    cigs_per_day = st.slider("Cigarettes per day (if smoker)", min_value=0, max_value=70, value=0, help="Number of cigarettes you smoke daily (0-70)")

    st.subheader("💉 Medical History")
    bp_meds = st.radio("On BP medication?", ["No", "Yes"])
    stroke = st.radio("History of stroke?", ["No", "Yes"])
    hypertension = st.radio("Prevalent hypertension?", ["No", "Yes"])
    diabetes = st.radio("Diabetes?", ["No", "Yes"])

    st.subheader("🩺 Health Measurements")
    tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=200.0, help="Typical range: 100-400")
    sys_bp = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=250.0, value=120.0, help="Typical range: 80-250")
    dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=50.0, max_value=150.0, value=80.0, help="Typical range: 50-150")
    bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, help="Body Mass Index (kg/m²)")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70, help="Typical range: 40-150 bpm")
    glucose = st.number_input("Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=90.0, help="Typical range: 40-400")

    submitted = st.form_submit_button("🔍 Calculate Risk")

# Prediction
if submitted:
    # Model info
    st.subheader("ℹ️ Model Information")
    if accuracy is not None and precision is not None:
        st.write(
            f"""
            **Algorithm:** CatBoost Classifier  
            **Accuracy:** {accuracy:.4f} (from model training)  
            **Precision:** {precision:.4f} (from model training)  

            **Interpretation:**  
            - `0` → 0-50% risk → Low Risk  
            - `1` → 50-100% risk → High Risk
            """
        )
    else:
        st.warning("Unable to fetch model metrics from MLflow. Using default values.")
        st.write(
            """
            **Algorithm:** CatBoost Classifier  
            **Accuracy:** Not available  
            **Precision:** Not available  

            **Interpretation:**  
            - `0` → 0-50% risk → Low Risk  
            - `1` → 50-100% risk → High Risk
            """
        )

    # Validate smoking logic
    if smoker == "No" and cigs_per_day > 0:
        st.error("Non-smokers must have 0 cigarettes per day.")
    else:
        # Prepare input vector
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
            "totChol": min(tot_chol, 300.0),
            "sysBP": min(sys_bp, 180.0),
            "diaBP": dia_bp,
            "BMI": bmi,
            "heartRate": heart_rate,
            "glucose": min(glucose, 200.0)
        }

        df_input = pd.DataFrame([input_dict])
        df_scaled = df_input.copy()
        numeric_cols = ["age", "education", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
        df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])

        # MLflow logging
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="user_prediction") as run:
            for key, value in input_dict.items():
                mlflow.log_param(key, value)
            prob = model.predict_proba(df_scaled)[:,1][0]
            pred_class = model.predict(df_scaled)[0]
            mlflow.log_metric("predicted_risk_probability", prob)
            mlflow.log_metric("predicted_risk_class", pred_class)
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
                json.dump(input_dict, tmp, indent=2)
                tmp_path = tmp.name
            mlflow.log_artifact(tmp_path, "input_data")
            os.remove(tmp_path)
            run_id = run.info.run_id
            logger.info(f"Logged MLflow run: {run_id}")

        # Map probability to risk label
        prob_percent = round(prob * 100, 2)
        risk_label = "Low Risk ✅" if pred_class == 0 else "High Risk ⚠️"
        message = (
            f"Your model prediction is **{pred_class} ({risk_label})**. "
            f"Based on your health profile, your estimated 10-year risk is **{prob_percent}%**. "
            f"{'Keep maintaining your healthy lifestyle!' if pred_class == 0 else 'Please consult a healthcare professional.'}"
        )

        # Results
        st.subheader("📊 Prediction Result")
        st.metric(label="Estimated 10-Year CHD Risk", value=f"{prob_percent}%")
        st.success(risk_label)
        st.info(message)
        st.write(f"MLflow Run ID: {run_id}")