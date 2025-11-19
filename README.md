# Framingham-CHD-Prediction-Pipeline
to run ui: in monitoring environment and run: streamlit run home_app.py
it will open ui of user input,datadrift,concept drift and mlflow

# database
mysql -u root -p -h 127.0.0.1 -P 3308 Framingham

# mlflow
mlflow ui --backend-store-uri sqlite:////home/anish/airflow/dags/monitoring/mlflow/mlflow.db --port 5000




# if grafana is not working then
docker update --restart unless-stopped grafana_test



'''
airflow/dags/framingham_dags/
├─ main_pipeline_dag.py         # Orchestrator DAG
├─ 00_start_services_dag.py     # Start services
├─ 01_data_ingest_dag.py        # Data ingestion
├─ 02_raw_validation_dag.py     # Raw data validation
├─ 03_star_schema_dag.py        # Star schema creation
├─ 04_preprocessing_dag.py      # Preprocessing
├─ 05_processed_validation_dag.py  # Post-process validation
├─ 06_model_training_dag.py     # Tuning + training
├─ 07_model_deploy_dag.py       # Deployment



# Email configuration (replace with your SMTP details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "anish_24152356@sunway.edu.np"  
SMTP_PASSWORD = "**********"
RECIPIENT_EMAIL = "anishkarki989@gmail.com"  

'''
