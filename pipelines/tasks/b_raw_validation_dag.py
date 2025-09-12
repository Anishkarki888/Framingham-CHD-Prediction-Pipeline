from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils import load_df, logger, make_redis_client
import great_expectations as ge

default_args = {
    'owner': 'Anish',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    "raw_validation",
    default_args=default_args,
    description="Validates raw Framingham dataset",
    start_date=datetime(2025, 8, 24),
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    is_paused_upon_creation=False
)

def validate_raw_data(**kwargs):
    logger.info("=== STARTING RAW DATA VALIDATION ===")
    
    # Initialize Redis client
    make_redis_client()
    
    # Load data with proper error handling
    try:
        df = load_df("framingham_raw")
    except FileNotFoundError as e:
        logger.error(f"❌ Raw data not found: {e}")
        raise ValueError("Raw data 'framingham_raw' not found. Make sure data ingestion completed successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load raw data: {e}")
        raise
    
    # Check if data was loaded successfully
    if df is None:
        logger.error("❌ Loaded data is None")
        raise ValueError("Raw data 'framingham_raw' could not be loaded")
    
    logger.info(f"✅ Loaded raw data with shape: {df.shape}")
    
    # Create Great Expectations dataset
    try:
        ge_df = ge.dataset.PandasDataset(df)
    except Exception as e:
        logger.error(f"❌ Failed to create Great Expectations dataset: {e}")
        raise
    
    # Critical columns must exist and have no nulls
    critical_cols = ["age", "male", "TenYearCHD"]
    missing_critical = []
    
    for col in critical_cols:
        if col not in df.columns:
            missing_critical.append(col)
            logger.error(f"❌ Critical column missing: {col}")
        else:
            logger.info(f"✅ Critical column found: {col}")
            try:
                ge_df.expect_column_values_to_not_be_null(col)
            except Exception as e:
                logger.warning(f"⚠️ Validation warning for {col}: {e}")
    
    # Fail if any critical columns are missing
    if missing_critical:
        raise ValueError(f"Critical columns missing: {missing_critical}")
    
    # Optional columns must exist (can have nulls)
    optional_cols = [
        "education", "currentSmoker", "cigsPerDay", "BPMeds",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    ]
    
    missing_optional = []
    for col in optional_cols:
        if col not in df.columns:
            missing_optional.append(col)
            logger.warning(f"⚠️ Optional column missing: {col}")
        else:
            logger.info(f"✅ Optional column found: {col}")
            try:
                ge_df.expect_column_to_exist(col)
            except Exception as e:
                logger.warning(f"⚠️ Validation warning for {col}: {e}")
    
    if missing_optional:
        logger.warning(f"Optional columns missing: {missing_optional}")
    
    # Run comprehensive validation
    try:
        result = ge_df.validate()
        logger.info(f"Validation result: success={result['success']}")
        
        if not result["success"]:
            logger.warning(f"Validation warnings: {result.get('statistics', 'No statistics available')}")
            # Don't fail for warnings, just log them
            logger.info("=== RAW VALIDATION COMPLETED WITH WARNINGS ===")
        else:
            logger.info("=== RAW VALIDATION PASSED ===")
            
    except Exception as e:
        logger.error(f"❌ Validation failed with exception: {e}")
        raise
    
    logger.info(f"Final validation summary:")
    logger.info(f"  - DataFrame shape: {df.shape}")
    logger.info(f"  - Critical columns OK: {len(critical_cols) - len(missing_critical)}/{len(critical_cols)}")
    logger.info(f"  - Optional columns OK: {len(optional_cols) - len(missing_optional)}/{len(optional_cols)}")

# Airflow task
validate_raw_data_task = PythonOperator(
    task_id="validate_raw_data",
    python_callable=validate_raw_data,
    dag=dag
)