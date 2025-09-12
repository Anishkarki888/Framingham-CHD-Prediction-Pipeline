import os
import pickle
import pandas as pd
import logging
import redis
from sqlalchemy import create_engine
import time
import pyarrow as pa
import pyarrow.parquet as pq

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory constants
MODEL_DIR = "/home/anish/airflow/dags/monitoring/models"
DATA_DIR = "/home/anish/airflow/dags/monitoring/data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Redis connection (optional)
redis_conn = None

def load_pickle(file_path, name="data"):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"✅ Loaded {name} from {file_path}")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load {name} from {file_path}: {e}")
        return None

def save_pickle(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✅ Saved data to {file_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save data to {file_path}: {e}")
        raise

def load_df(key, name="dataframe"):
    if redis_conn:
        try:
            retrieved = redis_conn.get(key)
            if retrieved is not None:
                df = pq.read_table(pa.BufferReader(retrieved)).to_pandas()
                logger.info(f"✅ Loaded {name} from Redis")
                return df
        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {e}")
    
    file_path = os.path.join(DATA_DIR, f"{key}.pkl")
    if os.path.exists(file_path):
        data = load_pickle(file_path, name)
        if data is not None and not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                logger.error(f"❌ Failed to convert {name} to DataFrame: {e}")
                return None
        return data
    raise FileNotFoundError(f"{key} not found in Redis or local pickle ({file_path})")

def store_df(key, df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    local_path = os.path.join(DATA_DIR, f"{key}.pkl")
    save_pickle(df, local_path)

    if redis_conn:
        try:
            table = pa.Table.from_pandas(df)
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf)
            redis_conn.set(key, buf.getvalue().to_pybytes())
            logger.info(f"✅ Stored {key} in Redis")
        except Exception as e:
            logger.warning(f"⚠️ Failed to store {key} in Redis: {e}")

def get_engine_with_retry(retries=5, delay=5):
    connection_string = "mysql+pymysql://root:Pa55W0rd123#@localhost:3308/Framingham"
    for attempt in range(1, retries + 1):
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("✅ Database engine created successfully")
            return engine
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(delay)
    return None

def make_redis_client():
    global redis_conn
    try:
        redis_conn = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=3)
        redis_conn.ping()
        logger.info("✅ Redis client created successfully")
        return redis_conn
    except Exception as e:
        logger.error(f"❌ Failed to create Redis client: {e}")
        redis_conn = None
        return None
