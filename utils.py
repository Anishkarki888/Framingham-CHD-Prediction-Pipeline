import os
import pickle
import pandas as pd
import logging
import redis
from sqlalchemy import create_engine, text
import time
import pyarrow as pa
import pyarrow.parquet as pq
from cryptography.fernet import Fernet

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------- Directories -------------------
BASE_DIR = "/home/anish/airflow/dags"
MODEL_DIR = os.path.join(BASE_DIR, "monitoring/models")
DATA_DIR = os.path.join(BASE_DIR, "monitoring/data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------- Redis -------------------
_redis_conn = None

def make_redis_client(host="localhost", port=6379, db=0, timeout=3):
    global _redis_conn
    try:
        _redis_conn = redis.Redis(
            host=host, port=port, db=db, socket_connect_timeout=timeout
        )
        _redis_conn.ping()
        logger.info("Redis client created successfully")
        return _redis_conn
    except Exception as e:
        logger.error(f"Failed to create Redis client: {e}")
        _redis_conn = None
        return None

def redis_conn():
    global _redis_conn
    return _redis_conn

# ------------------- Pickle Utilities -------------------
def load_pickle(file_path, name="data"):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded {name} from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {name} from {file_path}: {e}")
        return None

def save_pickle(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise

# ------------------- Redis + Pickle DataFrame -------------------
def load_df(key, name="dataframe"):
    r = redis_conn()
    if r:
        try:
            retrieved = r.get(key)
            if retrieved is not None:
                df = pq.read_table(pa.BufferReader(retrieved)).to_pandas()
                logger.info(f"Loaded {name} from Redis")
                return df if df.shape[1] > 1 else df.iloc[:, 0]
        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {e}")

    file_path = os.path.join(DATA_DIR, f"{key}.pkl")
    if os.path.exists(file_path):
        data = load_pickle(file_path, name)
        if isinstance(data, pd.DataFrame):
            return data if data.shape[1] > 1 else data.iloc[:, 0]
        elif isinstance(data, pd.Series):
            return data
        else:
            try:
                return pd.DataFrame(data)
            except Exception as e:
                logger.error(f"Failed to convert {name} to DataFrame: {e}")
                return None

    raise FileNotFoundError(f"{key} not found in Redis or local pickle ({file_path})")

def store_df(key, df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame or Series")

    local_path = os.path.join(DATA_DIR, f"{key}.pkl")
    save_pickle(df, local_path)

    r = redis_conn()
    if r:
        try:
            table = pa.Table.from_pandas(df)
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf)
            r.set(key, buf.getvalue().to_pybytes())
            logger.info(f"Stored {key} in Redis")
        except Exception as e:
            logger.warning(f"Failed to store {key} in Redis: {e}")

# ------------------- Database Engine -------------------
def get_engine_with_retry(retries=5, delay=5):
    connection_string = "mysql+pymysql://root:Pa55W0rd123#@127.0.0.1:3308/Framingham"
    for attempt in range(1, retries + 1):
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database engine created successfully")
            return engine
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(delay)
    return None

# ------------------- Encryption -------------------
# Generate a key once and save it securely. Example key:
ENCRYPTION_KEY = Fernet.generate_key()
cipher = Fernet(ENCRYPTION_KEY)

def encrypt_value(value: str) -> str:
    return cipher.encrypt(value.encode()).decode()

def decrypt_value(value: str) -> str:
    return cipher.decrypt(value.encode()).decode()

# ------------------- Save User Input to MariaDB -------------------
def save_user_input_to_db(user_data: dict):
    """
    Save a dictionary of user input and prediction to MariaDB.
    Uses REPLACE INTO to handle duplicate user_ids.
    """
    engine = get_engine_with_retry()
    if engine is None:
        logger.error("Database engine not available")
        return False

    try:
        # Create a copy of user_data
        user_data_to_save = user_data.copy()
        
        # Keep user_id as plain text (don't encrypt to avoid complications)
        user_data_to_save['user_id'] = str(user_data['user_id'])
        
        # Use REPLACE INTO instead of INSERT to handle duplicate primary keys
        columns = ", ".join(user_data_to_save.keys())
        placeholders = ", ".join([f":{k}" for k in user_data_to_save.keys()])
        sql = text(f"REPLACE INTO user_inputs ({columns}) VALUES ({placeholders})")

        with engine.begin() as conn:
            result = conn.execute(sql, user_data_to_save)
            logger.info(f"User input saved to DB for user_id={user_data['user_id']}")
            logger.info(f"Affected rows: {result.rowcount}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to insert user input to DB: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"SQL Error details: {str(e)}")
        
        # Print more debugging info
        try:
            logger.error(f"User data keys: {list(user_data.keys())}")
            logger.error(f"Data to save keys: {list(user_data_to_save.keys()) if 'user_data_to_save' in locals() else 'N/A'}")
        except:
            pass
            
        return False