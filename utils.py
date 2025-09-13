import os
import pickle
import pandas as pd
import logging
import redis
from sqlalchemy import create_engine
import time
import pyarrow as pa
import pyarrow.parquet as pq

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# Directory constants
# -------------------------------
BASE_DIR = "/home/anish/airflow/dags"
MODEL_DIR = os.path.join(BASE_DIR, "monitoring/models")
DATA_DIR = os.path.join(BASE_DIR, "monitoring/data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------
# Redis client holder
# -------------------------------
_redis_conn = None


def make_redis_client(host="localhost", port=6379, db=0, timeout=3):
    """Initialize global Redis client."""
    global _redis_conn
    try:
        _redis_conn = redis.Redis(
            host=host, port=port, db=db, socket_connect_timeout=timeout
        )
        _redis_conn.ping()
        logger.info("✅ Redis client created successfully")
        return _redis_conn
    except Exception as e:
        logger.error(f"❌ Failed to create Redis client: {e}")
        _redis_conn = None
        return None


def redis_conn():
    """Return the current Redis client (or None if not initialized)."""
    global _redis_conn
    return _redis_conn

# -------------------------------
# Pickle helpers
# -------------------------------
def load_pickle(file_path, name="data"):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"✅ Loaded {name} from {file_path}")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load {name} from {file_path}: {e}")
        return None


def save_pickle(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"✅ Saved data to {file_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save data to {file_path}: {e}")
        raise

# -------------------------------
# DataFrame / Series helpers
# -------------------------------
def load_df(key, name="dataframe"):
    """Load DataFrame (or Series) from Redis or local pickle."""
    r = redis_conn()
    if r:
        try:
            retrieved = r.get(key)
            if retrieved is not None:
                df = pq.read_table(pa.BufferReader(retrieved)).to_pandas()
                logger.info(f"✅ Loaded {name} from Redis")
                # If single column -> return Series for convenience
                if df.shape[1] == 1:
                    return df.iloc[:, 0]
                return df
        except Exception as e:
            logger.warning(f"⚠️ Redis get failed for {key}: {e}")

    file_path = os.path.join(DATA_DIR, f"{key}.pkl")
    if os.path.exists(file_path):
        data = load_pickle(file_path, name)
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                return data.iloc[:, 0]
            return data
        elif isinstance(data, pd.Series):
            return data
        else:
            try:
                return pd.DataFrame(data)
            except Exception as e:
                logger.error(f"❌ Failed to convert {name} to DataFrame: {e}")
                return None

    raise FileNotFoundError(
        f"{key} not found in Redis or local pickle ({file_path})"
    )


def store_df(key, df):
    """Store DataFrame or Series in Redis + local pickle."""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame or Series")

    # Store locally
    local_path = os.path.join(DATA_DIR, f"{key}.pkl")
    save_pickle(df, local_path)

    # Store in Redis
    r = redis_conn()
    if r:
        try:
            table = pa.Table.from_pandas(df)
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf)
            r.set(key, buf.getvalue().to_pybytes())
            logger.info(f"✅ Stored {key} in Redis")
        except Exception as e:
            logger.warning(f"⚠️ Failed to store {key} in Redis: {e}")

# -------------------------------
# Database helper
# -------------------------------
def get_engine_with_retry(retries=5, delay=5):
    """Create SQLAlchemy engine with retry mechanism."""
    connection_string = (
        "mysql+pymysql://root:Pa55W0rd123#@localhost:3308/Framingham"
    )
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
