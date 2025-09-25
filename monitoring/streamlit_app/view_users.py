import streamlit as st
import pandas as pd
from utils import get_engine_with_retry, decrypt_value
from sqlalchemy import text
import logging

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="View Saved User Inputs",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Saved User Inputs (Admin View)")

# ---------------- Connect to Database ----------------
engine = get_engine_with_retry()
if engine is None:
    st.error("❌ Could not connect to the database.")
    st.stop()

# ---------------- Fetch Data ----------------
try:
    with engine.connect() as conn:
        query = text("SELECT * FROM user_inputs")
        df = pd.read_sql(query, conn)
    st.success(f"✅ Fetched {len(df)} records from the database.")
except Exception as e:
    st.error(f"❌ Failed to fetch data: {str(e)}")
    logger.error(f"DB fetch error: {str(e)}")
    st.stop()

# ---------------- Decrypt user_id ----------------
if "user_id" in df.columns:
    try:
        df["user_id_decrypted"] = df["user_id"].apply(decrypt_value)
    except Exception as e:
        logger.warning(f"Failed to decrypt user_id: {e}")
        st.warning("⚠️ Some user_ids could not be decrypted.")

# ---------------- Display Data ----------------
st.subheader("User Data Table")
st.dataframe(df)

# ---------------- Optional: Download CSV ----------------
csv = df.to_csv(index=False).encode()
st.download_button(
    label="📥 Download as CSV",
    data=csv,
    file_name="user_inputs.csv",
    mime="text/csv"
)
