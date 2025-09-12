# prometheus_server.py
from prometheus_client import start_http_server
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    start_http_server(8001)
    logger.info("✅ Prometheus server running on :8001")
    while True:
        time.sleep(60)  # Keeps server alive
