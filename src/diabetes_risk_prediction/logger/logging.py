import logging
import os
import sys
from datetime import datetime


LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
# LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Define handlers
handlers = [
    logging.StreamHandler(sys.stdout),   # Console output
    logging.FileHandler(LOG_FILE, encoding='utf-8')        # File output
]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s-%(lineno)d - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers  # Specify handlers directly
)

# Optional test
if __name__ == "__main__":
    logging.info("✅ Logging setup successful.")









# import logging
# import sys
# import os
# from datetime import datetime
# from pythonjsonlogger import jsonlogger


# # ================================
# # LOG DIRECTORY SETUP
# # ================================
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)

# LOG_FILE = os.path.join(
#     LOG_DIR,
#     f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# )


# # ================================
# # SINGLE LOGGER SETUP FUNCTION
# # ================================
# def setup_logger(name: str = "diabetes_pipeline",
#                  level: str = "INFO",
#                  json_logs: bool = False) -> logging.Logger:

#     logger = logging.getLogger(name)
#     logger.setLevel(level)

#     # prevent duplicate handlers (VERY IMPORTANT)
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # ============================
#     # FORMATTER
#     # ============================
#     if json_logs:
#         formatter = jsonlogger.JsonFormatter(
#             "%(asctime)s %(name)s %(levelname)s %(message)s",
#             rename_fields={
#                 "asctime": "timestamp",
#                 "levelname": "level"
#             }
#         )
#     else:
#         formatter = logging.Formatter(
#             "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S"
#         )

#     # ============================
#     # HANDLERS
#     # ============================
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(formatter)

#     file_handler = logging.FileHandler(LOG_FILE)
#     file_handler.setFormatter(formatter)

#     # attach handlers
#     logger.addHandler(console_handler)
#     logger.addHandler(file_handler)

#     # ============================
#     # REDUCE NOISE FROM LIBRARIES
#     # ============================
#     logging.getLogger("urllib3").setLevel(logging.WARNING)
#     logging.getLogger("sklearn").setLevel(logging.WARNING)
#     logging.getLogger("uvicorn").setLevel(logging.INFO)

#     return logger


# # ================================
# # LOGGER FACTORY (USED IN ALL MODULES)
# # ================================
# def get_logger(name: str = __name__) -> logging.Logger:
#     return logging.getLogger(name)