import logging
from pathlib import Path

LOGGER = logging.getLogger("analysis_logger")
LOGGER.setLevel(logging.DEBUG)
log_file = Path(__file__).parent / "analysis_logger.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)
