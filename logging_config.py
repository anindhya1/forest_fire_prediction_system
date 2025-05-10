# logging_config.py
import logging
import os


def setup_logging():
    """
    Configure logging for the application
    """
    # Ensure log directory exists
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'forest_fire_prediction.log')),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# Use in application.py
logger = setup_logging()