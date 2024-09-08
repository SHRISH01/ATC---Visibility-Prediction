import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(logs_path: str = 'logs', log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with rotating file handler and console handler.

    Args:
        logs_path (str): Path to the directory where logs will be stored.
        log_level (int): Logging level. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger.
    """

    # Create logs directory if it doesn't exist
    os.makedirs(logs_path, exist_ok=True)
    
    # Define log file name with date and time
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(logs_path, log_filename)

    # Set up the logger
    logger = logging.getLogger('ML_Project_Logger')

    # Check if the logger already has handlers to avoid duplicates
    if not logger.hasHandlers():
        logger.setLevel(log_level)

        # Create handlers
        file_handler = RotatingFileHandler(
            log_filepath, 
            maxBytes=5 * 1024 * 1024,  # 5MB per log file
            backupCount=5  # Keep the last 5 log files
        )
        console_handler = logging.StreamHandler()

        # Define log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger(logs_path='logs', log_level=logging.INFO)
    logger.info("Logger is configured and ready to use.")
