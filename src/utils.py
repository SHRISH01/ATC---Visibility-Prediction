import dill
import os
from src.logger import setup_logger  
logger = setup_logger()

def save_object(obj, file_path):
    """
    Save a Python object as a pickle file using dill.

    Parameters:
    - obj: The Python object to save
    - file_path: The path where the object will be saved
    """
    try:
        logger.info(f"Saving object to {file_path}.")
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logger.info("Object saved successfully.")

    except Exception as e:
        logger.error(f"Error occurred while saving object to {file_path}: {e}")
        raise

def load_object(file_path):
    """
    Load a Python object from a pickle file using dill.

    Parameters:
    - file_path: The path from which the object will be loaded

    Returns:
    - The loaded Python object
    """
    try:
        logger.info(f"Loading object from {file_path}.")
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        logger.info("Object loaded successfully.")
        return obj

    except Exception as e:
        logger.error(f"Error occurred while loading object from {file_path}: {e}")
        raise
