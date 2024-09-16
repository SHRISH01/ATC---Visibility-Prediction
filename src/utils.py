import dill
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import setup_logger  
from src.exception import CustomException
logger = setup_logger()
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

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

def evaluate_model(X_train, y_train, X_valid, y_valid, models):
    try:
        report = {}
        print("Training Shape:", X_train.shape)
        print("Validation Shape:", X_valid.shape)

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_valid)

            model_train_score = r2_score(y_train, y_train_pred)
            model_test_score = r2_score(y_valid, y_test_pred)

            report[list(models.keys())[i]] = model_test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    def load_object(file_path):
        try:
            with open(file_path,"rb") as file_obj:
                return dill.load(file_obj)
                
        except Exception as e :
            raise CustomException(e, sys)




