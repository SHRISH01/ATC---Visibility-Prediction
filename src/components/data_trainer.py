import os
import sys
from dataclasses import dataclass

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# Importing the required algorithms we need to apply onto our dataset
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import setup_logger
from src.utils import save_object, evaluate_model

logger = setup_logger()

# We will use this class to load the data and apply the required algorithms

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, valid_arr):
        try:
            logger.info("Splitting training and test input data")
            X_train, y_train, X_valid, y_valid = (
                train_arr[:, :-1],
                train_arr[:, -1],
                valid_arr[:, :-1],
                valid_arr[:, -1]
            )

            models = {
                "DecisionTreeRegressor":GradientBoostingRegressor(max_depth=5, min_samples_split=10, n_estimators=300,
                          subsample=0.8)
            }

            model_report: dict = evaluate_model(
                X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
                models=models
            )

            # To get the best model score from the model dict:
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the models dict:
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logger.info("Training the best model")
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            predicted = best_model.predict(X_valid)
            logger.info("Model trained successfully")
            r2_score_ = r2_score(y_valid, predicted)
            return r2_score_ 

        except Exception as e:
            # Handle the exception with proper message and potentially traceback
            message = f"Error occured in python script name: {sys.argv[0]}"
            if e is not None:
                # Include traceback information if available
                message += f"\nLine number: {sys.exc_info()[2].tb_lineno}"
            raise CustomException(message)