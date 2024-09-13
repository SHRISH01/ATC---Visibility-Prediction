import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Setting project root and updating the system path to import necessary modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import setup_logger
from src.utils import save_object

logger = setup_logger()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def load_data(self, train_path: str, test_path: str):
        """Load training and test data from specified paths."""
        try:
            logger.info("Loading data from paths.")
            print("Loading data from paths...")  # Debugging print statement

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            print("Train Data Head:\n", train_data.head())
            print("Test Data Head:\n", test_data.head())

            X_train = train_data.drop(columns=['VISIBILITY', 'DATE'])
            y_train = train_data['VISIBILITY']

            X_test = test_data.drop(columns=['VISIBILITY', 'DATE'])
            y_test = test_data['VISIBILITY']

            logger.info("Data loading completed.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error("Error occurred while loading data")
            print("Error occurred while loading data:", e)  # Debugging print statement
            raise CustomException(e, sys)

    def drop_highly_correlated_columns(self, df, threshold=0.9):
        """Drop columns that are highly correlated with each other based on the given threshold."""
        try:
            logger.info("Dropping highly correlated columns.")
            print("Dropping highly correlated columns...")  # Debugging print statement

            corr_matrix = df.corr().abs()
            print("Correlation Matrix:\n", corr_matrix)  # Debugging print statement

            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

            df_dropped = df.drop(columns=to_drop)

            logger.info("Dropped columns: %s", to_drop)
            print("Dropped columns:", to_drop)  # Debugging print statement
            return df_dropped

        except Exception as e:
            logger.error("Error occurred while dropping highly correlated columns")
            print("Error occurred while dropping highly correlated columns:", e)  # Debugging print statement
            raise CustomException(e, sys)

    def feature_engineering(self, df):
        """Perform feature engineering, including dropping highly correlated columns."""
        try:
            logger.info("Starting feature engineering.")
            print("Starting feature engineering...")  # Debugging print statement

            df = self.drop_highly_correlated_columns(df)

            logger.info("Feature engineering completed.")
            print("Feature engineering completed.")  # Debugging print statement
            return df

        except Exception as e:
            logger.error("Error occurred during feature engineering")
            print("Error occurred during feature engineering:", e)  # Debugging print statement
            raise CustomException(e, sys)
    
    def preprocess_data(self, X_train, X_test, y_train):
        """Preprocess the data using a pipeline of scaling and feature selection."""
        try:
            logger.info("Starting data preprocessing.")
            print("Starting data preprocessing...")  # Debugging print statement

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(score_func=f_regression, k='all'))
            ])

            X_train_processed = pipeline.fit_transform(X_train, y_train)
            print("X_train_processed shape:", X_train_processed.shape)  # Debugging print statement

            X_test_processed = pipeline.transform(X_test)
            print("X_test_processed shape:", X_test_processed.shape)  # Debugging print statement

            save_object(pipeline, self.data_transformation_config.preprocessor_obj_file_path)
            logger.info(f"Preprocessor pipeline saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            logger.info("Data preprocessing completed.")
            return X_train_processed, X_test_processed

        except Exception as e:
            logger.error("Error occurred during data preprocessing")
            print("Error occurred during data preprocessing:", e)  # Debugging print statement
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """Initiate the data transformation process with the given train and test data paths."""
        try:
            X_train, X_test, y_train, y_test = self.load_data(train_path, test_path)
            X_train = self.feature_engineering(X_train)
            X_test = self.feature_engineering(X_test)
            X_train_processed, X_test_processed = self.preprocess_data(X_train, X_test, y_train)

            print("X_train_processed shape:", X_train_processed.shape)
            print("X_test_processed shape:", X_test_processed.shape)

            return (
                X_train_processed,
                X_test_processed,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logger.error("Error occurred in data transformation process")
            print("Error occurred in data transformation process:", e)  # Debugging print statement
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Specify paths for train and test data
        train_path = "artifacts/train.csv"
        test_path = "artifacts/test.csv"

        data_transformation = DataTransformation()
        # Pass paths as arguments to the initiate_data_transformation method
        data_transformation.initiate_data_transformation(train_path, test_path) 

    except Exception as e:
        logger.error("Error occurred in data transformation script")
        print("Error occurred in data transformation script:", e)  # Debugging print statement
        raise CustomException(e, sys)
