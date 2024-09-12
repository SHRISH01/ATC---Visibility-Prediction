
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from dataclasses import dataclass, field
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)
from src.logger import setup_logger  
logger = setup_logger()
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataPaths:
    train_path: str
    test_path: str
    preprocessor_path: str = field(default="artifacts/preprocessor.pkl")

def load_data(paths: DataPaths):
    # Your existing load_data function with paths.train_path and paths.test_path
    # as inputs, and other minor adjustments as needed.
    try:
        logger.info("Loading data from paths.")
        print("Loading data from paths...")  # Debugging print statement
        
        # Load the data from CSV files
        train_data = pd.read_csv(paths.train_path)
        test_data = pd.read_csv(paths.test_path)
        
        # Print the first few rows to verify
        print("Train Data Head:\n", train_data.head())
        print("Test Data Head:\n", test_data.head())
        
        # Drop the target column and DATE column from features
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

# Drop correlated columns function
def drop_highly_correlated_columns(df, threshold=0.9):
    """
    Drop columns from the dataframe that are highly correlated with each other.

    Parameters:
    - df: The dataframe to process
    - threshold: Correlation coefficient above which columns will be dropped

    Returns:
    - DataFrame with highly correlated columns removed
    """
    try:
        logger.info("Dropping highly correlated columns.")
        print("Dropping highly correlated columns...")  # Debugging print statement

        # Compute the absolute correlation matrix
        corr_matrix = df.corr().abs()
        print("Correlation Matrix:\n", corr_matrix)  # Debugging print statement

        # Create a mask to identify highly correlated columns
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

        # Drop highly correlated columns
        df_dropped = df.drop(columns=to_drop)

        logger.info("Dropped columns: %s", to_drop)
        print("Dropped columns:", to_drop)  # Debugging print statement
        return df_dropped

    except Exception as e:
        logger.error("Error occurred while dropping highly correlated columns")
        print("Error occurred while dropping highly correlated columns:", e)  # Debugging print statement
        raise CustomException(e, sys)

def feature_engineering(df):
    """
    Perform feature engineering on the dataframe.

    Parameters:
    - df: The dataframe to process

    Returns:
    - DataFrame with feature engineering applied
    """
    try:
        logger.info("Starting feature engineering.")
        print("Starting feature engineering...")  # Debugging print statement

        # Drop unnecessary columns
        df = drop_highly_correlated_columns(df)

        logger.info("Feature engineering completed.")
        print("Feature engineering completed.")  # Debugging print statement
        return df

    except Exception as e:
        logger.error("Error occurred during feature engineering")
        print("Error occurred during feature engineering:", e)  # Debugging print statement
        raise CustomException(e, sys)
    

def preprocess_data(X_train, X_test, y_train, preprocessor_path):
    """
    Apply preprocessing to the training and test data and save the preprocessor.

    Parameters:
    - X_train: Training feature data
    - X_test: Test feature data
    - y_train: Training target data
    - preprocessor_path: Path where the preprocessor pipeline will be saved

    Returns:
    - Processed training and test feature data
    """
    try:
        logger.info("Starting data preprocessing.")
        print("Starting data preprocessing...")  # Debugging print statement

        # Define the pipeline with preprocessing steps and feature selection
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Example preprocessing step
            ('feature_selection', SelectKBest(score_func=f_regression, k='all'))  # Adjust k as needed
        ])

        # Fit the pipeline on training data
        X_train_processed = pipeline.fit_transform(X_train, y_train)  # Pass y_train here
        print("X_train_processed shape:", X_train_processed.shape)  # Debugging print statement

        # Transform the test data
        X_test_processed = pipeline.transform(X_test)
        print("X_test_processed shape:", X_test_processed.shape)  # Debugging print statement

        # Save the pipeline as a pickle file
        save_object(pipeline, preprocessor_path)
        logger.info(f"Preprocessor pipeline saved at {preprocessor_path}")

        logger.info("Data preprocessing completed.")
        return X_train_processed, X_test_processed

    except Exception as e:
        logger.error("Error occurred during data preprocessing")
        print("Error occurred during data preprocessing:", e)  # Debugging print statement
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Initialize data paths with dataclass
        paths = DataPaths(
            train_path='artifacts/train.csv',  # Update with your actual path
            test_path='artifacts/test.csv',    # Update with your actual path
        )

        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_data(paths)
        X_train = feature_engineering(X_train)
        X_test = feature_engineering(X_test)
        X_train_processed, X_test_processed = preprocess_data(X_train, X_test, y_train, paths.preprocessor_path)

        # Print the final processed data shapes
        print("X_train_processed shape:", X_train_processed.shape)
        print("X_test_processed shape:", X_test_processed.shape)

        # Proceed with the next steps (e.g., model training)
        # ...

    except Exception as e:
        logger.error("Error occurred in data transformation script")
        print("Error occurred in data transformation script:", e)  # Debugging print statement
        raise CustomException(e, sys)
