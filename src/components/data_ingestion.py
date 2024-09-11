import sys
import os
from src.logger import setup_logger  # Ensure this is correctly imported
logger = setup_logger()
from src.exception import CustomException  # Ensure this is correctly imported
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info('Data Ingestion process started.')
        try:
            # Ensure the dataset file path is correct
            data_file_path = os.path.join('Dataset', 'NOAA-JKF-AP.csv')
            if not os.path.exists(data_file_path):
                logger.error(f"Data file not found at path: {data_file_path}")
                raise FileNotFoundError(f"Data file not found at path: {data_file_path}")

            df = pd.read_csv(data_file_path)
            logger.info(f'Dataset loaded successfully with shape: {df.shape}')

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info('Raw data saved.')

            # Split the data into train and test sets
            logger.info('Splitting data into train and test sets.')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info('Train and test data saved successfully.')

            logger.info('Data Ingestion process completed successfully.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.exception('Exception occurred during the data ingestion process.')
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")
