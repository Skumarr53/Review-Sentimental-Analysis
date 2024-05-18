import os
import sys

import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import DetailedError
from dataclasses import dataclass

from src.utils.utils import save_object, load_object
from src.utils.helper import text_cleaning, df_remove_nans

@dataclass
class DataCleanerConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    preprocessed_train_file_path: str = os.path.join('artifacts', 'train_processed.csv')
    preprocessed_test_file_path: str = os.path.join('artifacts', 'test_processed.csv')

class DataCleaner:
    def __init__(self):
        self.data_transformation_config = DataCleanerConfig()

    def apply_text_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text cleaning to the 'text' and 'Summary' columns of the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        data = df_remove_nans(data)
        data['Text'] = data['Text'].apply(text_cleaning)
        data['Summary'] = data['Summary'].apply(text_cleaning)
        return data

    def save_processed_data(self, data: pd.DataFrame, path: str) -> None:
        """
        Save the transformed data to a CSV file.

        Args:
            data (pd.DataFrame): The DataFrame to save.
            path (str): The path to save the CSV file.

        Returns:
            None
        """
        try:
            data.to_csv(path, index=False)
            logging.info(f"Data saved to {path}.")
        except Exception as e:
            logging.error(f"Error saving data to {path}.")
            raise DetailedError(e)

    def data_processing(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """
        Transform the data by applying text cleaning and saving the transformed data.

        Args:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Testing data.

        Returns:
            None
        """
        logging.info("Data Transformation Started")
        try:
            logging.info("Applying text cleaning to the training data")
            train_data = self.apply_text_cleaning(train_data)
            logging.info("Applying text cleaning to the testing data")
            test_data = self.apply_text_cleaning(test_data)

            logging.info("Saving the transformed training data")
            self.save_processed_data(train_data, self.data_transformation_config.preprocessed_train_file_path)
            logging.info("Saving the transformed testing data")
            self.save_processed_data(test_data, self.data_transformation_config.preprocessed_test_file_path)
            logging.info("Data Transformation Completed")
        except Exception as e:
            logging.error("Error in data transformation")
            raise DetailedError(e)

if __name__ == "__main__":
    # Example usage:
    train_data = pd.read_csv('artifacts/train.csv')
    test_data = pd.read_csv('artifacts/test.csv')
    
    data_processing = DataCleaner()
    data_processing.data_processing(train_data, test_data)