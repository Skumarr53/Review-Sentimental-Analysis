import os
import sqlite3
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger.logger import logging
from src.exception.exception import DetailedError

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    sql_path: str = os.path.join("artifacts", "database.sqlite")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def data_prep(self, data: pd.DataFrame) -> pd.DataFrame:
      """
      Prepare the data by converting the 'Score' column to binary classification.

      Args:
          data (pd.DataFrame): Raw data with the 'Score' column.

      Returns:
          pd.DataFrame: Data with binary classification in the 'Score' column.
      """
      # Refactor the code to improve readability and maintainability
      try:
          data = data[~(data['Summary'].isna() | data['Text'].isna())]
          data['Score'] = np.where(data['Score'] > 3, 'positive', 'negative')
          data = data.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"], keep='first', inplace=False)
          data = data[data.HelpfulnessNumerator <= data.HelpfulnessDenominator]
          logging.info("Data prep completed.")
          return data
      except Exception as e:
        logging.error("Error in data prep")
        raise DetailedError(e)


    def read_data_from_sql(self) -> pd.DataFrame:
        """
        Read data from the SQLite database.

        Returns:
            pd.DataFrame: Data read from the SQL table.
        """
        try:
            con = sqlite3.connect(self.ingestion_config.sql_path)
            raw_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""", con)
            con.close()
            logging.info("Read the SQL table as DataFrame.")
            return raw_data
        except Exception as e:
            logging.error("Error reading data from SQL database.")
            raise DetailedError(e)

    def save_data_to_csv(self, data: pd.DataFrame, path: str) -> None:
        """
        Save the DataFrame to a CSV file.

        Args:
            data (pd.DataFrame): Data to save.
            path (str): Path to save the CSV file.
        """
        try:
            data.to_csv(path, index=False, header=True)
            logging.info(f"Data saved to {path}.")
        except Exception as e:
            logging.error(f"Error saving data to {path}.")
            raise DetailedError(e)
        
    def sample_balanced_data(self, data):
        # Select 7,500 positive samples
        positive_samples = data[data['Score'] == 'positive'].sample(n=7500, random_state=42)

        # Select 7,500 negative samples
        negative_samples = data[data['Score'] == 'negative'].sample(n=7500, random_state=42)

        # Combine positive and negative samples into a balanced dataset
        balanced_data = pd.concat([positive_samples, negative_samples])
        return balanced_data
    
    def split_data(self, data: pd.DataFrame) -> tuple:
        """
        Split the data into training and testing sets.

        Args:
            data (pd.DataFrame): Data to split.

        Returns:
            tuple: Paths to the train and test datasets.
        """
        try:
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
            self.save_data_to_csv(train_set, self.ingestion_config.train_data_path)
            self.save_data_to_csv(test_set, self.ingestion_config.test_data_path)
            logging.info("Train-test split completed.")
            return (train_set, train_set)
        except Exception as e:
            logging.error("Error during train-test split.")
            raise DetailedError(e)
    
    def initiate_data_ingestion(self) -> tuple:
        """
        Initiate the data ingestion process.

        Returns:
            tuple: Paths to the train and test datasets.
        """
        logging.info("Data ingestion started.")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            raw_data = self.read_data_from_sql()
            raw_data = self.data_prep(raw_data)
            logging.info("Data preparation completed.")

            ##  craete suample balanced data 15k size
            raw_data = self.sample_balanced_data(raw_data)
            
            self.save_data_to_csv(raw_data, self.ingestion_config.raw_data_path)
            logging.info("Raw data saved.")
            
            return self.split_data(raw_data)
        except Exception as e:
            logging.error("Exception occurred during data ingestion.")
            raise DetailedError(e)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
