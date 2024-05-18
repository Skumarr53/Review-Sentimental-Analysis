import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from src.config.constants import LABEL_MAP
from src.utils.helper import df_remove_nans
from src.utils.utils import save_object
from src.utils.transformers import TFIDF_BERT_transformer, ColumnExtractor, Converter
from src.logger.logger import logging
from src.exception.exception import DetailedError

@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'transformer.pkl')
    preprocessed_train_file_path: str = os.path.join('artifacts', 'train_processed.csv')
    preprocessed_test_file_path: str = os.path.join('artifacts', 'test_processed.csv')
    transformed_train_file_path: str = os.path.join('artifacts', 'train_transformed.npy')
    transformed_test_file_path: str = os.path.join('artifacts', 'test_transformed.npy')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformConfig()

    def get_avg_data_transformation_object(self):
        try:
            trans_pipeline = Pipeline([
                ('features',FeatureUnion([
                    ('Summary_vectorizer',Pipeline([
            ('extract',ColumnExtractor(['Summary_Review'])),
            ('converter', Converter()),
            ('vectorize',TFIDF_BERT_transformer()),
        ]))]))])
            return trans_pipeline
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise DetailedError(e)
    
    def initiate_datatransformation(self):
        try:
            train_df = pd.read_csv(self.data_transformation_config.preprocessed_train_file_path)
            test_df = pd.read_csv(self.data_transformation_config.preprocessed_test_file_path)

            train_df = df_remove_nans(train_df)
            test_df = df_remove_nans(test_df)

            train_df['Summary_Review'] = train_df['Summary'] +'. '+ train_df['Text']
            test_df['Summary_Review'] = test_df['Summary'] + '. ' + test_df['Text']
        
            logging.info("Data Transformation initiated")
            trans_pipeline = self.get_avg_data_transformation_object()

            target_column_name = 'Score'

            target_feature_train_df = train_df[target_column_name].map(LABEL_MAP)
            target_feature_test_df = test_df[target_column_name].map(LABEL_MAP)
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = trans_pipeline.fit_transform(train_df)
            input_feature_test_arr = trans_pipeline.transform(test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(trans_pipeline, self.data_transformation_config.preprocessor_obj_file_path)
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr) 

            logging.info("Data Transformation completed")
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise DetailedError(e)
        

if __name__ == "__main__":
    obj = DataTransformation(DataTransformConfig())
    obj.initiate_datatransformation('artifacts/train_processed.csv', 'artifacts/test_processed.csv')