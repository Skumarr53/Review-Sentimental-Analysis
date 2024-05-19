from __future__ import annotations
import os, json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.logger.logger import logging
from src.exception.exception import DetailedError

from torch.multiprocessing import set_start_method

# Set the start method for multiprocessing to 'spawn'
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataCleaner 
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Define the default arguments
default_args = {
    'retries': 3,
    'retry_delay': pendulum.Duration(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    'sentimental_analysis_training_pipeline',
    default_args=default_args,
    description="Review classification training pipeline",
    schedule_interval='@weekly',
    start_date=pendulum.datetime(2024, 4, 24, tz="UTC"),
    catchup=False,
    tags=["machine_learning", "classification", "sentimental analysis"],
) as dag:
    dag.doc_md = __doc__

    def training_ingestion(**kwargs):
        """Ingest training data into the DAG."""
        try:
            ti = kwargs['ti']
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()
            ti.xcom_push('data_ingestion_artifact', {
                '_train_data': train_data,
                '_test_data': test_data
            })
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise DetailedError(f"Data ingestion failed: {str(e)}")

    def training_preprocessing(**kwargs):
        """Perform data preprocessing as part of the training pipeline."""
        try:
            ti = kwargs['ti']
            ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifact")
            train_data, test_data = ingestion_artifact["_train_data"], ingestion_artifact["_test_data"]

            data_preprocessor = DataCleaner()
            data_preprocessor.data_processing(train_data, test_data)
        except Exception as e:
            logging.error(f"Error during data preprocessing: {str(e)}")
            raise DetailedError(f"Data preprocessing failed: {str(e)}")

    def training_transformation(**kwargs):
        """Perform data transformation as part of the training pipeline."""
        try:
            data_transformation = DataTransformation()
            data_transformation.initiate_datatransformation()
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise DetailedError(f"Data transformation failed: {str(e)}")
    
    def training_model(**kwargs):
        """Train the model as part of the training pipeline."""
        try:
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer()
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise DetailedError(f"Model training failed: {str(e)}")

    # Define the tasks
    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=training_ingestion,
    )

    data_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=training_preprocessing,
    )

    data_transformation = PythonOperator(
        task_id='data_transformation',
        python_callable=training_transformation,
    )

    model_trainer = PythonOperator(
        task_id='model_trainer',
        python_callable=training_model,
    )

    # Set task dependencies
    data_ingestion >> data_preprocessing >> data_transformation >> model_trainer
