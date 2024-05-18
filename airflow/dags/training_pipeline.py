from __future__ import annotations
import os, json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.logger.logger import logging
from src.exception.exception import DetailedError

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataCleaner 
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


with DAG(
  'Sentimental_analysis_training_pipeline',
  default_args={
    'retries': 0,
    'retry_delay': pendulum.Duration(minutes=5),
  },
    description="Review classification training pipeline",
  schedule='@weekly',
  start_date=pendulum.datetime(2024, 4, 24, tz="UTC"),
  catchup=False,
    tags=["machine_learning ", "classification","sentimental analysis"],
) as dag:
    dag.doc_md = __doc__

    def training_ingestion(**kwargs):
        """Ingest training data into the DAG."""
        ti = kwargs['ti']
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()
        ti.xcom_push('data_ingestion_artifact', {
            '_train_data': train_data,
            '_test_data': test_data
        })

    def training_preprocessing(**kwargs):
        """
        Perform data preprocessing as part of the training pipeline.
        """
        ti = kwargs['ti']
        ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifact")
        train_data, test_data = ingestion_artifact["_train_data"], ingestion_artifact["_test_data"]

        data_preprocessor = DataCleaner()
        data_preprocessor.data_processing(train_data, test_data)

    def training_transformation(**kwargs):
        ti = kwargs['ti']
        data_transformation = DataTransformation()
        data_transformation.initiate_datatransformation()
    
    def training_model(**kwargs):
        ti = kwargs['ti']
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer()
    
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

    data_ingestion >> data_preprocessing >> data_transformation >> model_trainer
