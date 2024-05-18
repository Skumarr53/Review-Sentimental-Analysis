import os, yaml, json
from dataclasses import dataclass

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from src.logger.logger import logging
from src.exception.exception import DetailedError


from src.utils.utils import evaluate_classification

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from src.config import constants
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    transformed_train_file_path = os.path.join('artifacts', 'train_transformed.npy')
    transformed_test_file_path = os.path.join('artifacts', 'test_transformed.npy')
    param_file_path = os.path.join('params.yaml')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.params = self.load_params()
        self.client = MlflowClient()
        self.model_name = constants.MLFLOW_MODEL_NAME

    def load_params(self):
        print(os.getcwd)
        with open(self.model_trainer_config.param_file_path, 'r') as file:
             params = yaml.safe_load(file)
        return params
    
    def log_metrics(self, y_test, y_pred):
        accuracy, precision, recall, f1 = evaluate_classification(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)

    def get_model_and_params(self):
        models = {
            'LogisticRegression': (LogisticRegression(), self.params['model_training']['LogisticRegression']),
            'NaiveBayes': (GaussianNB(), {k: [float(i) for i in v] for k, v in self.params['model_training']['NaiveBayes'].items()}),
            'SVM': (SVC(), self.params['model_training']['SVM'])
        }
        return models
    
    def register_model_version(self):

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        # Check if the registered model already exists
        try:
            self.client.get_registered_model(self.model_name)
            model_exists = True
        except MlflowException:
            model_exists = False

        if not model_exists:
            # Create the registered model if it does not exist
            self.client.create_registered_model(self.model_name)

        # Create a new version of the model
        model_version = self.client.create_model_version(self.model_name, model_uri, model_uri)
        
        self.client.transition_model_version_stage(
                            name=self.model_name,
                            version=model_version.version,
                            stage="Production"
                        )
        logging.info(f'Registered Model: {self.model_name} with Version: {model_version.version}')

    def initiate_model_training(self):
        try:
            # MLflow Tracking
            mlflow.set_experiment(constants.MLFLOW_EXP)

            # Load preprocessed data
            train_array = np.load(self.model_trainer_config.transformed_train_file_path)
            test_array = np.load(self.model_trainer_config.transformed_test_file_path)

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = self.get_model_and_params()

            best_model_name = None
            best_score = -np.inf
            best_model = None

            for model_name, (model, params) in tqdm(models.items()):
                with mlflow.start_run():
                    logging.info(f"Training {model_name} with hyperparameter tuning")
                    mlflow.log_param("model_name", model_name)

                    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='roc_auc', cv=3)
                    grid_search.fit(X_train, y_train)
                    
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)

                    mlflow.log_params(grid_search.best_params_)
                    self.log_metrics(y_test, y_pred)

                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model = grid_search.best_estimator_
                        mlflow.sklearn.log_model(best_model, "model")
                    
                    self.register_model_version()
            
        except Exception as e:
            logging.error('Exception occurred during model training', exc_info=True)
            raise DetailedError(e)

if __name__ == '__main__':
    obj = ModelTrainer()
    obj.initiate_model_training()