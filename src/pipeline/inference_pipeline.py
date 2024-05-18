import os
import sys
import json
import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import DetailedError
from src.utils.utils import load_object
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from src.config import constants
from src.utils.helper import text_cleaning
from pdb import set_trace

class PredictionPipeline:
  def __init__(self):
    self.latest_version_info = MlflowClient().get_latest_versions(name=constants.MLFLOW_MODEL_NAME, stages=["Production"])[0]
    self.transformer = load_object(os.path.join("artifacts", "transformer.pkl"))
    self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{constants.MLFLOW_MODEL_NAME}/{self.latest_version_info.version}")

    
  def predict(self, text):
    try:
      text = text_cleaning(text[0])
      features = pd.DataFrame({'Summary_Review': [text]})
      trans_features = self.transformer.transform(features)
      
      prediction = self.model.predict(trans_features)

      return prediction

    except Exception as e:
      raise DetailedError(e)
    
if __name__ == "__main__":
  obj = PredictionPipeline()
  prediction = obj.predict(["This is a test review"])
  print(prediction)