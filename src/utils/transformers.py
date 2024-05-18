from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertModel
import spacy
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class TFIDF_BERT_transformer(BaseEstimator, TransformerMixin):

    def __init__(self, bert_model_name='distilbert-base-nli-stsb-mean-tokens', TFIDF_model=None):
        self.bert_model_name = bert_model_name
        self.model = SentenceTransformer(self.bert_model_name)
        self.embeddings_cache = {}

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.model.encode(X)


class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

class ColumnExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols