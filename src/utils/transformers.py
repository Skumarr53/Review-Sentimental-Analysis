from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer


class TFIDF_BERT_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, bert_model_name='distilbert-base-nli-stsb-mean-tokens'):
        self.bert_model_name = bert_model_name
        self.model = None  # Initialize as None
    
    def _initialize_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.bert_model_name)

    def fit(self, X, y=None):
        self._initialize_model()  # Ensure model is initialized
        return self
    
    def transform(self, X):
        self._initialize_model()  # Ensure model is initialized
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