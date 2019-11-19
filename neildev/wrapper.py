from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin


class Wrapper(BaseEstimator, ClassifierMixin, TransformerMixin):

    def __init__(self, model):

        self.model = model

    def fit(self, X, y, sample_weight=None):

        model.fit(X, y, sample_weight=sample_weight)
    
        return self

    def predict(self, X):

        return model.predict(X)

    def predict_proba(self, X):
    
        return model.predict_proba(X)
