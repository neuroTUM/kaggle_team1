from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractionClass(ABC, TransformerMixin, BaseEstimator):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

