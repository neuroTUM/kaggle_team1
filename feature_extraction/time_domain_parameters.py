import numpy as np

from .feature_extraction_class import FeatureExtractionClass


class TimeDomainParameters(FeatureExtractionClass):
    """Log variance of the signal and of its first `order` time derivatives, per band and channel"""

    def __init__(self, order):
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        parameters = [
            np.log(np.var(np.diff(X, n=derivative, axis=-1), axis=-1))
            for derivative in range(self.order + 1)
        ]
        return np.stack(parameters, axis=-1).reshape(len(X), -1)
