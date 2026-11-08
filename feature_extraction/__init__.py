import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin


class TangentSpaceFeatures(TransformerMixin, BaseEstimator):
    """Covariance per band mapped to the tangent space: (trials, bands, channels, times) -> (trials, features)

    Each band gets its own reference point, the Riemannian mean of the training covariances of that band.
    With 8 channels every band gives 8 * 9 / 2 = 36 features.
    """

    def __init__(self, estimator="oas", metric="riemann"):
        self.estimator = estimator
        self.metric = metric

    def fit(self, X, y=None):
        self.covariances_ = Covariances(estimator=self.estimator)
        self.tangent_spaces_ = []
        for band in range(X.shape[1]):
            covariances = self.covariances_.transform(X[:, band])
            self.tangent_spaces_.append(TangentSpace(metric=self.metric).fit(covariances))
        return self

    def transform(self, X):
        features = []
        for band, tangent_space in enumerate(self.tangent_spaces_):
            covariances = self.covariances_.transform(X[:, band])
            features.append(tangent_space.transform(covariances))
        return np.hstack(features)
