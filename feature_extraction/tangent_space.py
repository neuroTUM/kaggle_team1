import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TangentSpaceFeatures(TransformerMixin, BaseEstimator):
    """features per frequency band with riemannian tangent spaces"""

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
        check_is_fitted(self)  # clear error if transform is called before fit
        features = []
        for band, tangent_space in enumerate(self.tangent_spaces_):
            covariances = self.covariances_.transform(X[:, band])
            features.append(tangent_space.transform(covariances))
        return np.hstack(features)
