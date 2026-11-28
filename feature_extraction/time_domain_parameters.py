import numpy as np

from .feature_extraction_class import FeatureExtractionClass


class TimeDomainParameters(FeatureExtractionClass):
    """Log variance of the signal and of its first `order` time derivatives, per band and channel

    neighbours: per channel, the indices of the channels whose mean is subtracted first
    (Laplacian); None keeps the channels as they are
    """

    def __init__(self, order, neighbours=None):
        self.order = order
        self.neighbours = neighbours

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.neighbours is not None:
            X = laplacian(X, self.neighbours)
        parameters = [
            np.log(np.var(np.diff(X, n=derivative, axis=-1), axis=-1))
            for derivative in range(self.order + 1)
        ]
        return np.stack(parameters, axis=-1).reshape(len(X), -1)


def laplacian(X, neighbours):
    """Each channel minus the mean of its neighbour channels, along the channel axis"""
    means = [X[..., list(indices), :].mean(axis=-2) for indices in neighbours]
    return X - np.stack(means, axis=-2)
