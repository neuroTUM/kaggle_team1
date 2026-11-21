

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)