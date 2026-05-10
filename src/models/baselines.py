import numpy as np


class OrdinaryLeastSquares:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction.")
        return X @ self.weights


class RidgeRegression:
    def __init__(self, lam=1.0):
        self.lam = lam
        self.weights = None

    def fit(self, X, y):
        n_features = X.shape[1]

        penalty = self.lam * np.eye(n_features)
        penalty[0, 0] = 0.0  # do not penalize intercept

        self.weights = np.linalg.pinv(X.T @ X + penalty) @ X.T @ y
        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction.")
        return X @ self.weights