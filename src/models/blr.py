import numpy as np


class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None

    def fit(self, X, y):
        n_features = X.shape[1]

        A = self.alpha * np.eye(n_features) + self.beta * (X.T @ X)

        self.S_N = np.linalg.inv(A)
        self.m_N = self.beta * self.S_N @ X.T @ y

        return self

    def predict(self, X, return_std=False):
        if self.m_N is None or self.S_N is None:
            raise ValueError("Model must be fitted before prediction.")

        mean = X @ self.m_N
        variance = (1.0 / self.beta) + np.sum((X @ self.S_N) * X, axis=1)
        std = np.sqrt(variance)

        if return_std:
            return mean, std

        return mean

    def fit_evidence(self, X, y, max_iter=100, tol=1e-5):
        n_samples, n_features = X.shape

        alpha = self.alpha
        beta = self.beta

        eigenvalues = np.linalg.eigvalsh(X.T @ X)

        for iteration in range(max_iter):
            alpha_old = alpha
            beta_old = beta

            self.alpha = alpha
            self.beta = beta
            self.fit(X, y)

            gamma = np.sum((beta * eigenvalues) / (alpha + beta * eigenvalues))

            weight_norm = self.m_N.T @ self.m_N
            residuals = y - X @ self.m_N
            sse = residuals.T @ residuals

            alpha = gamma / weight_norm
            beta = (n_samples - gamma) / sse

            if abs(alpha - alpha_old) < tol and abs(beta - beta_old) < tol:
                break

        self.alpha = alpha
        self.beta = beta
        self.fit(X, y)

        return self