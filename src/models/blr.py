import numpy as np


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with Gaussian likelihood and Gaussian prior.

    Model:
        y | X, w, beta ~ N(Xw, beta^{-1} I)
        w | alpha      ~ N(0, alpha^{-1} I)

    Parameters
    ----------
    alpha : float
        Prior precision. Larger alpha means stronger shrinkage.
    beta : float
        Noise precision. Larger beta means lower assumed observation noise.
    fit_intercept_prior : bool
        If False, the intercept is weakly regularized instead of using the same prior
        precision as the other coefficients.
    """

    def __init__(self, alpha=1.0, beta=1.0, fit_intercept_prior=False):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fit_intercept_prior = fit_intercept_prior

        self.posterior_mean_ = None
        self.posterior_cov_ = None
        self.prior_precision_ = None
        self.log_evidence_ = None
        self.n_iter_ = 0

    def _make_prior_precision(self, n_features):
        """
        Construct prior precision matrix.

        By default, the intercept is almost unregularized because it only represents
        the global mean level of the target.
        """
        prior_precision = self.alpha * np.eye(n_features)

        if not self.fit_intercept_prior:
            prior_precision[0, 0] = 1e-8

        return prior_precision

    def fit(self, X, y):
        """
        Fit BLR posterior in closed form.

        Posterior:
            S_N = (S_0^{-1} + beta X^T X)^{-1}
            m_N = beta S_N X^T y
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_features = X.shape[1]

        self.prior_precision_ = self._make_prior_precision(n_features)

        posterior_precision = self.prior_precision_ + self.beta * (X.T @ X)

        self.posterior_cov_ = np.linalg.pinv(posterior_precision)
        self.posterior_mean_ = self.beta * self.posterior_cov_ @ X.T @ y

        self.log_evidence_ = self.log_marginal_likelihood(X, y)

        return self

    def predict(self, X, return_std=False, return_interval=False, credibility=0.95):
        """
        Posterior predictive distribution.

        For each test point x_*:

            p(y_* | x_*, X, y) = N(mean, variance)

        where:
            mean = x_*^T m_N
            variance = beta^{-1} + x_*^T S_N x_*
        """
        if self.posterior_mean_ is None or self.posterior_cov_ is None:
            raise ValueError("Model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)

        mean = X @ self.posterior_mean_
        variance = (1.0 / self.beta) + np.sum((X @ self.posterior_cov_) * X, axis=1)
        variance = np.maximum(variance, 1e-12)
        std = np.sqrt(variance)

        if return_interval:
            if credibility != 0.95:
                raise NotImplementedError("Only 95% intervals are currently implemented.")

            lower = mean - 1.96 * std
            upper = mean + 1.96 * std

            if return_std:
                return mean, std, lower, upper

            return mean, lower, upper

        if return_std:
            return mean, std

        return mean

    def log_marginal_likelihood(self, X, y):
        """
        Compute the BLR log marginal likelihood:

            p(y | X, alpha, beta) = N(y | 0, C)

        where:
            C = beta^{-1} I + X S_0 X^T

        Equivalent stable form:

            log p(y) =
                D/2 log alpha
              + N/2 log beta
              - E(m_N)
              - 1/2 log |A|
              - N/2 log(2pi)

        where:
            A = alpha I + beta X^T X
            E(m_N) = beta/2 ||y - Xm_N||^2 + alpha/2 ||m_N||^2
        """
        if self.posterior_mean_ is None:
            return None

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        A = self.prior_precision_ + self.beta * (X.T @ X)

        sign, logdet_A = np.linalg.slogdet(A)
        if sign <= 0:
            raise np.linalg.LinAlgError("Posterior precision matrix is not positive definite.")

        residuals = y - X @ self.posterior_mean_

        # Intercept is excluded from the alpha penalty if fit_intercept_prior=False.
        prior_penalty = self.posterior_mean_.T @ self.prior_precision_ @ self.posterior_mean_

        energy = 0.5 * self.beta * (residuals.T @ residuals) + 0.5 * prior_penalty

        sign_prior, logdet_prior_precision = np.linalg.slogdet(self.prior_precision_)
        if sign_prior <= 0:
            # For the near-flat intercept prior, determinant is tiny but valid.
            logdet_prior_precision = np.sum(np.log(np.diag(self.prior_precision_)))

        log_evidence = (
            0.5 * logdet_prior_precision
            + 0.5 * n_samples * np.log(self.beta)
            - energy
            - 0.5 * logdet_A
            - 0.5 * n_samples * np.log(2.0 * np.pi)
        )

        return float(log_evidence)

    def fit_evidence(self, X, y, max_iter=500, tol=1e-6, verbose=False):
        """
        Evidence maximization using MacKay fixed-point updates.

        Updates:
            gamma = sum_i beta lambda_i / (alpha + beta lambda_i)
            alpha = gamma / ||m_N||^2
            beta  = (N - gamma) / ||y - Xm_N||^2

        Note:
        The intercept is excluded from the effective shrinkage calculation.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        # Exclude intercept from evidence shrinkage update if desired.
        if self.fit_intercept_prior:
            X_reg = X
        else:
            X_reg = X[:, 1:]

        eigenvalues = np.linalg.eigvalsh(X_reg.T @ X_reg)

        alpha = self.alpha
        beta = self.beta

        for iteration in range(max_iter):
            alpha_old = alpha
            beta_old = beta

            self.alpha = alpha
            self.beta = beta
            self.fit(X, y)

            if self.fit_intercept_prior:
                weights_for_alpha = self.posterior_mean_
            else:
                weights_for_alpha = self.posterior_mean_[1:]

            gamma = np.sum((beta * eigenvalues) / (alpha + beta * eigenvalues))

            weight_norm = weights_for_alpha.T @ weights_for_alpha
            residuals = y - X @ self.posterior_mean_
            sse = residuals.T @ residuals

            alpha = gamma / max(weight_norm, 1e-12)
            beta = (n_samples - gamma) / max(sse, 1e-12)

            alpha = max(alpha, 1e-12)
            beta = max(beta, 1e-12)

            delta_alpha = abs(alpha - alpha_old) / max(alpha_old, 1e-12)
            delta_beta = abs(beta - beta_old) / max(beta_old, 1e-12)

            if verbose:
                print(
                    f"iter={iteration:03d} "
                    f"alpha={alpha:.6f} beta={beta:.6f} "
                    f"log_evidence={self.log_evidence_:.6f}"
                )

            if delta_alpha < tol and delta_beta < tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = max_iter

        self.alpha = alpha
        self.beta = beta
        self.fit(X, y)

        return self

    def coefficient_summary(self, feature_names=None, top_k=None, exclude_intercept=True):
        """
        Return posterior coefficient summary.

        Includes:
            posterior mean
            posterior standard deviation
            95% credible interval
        """
        if self.posterior_mean_ is None or self.posterior_cov_ is None:
            raise ValueError("Model must be fitted before calling coefficient_summary().")

        n_features = len(self.posterior_mean_)

        if feature_names is None:
            feature_names = [f"x_{j}" for j in range(n_features)]

        posterior_std = np.sqrt(np.maximum(np.diag(self.posterior_cov_), 1e-12))

        rows = []
        for name, mean, std in zip(feature_names, self.posterior_mean_, posterior_std):
            if exclude_intercept and name == "intercept":
                continue

            rows.append(
                {
                    "feature": name,
                    "mean": float(mean),
                    "std": float(std),
                    "lower_95": float(mean - 1.96 * std),
                    "upper_95": float(mean + 1.96 * std),
                    "abs_mean": float(abs(mean)),
                }
            )

        rows = sorted(rows, key=lambda row: row["abs_mean"], reverse=True)

        if top_k is not None:
            rows = rows[:top_k]

        return rows