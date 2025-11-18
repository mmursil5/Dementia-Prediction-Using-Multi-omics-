import numpy as np
from scipy.stats import rankdata, norm
from sklearn.base import BaseEstimator, TransformerMixin


class InverseNormalTransformer(BaseEstimator, TransformerMixin):
    """
    Rank-based inverse normal transform (INT) as a scikit-learn transformer.

    Applied column-wise after imputation.
    """

    def fit(self, X, y=None):
        # No learned parameters; the transform is based on ranks at transform time.
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_trans = np.empty_like(X)
        n = X.shape[0]

        for j in range(X.shape[1]):
            col = X[:, j]
            ranks = rankdata(col, method="average")
            X_trans[:, j] = norm.ppf((ranks - 0.5) / n)

        return X_trans
