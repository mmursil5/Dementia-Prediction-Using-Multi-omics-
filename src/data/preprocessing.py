import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import rankdata, norm


def inverse_normal_transform(X):
    """
    Apply rank-based inverse normal transform column-wise.
    X: numpy array of shape (n_samples, n_features)
    """
    X = np.asarray(X, dtype=float)
    X_transformed = np.empty_like(X, dtype=float)
    n = X.shape[0]

    for j in range(X.shape[1]):
        col = X[:, j]
        ranks = rankdata(col, method="average")
        uniform = (ranks - 0.5) / n
        X_transformed[:, j] = norm.ppf(uniform)

    return X_transformed


def load_data(file_path, target_col="diagn", shuffle=True, random_state=44):
    data = pd.read_csv(file_path)
    if shuffle:
        data = data.sample(frac=1, random_state=random_state)
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return X, y


def preprocess_features(
    X,
    missing_threshold=0.30,
    knn_neighbors=5,
    knn_weights="distance",
):
    """
    - Filter features with > missing_threshold proportion missing
    - KNN impute remaining
    - Inverse normal transform
    Returns:
        X_transformed: np.ndarray
        feature_names: pd.Index
    """
    # 1. Filter by missingness
    missing_frac = X.isnull().mean()
    cols_to_keep = missing_frac[missing_frac <= missing_threshold].index
    X_filtered = X[cols_to_keep]
    feature_names = X_filtered.columns

    # 2. KNN imputation
    imputer = KNNImputer(n_neighbors=knn_neighbors, weights=knn_weights)
    X_imputed = imputer.fit_transform(X_filtered)

    # 3. Inverse normal transform
    X_transformed = inverse_normal_transform(X_imputed)

    return X_transformed, feature_names


def select_and_rank_features(X_transformed, y, feature_names, k=11):
    """
    Select top-k features using ANOVA F-test and order them by F-score (descending).
    Returns:
        X_selected_ranked: ndarray (n_samples, k)
        selected_features_ranked: np.ndarray of feature names
        selector: fitted SelectKBest object
        rank_order: indices of selected features sorted by F-score
    """
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_transformed, y)
    selected_features = feature_names[selector.get_support()]

    scores_all = selector.scores_
    mask = selector.get_support()
    scores_selected = scores_all[mask]
    rank_order = np.argsort(scores_selected)[::-1]

    X_selected_ranked = X_selected[:, rank_order]
    selected_features_ranked = selected_features[rank_order]

    return X_selected_ranked, selected_features_ranked, selector, rank_order
