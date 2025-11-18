import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

from src.models.models import RegularizedDNN


def evaluate_feature_subset(X, y, feature_indices, n_splits=5, n_epochs=40, random_state=44):
    """
    Evaluate subset of features using 5-fold CV, RandomUnderSampler, and AUC.
    X: numpy array (training set)
    y: array-like (training labels)
    feature_indices: list of feature indices to include
    """
    X_sub = X[:, feature_indices]
    y_array = np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sub, y_array), 1):
        X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
        y_tr, y_val = y_array[train_idx], y_array[val_idx]

        rus = RandomUnderSampler(random_state=random_state)
        X_tr_res, y_tr_res = rus.fit_resample(X_tr, y_tr)

        X_tr_tensor = torch.tensor(X_tr_res, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_tr_res, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        model_cv = RegularizedDNN(input_dim=len(feature_indices), num_classes=2)
        criterion_cv = nn.CrossEntropyLoss()
        optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=0.0005, weight_decay=1e-5)

        for epoch in range(n_epochs):
            model_cv.train()
            optimizer_cv.zero_grad()
            outputs = model_cv(X_tr_tensor)
            loss = criterion_cv(outputs, y_tr_tensor)
            loss.backward()
            optimizer_cv.step()

        model_cv.eval()
        with torch.no_grad():
            logits_val = model_cv(X_val_tensor)
            probs_val = torch.softmax(logits_val, dim=1)[:, 1].cpu().numpy()

        auc_fold = roc_auc_score(y_val_tensor.numpy(), probs_val)
        aucs.append(auc_fold)
        print(f"  Fold {fold}, subset {feature_indices}, AUC={auc_fold:.4f}")

    mean_auc = np.mean(aucs)
    print(f"Mean CV AUC for subset {feature_indices} = {mean_auc:.4f}")
    return mean_auc


def sequential_forward_selection_ranked(X, y, random_state=44, n_splits=5, patience=3):
    """
    X: training data with columns already ordered by univariate ranking (highest F-score first)
    y: training labels
    Features are added in ranked order; feature addition stops when the mean CV AUC
    shows no further improvement after adding `patience` more features.
    Returns:
        best_subset_idx: list of indices [0, ..., m-1] corresponding to the best prefix length m
        best_auc: mean CV AUC for that subset
        auc_per_size: list of mean AUCs for subset sizes 1, 2, ..., until stopping
    """
    n_features = X.shape[1]
    best_auc = -np.inf
    best_subset_size = 0
    auc_per_size = []
    no_improvement_count = 0

    for m in range(1, n_features + 1):
        subset_idx = list(range(m))  # first m ranked features
        print(f"\nEvaluating first {m} ranked features: indices {subset_idx}")
        mean_auc = evaluate_feature_subset(
            X, y, subset_idx, n_splits=n_splits, random_state=random_state
        )
        auc_per_size.append(mean_auc)

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_subset_size = m
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(
                f"No improvement in mean CV AUC (current={mean_auc:.4f}, "
                f"best={best_auc:.4f}); no_improvement_count={no_improvement_count}"
            )
            if no_improvement_count >= patience:
                print(
                    f"No further improvement in mean CV AUC after adding {patience} more "
                    f"features, stopping SFS."
                )
                break

    best_subset_idx = list(range(best_subset_size))
    print(f"\nBest subset size m = {best_subset_size}, CV AUC = {best_auc:.4f}")
    print(f"Best subset indices (wrt ranked SelectKBest output): {best_subset_idx}")
    return best_subset_idx, best_auc, auc_per_size
