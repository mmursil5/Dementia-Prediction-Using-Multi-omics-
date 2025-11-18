import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.inspection import permutation_importance

from src.models.models import PyTorchModelWrapper


def plot_permutation_importance(model, X_test_sfs, y_test, feature_names):
    """
    Compute and plot cumulative permutation importance.
    """
    wrapper = PyTorchModelWrapper(model)

    result = permutation_importance(
        wrapper,
        X_test_sfs,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring='accuracy'
    )

    importance = np.abs(result.importances_mean)
    cumulative_importance = np.cumsum(np.sort(importance))

    sorted_idx = importance.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[sorted_idx], cumulative_importance, color='steelblue')
    plt.xlabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance using Permutation Importance (SFS-selected features)')
    plt.show()


def shap_summary_plot(model, X_train_sfs, X_test_sfs, feature_names):
    """
    SHAP KernelExplainer summary plot on a subset of test data.
    """
    shap.initjs()

    background_size = min(100, X_train_sfs.shape[0])
    background_idx = np.random.choice(X_train_sfs.shape[0], background_size, replace=False)
    background_data = X_train_sfs[background_idx]

    nsamples = min(200, X_test_sfs.shape[0])
    explain_idx = np.random.choice(X_test_sfs.shape[0], nsamples, replace=False)
    X_explain = X_test_sfs[explain_idx]

    def model_predict_proba_class1(X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs

    explainer = shap.KernelExplainer(model_predict_proba_class1, background_data)
    shap_values = explainer.shap_values(X_explain)

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=True
    )
