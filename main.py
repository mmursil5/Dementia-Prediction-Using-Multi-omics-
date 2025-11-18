from sklearn.model_selection import train_test_split

from src.data.preprocessing import load_data, preprocess_features, select_and_rank_features
from src.training.feature_selection import sequential_forward_selection_ranked
from src.training.training import train_model_with_smote, evaluate_model
from src.analysis.explainability import plot_permutation_importance, shap_summary_plot


def main():
    file_path = 'data/AD_metabolites_covariants_merged2.csv'
    target_col = 'diagn'

    # 1. Load data
    X, y = load_data(file_path, target_col=target_col)

    # 2. Preprocess features (missingness filter, KNN, inverse normal)
    X_transformed, feature_names = preprocess_features(X)

    # 3. SelectKBest + rank by F-statistic
    X_selected_ranked, selected_features_ranked, selector, rank_order = select_and_rank_features(
        X_transformed, y, feature_names, k=11
    )

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected_ranked, y, test_size=0.30, random_state=42, stratify=y
    )

    # 5. SFS with 5-fold CV + RandomUnderSampler + AUC (patience=3)
    best_feature_indices, best_cv_auc, auc_curve = sequential_forward_selection_ranked(
        X_train, y_train, random_state=44, patience=3
    )

    X_train_sfs = X_train[:, best_feature_indices]
    X_test_sfs = X_test[:, best_feature_indices]
    selected_features_sfs = selected_features_ranked[best_feature_indices]
    m = len(best_feature_indices)

    print(f"\nFinal chosen subset size m = {m}")
    print("Selected features (after SFS):")
    for f in selected_features_sfs:
        print("  -", f)

    # 6. Train final DNN with SMOTE
    model = train_model_with_smote(X_train_sfs, y_train)

    # 7. Evaluate on test set
    metrics, y_prob_pos, y_pred = evaluate_model(model, X_test_sfs, y_test)

    print("\nModel Performance (final model with SFS-selected features):")
    for k, v in metrics.items():
        if k == "Confusion Matrix":
            print(f"{k}:\n{v}")
        else:
            print(f"{k}: {v:.4f}")

    # 8. Permutation importance
    plot_permutation_importance(model, X_test_sfs, y_test, selected_features_sfs)

    # 9. SHAP summary plot
    shap_summary_plot(model, X_train_sfs, X_test_sfs, selected_features_sfs)


if __name__ == "__main__":
    main()
