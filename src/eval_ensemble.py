from eisp import proxy_tasks, ensemble, visualization
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import os


def eval_ensemble():
    feature_vectors: proxy_tasks.FeatureVectors = proxy_tasks.FeatureVectors.from_files(
        "./features"
    )
    labels = (pd.read_csv("rcpd/rcpd_annotation_fix.csv")["csam"] * 1).to_numpy()

    print(
        f"Feature vectors loaded with features: {list(feature_vectors.get_all_features().keys())}"
    )
    print(f"Labels loaded with shape: {labels.shape}")
    for feature_name, feature_vector in feature_vectors.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

    train_features, test_features, train_indices, test_indices = (
        feature_vectors.train_test_split(test_size=0.2, random_state=42)
    )

    train_labels: np.ndarray = labels[train_indices]
    test_labels = labels[test_indices]

    for feature_name, feature_vector in train_features.get_all_features().items():
        print(f"Train Feature: {feature_name}, Shape: {feature_vector.shape}")
    print(f"Train Labels shape: {train_labels.shape}")

    params = {
        "objective": "binary:logistic",
        "seed": 42,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train ensemble with default features
    ensemble_model = ensemble.Ensemble(
        feature_vectors=train_features, labels=train_labels, debug=True
    )
    ensemble_model.train(
        model_type="xgboost",
        optimization_trials=5,
        optimization_direction="maximize",
        metric_function=lambda y_true, y_pred: balanced_accuracy_score(
            y_true, np.round(y_pred)
        ),
        should_extract_shap=True,
        hyperparams=params,
    )

    print("Ensemble training on RCPD completed successfully.")
    print(f"Val metric: {ensemble_model.val_metric}")

    shap_values = ensemble_model.shap
    shap_aggregated = ensemble_model.shap_aggregated
    print({k: v.shape for k, v in shap_values.items()})
    print({k: v for k, v in shap_aggregated.items()})
    if not os.path.exists("results/"):
        os.makedirs("results/")

    if not os.path.exists("results/ensemble/"):
        os.makedirs("results/ensemble/")

    # Save on disk train and test indices
    np.save("./results/train_indices.npy", train_indices)
    np.save("./results/test_indices.npy", test_indices)

    # Plot feature importance
    feature_importance_save_path = "./results/ensemble/feature_importance.png"
    visualization.plot_feature_importance(
        shap_aggregated,
        save_path=feature_importance_save_path,
    )
    print(f"Feature importance plot saved to {feature_importance_save_path}")

    # save true and predicted labels
    preds = ensemble_model.pred_labels
    true = ensemble_model.true_labels
    np.save("./results/ensemble/pred_labels.npy", preds)
    np.save("./results/ensemble/true_labels.npy", true)

    confusion_matrix_save_path = "./results/ensemble/confusion_matrix.png"
    visualization.plot_confusion_matrix(
        true,
        np.round(preds),
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )
    print(f"Confusion matrix plot saved to {confusion_matrix_save_path}")

    # save shap aggregated values
    np.save("./results/ensemble/shap_aggregated.npy", shap_aggregated)
    # save val metric
    with open("./results/ensemble/val_metric.txt", "w") as f:
        f.write(str(ensemble_model.val_metric))

    features = list(test_features.get_all_features().values())
    x_test = np.concatenate(features, axis=1)
    test_metric, test_preds = ensemble_model.test_xgboost(x_test, test_labels, metric_function=lambda y_true, y_pred: balanced_accuracy_score(y_true, np.round(y_pred)))
    print(f"Test metric: {test_metric}")
    print("Plotting test confusion matrix...")
    confusion_matrix_save_path = "./results/ensemble/test_confusion_matrix.png"
    visualization.plot_confusion_matrix(
        test_labels,
        np.round(test_preds),
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )
    print(f"Test confusion matrix plot saved to {confusion_matrix_save_path}")

    # saving test preds and test metric
    np.save("./results/ensemble/test_pred_labels.npy", test_preds)
    with open("./results/ensemble/test_metric.txt", "w") as f:
        f.write(str(test_metric))


def eval_ensemble_pca():
    feature_vectors: proxy_tasks.FeatureVectors = proxy_tasks.FeatureVectors.from_files(
        "./features"
    )
    labels = (pd.read_csv("rcpd/rcpd_annotation_fix.csv")["csam"] * 1).to_numpy()
    feature_vectors = feature_vectors.apply_pca()

    print(
        f"Feature vectors loaded with features: {list(feature_vectors.get_all_features().keys())}"
    )
    print(f"Labels loaded with shape: {labels.shape}")
    for feature_name, feature_vector in feature_vectors.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

    train_features, test_features, train_indices, test_indices = (
        feature_vectors.train_test_split(test_size=0.2, random_state=42)
    )

    train_labels: np.ndarray = labels[train_indices]
    test_labels = labels[test_indices]

    for feature_name, feature_vector in train_features.get_all_features().items():
        print(f"Train Feature: {feature_name}, Shape: {feature_vector.shape}")
    print(f"Train Labels shape: {train_labels.shape}")

    params = {
        "objective": "binary:logistic",
        "seed": 42,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train ensemble with default features
    ensemble_model = ensemble.Ensemble(
        feature_vectors=train_features, labels=train_labels, debug=True
    )
    ensemble_model.train(
        model_type="xgboost",
        optimization_trials=5,
        optimization_direction="maximize",
        metric_function=lambda y_true, y_pred: balanced_accuracy_score(
            y_true, np.round(y_pred)
        ),
        should_extract_shap=True,
        hyperparams=params,
    )

    print("Ensemble training on RCPD completed successfully.")
    print(f"Val metric: {ensemble_model.val_metric}")

    shap_values = ensemble_model.shap
    shap_aggregated = ensemble_model.shap_aggregated
    print({k: v.shape for k, v in shap_values.items()})
    print({k: v for k, v in shap_aggregated.items()})
    if not os.path.exists("results/"):
        os.makedirs("results/")

    if not os.path.exists("results/ensemble_pca/"):
        os.makedirs("results/ensemble_pca/")

    # Plot feature importance
    feature_importance_save_path = "./results/ensemble_pca/feature_importance.png"
    visualization.plot_feature_importance(
        shap_aggregated,
        save_path=feature_importance_save_path,
    )
    print(f"Feature importance plot saved to {feature_importance_save_path}")

    # save true and predicted labels
    preds = ensemble_model.pred_labels
    true = ensemble_model.true_labels
    np.save("./results/ensemble_pca/pred_labels.npy", preds)
    np.save("./results/ensemble_pca/true_labels.npy", true)

    confusion_matrix_save_path = "./results/ensemble_pca/confusion_matrix.png"
    visualization.plot_confusion_matrix(
        true,
        np.round(preds),
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )
    print(f"Confusion matrix plot saved to {confusion_matrix_save_path}")

    # save shap aggregated values
    np.save("./results/ensemble_pca/shap_aggregated.npy", shap_aggregated)
    # save val metric
    with open("./results/ensemble_pca/val_metric.txt", "w") as f:
        f.write(str(ensemble_model.val_metric))

    features = list(test_features.get_all_features().values())
    x_test = np.concatenate(features, axis=1)
    test_metric, test_preds = ensemble_model.test_xgboost(x_test, test_labels, metric_function=lambda y_true, y_pred: balanced_accuracy_score(y_true, np.round(y_pred)))
    print(f"Test metric: {test_metric}")
    print("Plotting test confusion matrix...")
    confusion_matrix_save_path = "./results/ensemble_pca/test_confusion_matrix.png"
    visualization.plot_confusion_matrix(
        test_labels,
        np.round(test_preds),
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )
    print(f"Test confusion matrix plot saved to {confusion_matrix_save_path}")

    # saving test preds and test metric
    np.save("./results/ensemble_pca/test_pred_labels.npy", test_preds)
    with open("./results/ensemble_pca/test_metric.txt", "w") as f:
        f.write(str(test_metric))


def eval_ensemble_kfold():
    feature_vectors: proxy_tasks.FeatureVectors = proxy_tasks.FeatureVectors.from_files(
        "./features"
    )
    labels = (pd.read_csv("rcpd/rcpd_annotation_fix.csv")["csam"] * 1).to_numpy()
    print(
        f"Feature vectors loaded with features: {list(feature_vectors.get_all_features().keys())}"
    )
    print(f"Labels loaded with shape: {labels.shape}")
    for feature_name, feature_vector in feature_vectors.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

    train_features, _test_features, train_indices, test_indices = (
        feature_vectors.train_test_split(test_size=0.2, random_state=42)
    )

    train_labels: np.ndarray = labels[train_indices]
    _test_labels = labels[test_indices]

    for feature_name, feature_vector in train_features.get_all_features().items():
        print(f"Train Feature: {feature_name}, Shape: {feature_vector.shape}")
    print(f"Train Labels shape: {train_labels.shape}")

    params = {
        "objective": "binary:logistic",
        "seed": 42,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train ensemble with default features and eval with k-fold cross validation
    ensemble_model = ensemble.EnsembleKFold(
        feature_vectors=train_features, labels=train_labels, debug=True
    )
    ensemble_model.hyperparams = params
    ensemble_model.train_k_fold(
        k=5,
        model_type="xgboost",
        optimization_trials=5,
        optimization_direction="maximize",
        metric_function=lambda y_true, y_pred: balanced_accuracy_score(
            y_true, np.round(y_pred)
        ),
        should_extract_shap=True,
    )

    print("Ensemble k-fold training on RCPD completed successfully.")
    print(f"Val metrics per fold: {ensemble_model.val_metric_k_fold}")
    print(f"Average Val metric: {ensemble_model.average_val_metric}")

    if not os.path.exists("results/"):
        os.makedirs("results/")

    if not os.path.exists("results/ensemble_kfold/"):
        os.makedirs("results/ensemble_kfold/")

    mean_shap_values_over_folds = {}
    for feature in ensemble_model.shap_aggregated_k_fold.keys():
        mean_shap_values_over_folds[feature] = np.mean(
            ensemble_model.shap_aggregated_k_fold[feature]
        )
    print("Mean SHAP values over folds:", mean_shap_values_over_folds)

    # save mean shap aggregated values over folds
    np.save(
        "./results/ensemble_kfold/mean_shap_aggregated.npy", mean_shap_values_over_folds
    )
    # save shap aggregated values per fold, one file per fold
    for fold_idx, shap_aggregated in enumerate(
        ensemble_model.shap_aggregated_k_fold.values()
    ):
        np.save(
            f"./results/ensemble_kfold/shap_aggregated_fold_{fold_idx}.npy",
            np.array(shap_aggregated),
        )

    # save average val metric
    with open("./results/ensemble_kfold/average_val_metric.txt", "w") as f:
        f.write(str(ensemble_model.average_val_metric))

    # save preds and true labels per fold
    for fold_idx, (preds, true) in enumerate(
        zip(ensemble_model.pred_labels_k_fold, ensemble_model.true_labels_k_fold)
    ):
        np.save(f"./results/ensemble_kfold/pred_labels_fold_{fold_idx}.npy", preds)
        np.save(f"./results/ensemble_kfold/true_labels_fold_{fold_idx}.npy", true)

    # plot feature importance using mean shap values over folds
    feature_importance_save_path = "./results/ensemble_kfold/feature_importance.png"
    visualization.plot_feature_importance(
        mean_shap_values_over_folds,
        save_path=feature_importance_save_path,
    )
    print(f"Feature importance plot saved to {feature_importance_save_path}")

    # plot confusion matrix over all folds
    # generate pred_labels and true_labels by concatenating predictions from each fold
    all_pred_labels = []
    for fold_pred in ensemble_model.pred_labels_k_fold:
        all_pred_labels.append(np.round(fold_pred))
    all_pred_labels = np.concatenate(all_pred_labels, axis=0)

    all_true_labels = np.concatenate(ensemble_model.true_labels_k_fold, axis=0)

    confusion_matrix_save_path = "./results/ensemble_kfold/confusion_matrix.png"
    visualization.plot_confusion_matrix(
        all_true_labels,
        all_pred_labels,
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )
    print(f"Confusion matrix plot saved to {confusion_matrix_save_path}")

    # save val metrics per fold
    with open("./results/ensemble_kfold/val_metrics_k_fold.txt", "w") as f:
        for metric in ensemble_model.val_metric_k_fold:
            f.write(str(metric) + "\n")


def eval_ensemble_kfold_pca():
    feature_vectors: proxy_tasks.FeatureVectors = proxy_tasks.FeatureVectors.from_files(
        "./features"
    )
    labels = (pd.read_csv("rcpd/rcpd_annotation_fix.csv")["csam"] * 1).to_numpy()
    feature_vectors = feature_vectors.apply_pca()

    print(
        f"Feature vectors loaded with features: {list(feature_vectors.get_all_features().keys())}"
    )
    print(f"Labels loaded with shape: {labels.shape}")
    for feature_name, feature_vector in feature_vectors.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

    train_features, _test_features, train_indices, test_indices = (
        feature_vectors.train_test_split(test_size=0.2, random_state=42)
    )

    train_labels: np.ndarray = labels[train_indices]
    _test_labels = labels[test_indices]

    for feature_name, feature_vector in train_features.get_all_features().items():
        print(f"Train Feature: {feature_name}, Shape: {feature_vector.shape}")
    print(f"Train Labels shape: {train_labels.shape}")

    params = {
        "objective": "binary:logistic",
        "seed": 42,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train ensemble with default features and eval with k-fold cross validation
    ensemble_model = ensemble.EnsembleKFold(
        feature_vectors=train_features, labels=train_labels, debug=True
    )
    ensemble_model.hyperparams = params
    ensemble_model.train_k_fold(
        k=5,
        model_type="xgboost",
        optimization_trials=5,
        optimization_direction="maximize",
        metric_function=lambda y_true, y_pred: balanced_accuracy_score(
            y_true, np.round(y_pred)
        ),
        should_extract_shap=True,
    )

    print("Ensemble k-fold training on RCPD completed successfully.")
    print(f"Val metrics per fold: {ensemble_model.val_metric_k_fold}")
    print(f"Average Val metric: {ensemble_model.average_val_metric}")

    if not os.path.exists("results/"):
        os.makedirs("results/")

    if not os.path.exists("results/ensemble_kfold_pca/"):
        os.makedirs("results/ensemble_kfold_pca/")

    mean_shap_values_over_folds = {}
    for feature in ensemble_model.shap_aggregated_k_fold.keys():
        mean_shap_values_over_folds[feature] = np.mean(
            ensemble_model.shap_aggregated_k_fold[feature]
        )
    print("Mean SHAP values over folds:", mean_shap_values_over_folds)

    # save mean shap aggregated values over folds
    np.save(
        "./results/ensemble_kfold_pca/mean_shap_aggregated.npy",
        mean_shap_values_over_folds,
    )
    # save shap aggregated values per fold, one file per fold
    for fold_idx, shap_aggregated in enumerate(
        ensemble_model.shap_aggregated_k_fold.values()
    ):
        np.save(
            f"./results/ensemble_kfold_pca/shap_aggregated_fold_{fold_idx}.npy",
            np.array(shap_aggregated),
        )

    # save average val metric
    with open("./results/ensemble_kfold_pca/average_val_metric.txt", "w") as f:
        f.write(str(ensemble_model.average_val_metric))

    # save preds and true labels per fold
    for fold_idx, (preds, true) in enumerate(
        zip(ensemble_model.pred_labels_k_fold, ensemble_model.true_labels_k_fold)
    ):
        np.save(f"./results/ensemble_kfold_pca/pred_labels_fold_{fold_idx}.npy", preds)
        np.save(f"./results/ensemble_kfold_pca/true_labels_fold_{fold_idx}.npy", true)

    # plot feature importance using mean shap values over folds
    feature_importance_save_path = "./results/ensemble_kfold_pca/feature_importance.png"
    visualization.plot_feature_importance(
        mean_shap_values_over_folds,
        save_path=feature_importance_save_path,
    )
    print(f"Feature importance plot saved to {feature_importance_save_path}")

    # plot confusion matrix over all folds
    # generate pred_labels and true_labels by concatenating predictions from each fold
    all_pred_labels = []
    for fold_pred in ensemble_model.pred_labels_k_fold:
        all_pred_labels.append(np.round(fold_pred))
    all_pred_labels = np.concatenate(all_pred_labels, axis=0)

    all_true_labels = np.concatenate(ensemble_model.true_labels_k_fold, axis=0)

    confusion_matrix_save_path = "./results/ensemble_kfold_pca/confusion_matrix.png"
    visualization.plot_confusion_matrix(
        all_true_labels,
        all_pred_labels,
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )
    print(f"Confusion matrix plot saved to {confusion_matrix_save_path}")

    # save val metrics per fold
    with open("./results/ensemble_kfold_pca/val_metrics_k_fold.txt", "w") as f:
        for metric in ensemble_model.val_metric_k_fold:
            f.write(str(metric) + "\n")


def eval_ensemble_combinatorics():
    feature_vectors: proxy_tasks.FeatureVectors = proxy_tasks.FeatureVectors.from_files(
        "./features"
    )
    labels = (pd.read_csv("rcpd/rcpd_annotation_fix.csv")["csam"] * 1).to_numpy()

    print(
        f"Feature vectors loaded with features: {list(feature_vectors.get_all_features().keys())}"
    )
    print(f"Labels loaded with shape: {labels.shape}")
    for feature_name, feature_vector in feature_vectors.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

    train_features, test_features, train_indices, test_indices = (
        feature_vectors.train_test_split(test_size=0.2, random_state=42)
    )

    train_labels: np.ndarray = labels[train_indices]
    test_labels = labels[test_indices]

    for feature_name, feature_vector in train_features.get_all_features().items():
        print(f"Train Feature: {feature_name}, Shape: {feature_vector.shape}")
    print(f"Train Labels shape: {train_labels.shape}")

    params = {
        "objective": "binary:logistic",
        "seed": 42,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train ensemble with default features
    ensemble_model = ensemble.Ensemble(
        feature_vectors=train_features, labels=train_labels, debug=True
    )
    ensemble_model.train(
        model_type="xgboost",
        optimization_trials=5,
        optimization_direction="maximize",
        metric_function=lambda y_true, y_pred: balanced_accuracy_score(
            y_true, np.round(y_pred)
        ),
        should_extract_shap=True,
        hyperparams=params,
    )
    print("Ensemble training on RCPD completed successfully.")

    ensemble_combinatorics: ensemble.EnsembleCombinatorics = (
        ensemble.EnsembleCombinatorics.from_ensemble(ensemble_model)
    )
    ensemble_combinatorics.train_combinatorics(
        model_type="xgboost",
        metric_function=lambda y_true, y_pred: balanced_accuracy_score(
            y_true, np.round(y_pred)
        ),
    )

    print("Ensemble combinatorics training on RCPD completed successfully.")
    print(f"Best Combinatorics Val metric: {ensemble_combinatorics.best_val_metric}")
    print(
        f"Best feature combination: {ensemble_combinatorics.best_feature_combination}"
    )

    if not os.path.exists("results/"):
        os.makedirs("results/")
    if not os.path.exists("results/ensemble_combinatorics/"):
        os.makedirs("results/ensemble_combinatorics/")

    print("Plotting confusion matrix for best combinatorics model...")
    confusion_matrix_save_path = "./results/ensemble_combinatorics/confusion_matrix.png"
    visualization.plot_confusion_matrix(
        true_labels=ensemble_combinatorics.best_true_labels,
        pred_labels=np.round(ensemble_combinatorics.best_pred_labels),
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )

    print(f"Confusion matrix plot saved to {confusion_matrix_save_path}")

    print("Saving combinatorics training data to disk...")

    data_save_path = "./results/ensemble_combinatorics/ensemble_combinatorics_data.csv"

    ensemble_combinatorics.save_training_data_to_disk(data_save_path)

    # save preds and true labels
    np.save(
        "./results/ensemble_combinatorics/pred_labels.npy",
        ensemble_combinatorics.best_pred_labels,
    )
    np.save(
        "./results/ensemble_combinatorics/true_labels.npy",
        ensemble_combinatorics.best_true_labels,
    )

    features = list(test_features.get_all_features()[name] for name in ensemble_combinatorics.best_feature_combination)
    x_test = np.concatenate(features, axis=1)
    test_metric, test_preds = ensemble_combinatorics.test_xgboost(x_test, test_labels, metric_function=lambda y_true, y_pred: balanced_accuracy_score(y_true, np.round(y_pred)))
    print(f"Test metric: {test_metric}")
    print("Plotting test confusion matrix...")
    confusion_matrix_save_path = "./results/ensemble_combinatorics/test_confusion_matrix.png"
    visualization.plot_confusion_matrix(
        test_labels,
        np.round(test_preds),
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )
    print(f"Test confusion matrix plot saved to {confusion_matrix_save_path}")

    # saving test preds and test metric
    np.save("./results/ensemble_combinatorics/test_pred_labels.npy", test_preds)
    with open("./results/ensemble_combinatorics/test_metric.txt", "w") as f:
        f.write(str(test_metric))


def eval_ensemble_combinatorics_pca():
    feature_vectors: proxy_tasks.FeatureVectors = proxy_tasks.FeatureVectors.from_files(
        "./features"
    )
    feature_vectors = feature_vectors.apply_pca()
    labels = (pd.read_csv("rcpd/rcpd_annotation_fix.csv")["csam"] * 1).to_numpy()

    print(
        f"Feature vectors loaded with features: {list(feature_vectors.get_all_features().keys())}"
    )
    print(f"Labels loaded with shape: {labels.shape}")
    for feature_name, feature_vector in feature_vectors.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

    train_features, test_features, train_indices, test_indices = (
        feature_vectors.train_test_split(test_size=0.2, random_state=42)
    )

    train_labels: np.ndarray = labels[train_indices]
    test_labels = labels[test_indices]

    for feature_name, feature_vector in train_features.get_all_features().items():
        print(f"Train Feature: {feature_name}, Shape: {feature_vector.shape}")
    print(f"Train Labels shape: {train_labels.shape}")

    params = {
        "objective": "binary:logistic",
        "seed": 42,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train ensemble with default features
    ensemble_model = ensemble.Ensemble(
        feature_vectors=train_features, labels=train_labels, debug=True
    )
    ensemble_model.train(
        model_type="xgboost",
        optimization_trials=5,
        optimization_direction="maximize",
        metric_function=lambda y_true, y_pred: balanced_accuracy_score(
            y_true, np.round(y_pred)
        ),
        should_extract_shap=True,
        hyperparams=params,
    )
    print("Ensemble training on RCPD completed successfully.")

    ensemble_combinatorics: ensemble.EnsembleCombinatorics = (
        ensemble.EnsembleCombinatorics.from_ensemble(ensemble_model)
    )
    ensemble_combinatorics.train_combinatorics(
        model_type="xgboost",
        metric_function=lambda y_true, y_pred: balanced_accuracy_score(
            y_true, np.round(y_pred)
        ),
    )

    print("Ensemble combinatorics training on RCPD completed successfully.")
    print(f"Best Combinatorics Val metric: {ensemble_combinatorics.best_val_metric}")
    print(
        f"Best feature combination: {ensemble_combinatorics.best_feature_combination}"
    )

    if not os.path.exists("results/"):
        os.makedirs("results/")
    if not os.path.exists("results/ensemble_combinatorics_pca/"):
        os.makedirs("results/ensemble_combinatorics_pca/")

    print("Plotting confusion matrix for best combinatorics model...")
    confusion_matrix_save_path = (
        "./results/ensemble_combinatorics_pca/confusion_matrix.png"
    )
    visualization.plot_confusion_matrix(
        true_labels=ensemble_combinatorics.best_true_labels,
        pred_labels=np.round(ensemble_combinatorics.best_pred_labels),
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )

    print(f"Confusion matrix plot saved to {confusion_matrix_save_path}")

    print("Saving combinatorics training data to disk...")

    data_save_path = (
        "./results/ensemble_combinatorics_pca/ensemble_combinatorics_data.csv"
    )

    ensemble_combinatorics.save_training_data_to_disk(data_save_path)

    # save preds and true labels
    np.save(
        "./results/ensemble_combinatorics_pca/pred_labels.npy",
        ensemble_combinatorics.best_pred_labels,
    )
    np.save(
        "./results/ensemble_combinatorics_pca/true_labels.npy",
        ensemble_combinatorics.best_true_labels,
    )

    features = list(test_features.get_all_features()[name] for name in ensemble_combinatorics.best_feature_combination)
    x_test = np.concatenate(features, axis=1)
    test_metric, test_preds = ensemble_combinatorics.test_xgboost(x_test, test_labels, metric_function=lambda y_true, y_pred: balanced_accuracy_score(y_true, np.round(y_pred)))
    print(f"Test metric: {test_metric}")
    print("Plotting test confusion matrix...")
    confusion_matrix_save_path = "./results/ensemble_combinatorics_pca/test_confusion_matrix.png"
    visualization.plot_confusion_matrix(
        test_labels,
        np.round(test_preds),
        class_names=["Non-CSAM", "CSAM"],
        save_path=confusion_matrix_save_path,
    )
    print(f"Test confusion matrix plot saved to {confusion_matrix_save_path}")

    # saving test preds and test metric
    np.save("./results/ensemble_combinatorics_pca/test_pred_labels.npy", test_preds)
    with open("./results/ensemble_combinatorics_pca/test_metric.txt", "w") as f:
        f.write(str(test_metric))
