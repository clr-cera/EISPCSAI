import logging
from matplotlib import pyplot as plt
import sklearn
import xgboost as xgb
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import shap
import itertools
from utils import load_features


def train_ensemble_sentiment(
    path_to_features, path_to_labels, feature_sizes=[4096, 1, 768, 768, 256, 384]
):
    numeric_params = {
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_depth": 7,
        "eta": 0.1,
    }
    multi_option_params = {
        "tree_method": "hist",
        "objective": "binary:logistic",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_depth": 7,
        "eta": 0.1,
    }

    X = np.load(path_to_features)
    dfy = pd.read_csv(path_to_labels, sep=";")

    # These options were never assigned, so for consistency in metrics, we remove them
    columns = [c for c in dfy.columns if f"Q5.5" in c or f"Q5.6" in c or f"Q5.7" in c]
    dfy = dfy.drop(columns=columns)
    data = [
        train_by_question(1, feature_sizes, numeric_params, X, dfy),
        train_by_question(2, feature_sizes, numeric_params, X, dfy),
        train_by_question(3, feature_sizes, multi_option_params, X, dfy),
        train_by_question(4, feature_sizes, multi_option_params, X, dfy),
        train_by_question(5, feature_sizes, multi_option_params, X, dfy),
    ]

    save_data(data, "results/results.csv")


def train_ensemble_sentiment_combination(
    path_to_features, path_to_labels, feature_sizes=[4096, 1, 768, 768, 256, 384]
):
    numeric_params = {
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_depth": 7,
        "eta": 0.1,
    }
    multi_option_params = {
        "tree_method": "hist",
        "objective": "binary:logistic",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_depth": 7,
        "eta": 0.1,
    }

    # Prepare features for combination
    feature_vectors = load_features(path_to_features, feature_sizes)
    feature_names = ["Age_gender", "Ita", "Objects", "NSFW", "Scene", "Thamiris Scene"]
    features = zip(feature_vectors, feature_names)

    # Iterate over combinations of features
    data = []
    for i in range(1, 6):
        combinations = itertools.combinations(features, i)
        for combo in combinations:
            logging.info(f"Training ensemble with features: {[f[1] for f in combo]}")
            X = np.concatenate([f[0] for f in combo], axis=1)

            dfy = pd.read_csv(path_to_labels, sep=";")

            # These options were never assigned, so for consistency in metrics, we remove them
            columns = [
                c for c in dfy.columns if f"Q5.5" in c or f"Q5.6" in c or f"Q5.7" in c
            ]
            dfy = dfy.drop(columns=columns)
            data.append(
                [
                    i,
                    "".join([f[1] for f in combo]),
                    train_by_question(1, None, numeric_params, X, dfy),
                    train_by_question(2, None, numeric_params, X, dfy),
                    train_by_question(3, None, multi_option_params, X, dfy),
                    train_by_question(4, None, multi_option_params, X, dfy),
                    train_by_question(5, None, multi_option_params, X, dfy),
                ]
            )

    save_data(data, "results/results_combinatorics.csv")


def train_by_question(question_number, feature_sizes, xb_parameters, X, dfy):
    # Get labels for the specific question
    columns = [c for c in dfy.columns if f"Q{question_number}" in c]
    grouped = dfy[columns].copy()
    grouped.columns = ["".join(col.split(".")[1:]) for col in columns]
    mean_by_answer = grouped.T.groupby(by=grouped.columns).mean().T
    y = mean_by_answer.values
    y = np.round(y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    logging.info(f"Labels shape: {y.shape}")
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Train shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Test shape: {X_test.shape}, {y_test.shape}")

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train the model
    bst = xgb.train(xb_parameters, dtrain, num_boost_round=128)

    # Get Predictions
    X_test = xgb.DMatrix(X_test)
    predictions = bst.predict(X_test)

    # Get metric
    metric = None
    opt_metric = []
    if xb_parameters["objective"] == "reg:squarederror":
        metric = sklearn.metrics.mean_squared_error(y_test, predictions)
        logging.info(f"Metric for Q{question_number}: {metric}")
        opt_metric.append(metric)
    else:
        predicted_classes = np.round(predictions)
        for i in range(y_test.shape[1]):
            opt_metric.append(
                balanced_accuracy_score(y_test[:, i], predicted_classes[:, i])
            )

        opt_metric = np.array(opt_metric)
        logging.info(f"Acc Q{question_number}: {opt_metric}")
        metric = opt_metric.mean()

    # If there are feature sizes, calculate SHAP values and save plots
    if feature_sizes:
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_train)
        logging.info(f"SHAP values shape: {shap_values.shape}")

        # If it is a numeric question, we expand the dimensions to generalize with multi-option questions
        shap_data = []
        if len(shap_values.shape) == 2:
            shap_values = np.expand_dims(shap_values, axis=2)

        # Now we iterate over the options
        for i in range(shap_values.shape[2]):
            logging.info(
                f"SHAP values for Q{question_number}.{i}: {shap_values[:, :, i].shape}"
            )
            # We calculate the SHAP values for each feature size and save the plots
            (
                age_gender_shap,
                ita_shap,
                objects_shap,
                nsfw_shap,
                scene_shap,
                thamiris_scene_shap,
            ) = save_shap_plot(
                shap_values[:, :, i],
                opt_metric[i],
                feature_sizes,
                f"shap_plot_q{question_number}.{i}.png",
            )
            shap_data.append(
                (
                    age_gender_shap,
                    ita_shap,
                    objects_shap,
                    nsfw_shap,
                    scene_shap,
                    thamiris_scene_shap,
                )
            )
        (
            age_gender_shap,
            ita_shap,
            objects_shap,
            nsfw_shap,
            scene_shap,
            thamiris_scene_shap,
        ) = np.mean(shap_data, axis=0)

        bst.save_model(f"models/ensemble/ensemble_sentiment_q{question_number}.json")
        return (
            metric,
            age_gender_shap,
            ita_shap,
            objects_shap,
            nsfw_shap,
            scene_shap,
            thamiris_scene_shap,
        )
    return metric


def save_confusion_matrix(y_test, predicted_classes, img_name="confusion_matrix.png"):
    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig(f"results/{img_name}")
    plt.clf()

    logging.info(f"Confusion matrix saved to results/{img_name}")


def save_shap_plot(
    shap_values,
    metric,
    feature_sizes: list[int],
    img_name="shap_plot.png",
):
    age_gender_shap = np.sum(shap_values[:, : feature_sizes[0]], axis=1)
    ita_shap = np.sum(
        shap_values[:, feature_sizes[0] : feature_sizes[0] + feature_sizes[1]], axis=1
    )
    objects_shap = np.sum(
        shap_values[
            :,
            feature_sizes[0]
            + feature_sizes[1] : feature_sizes[0]
            + feature_sizes[1]
            + feature_sizes[2],
        ],
        axis=1,
    )
    nsfw_shap = np.sum(
        shap_values[
            :,
            feature_sizes[0]
            + feature_sizes[1]
            + feature_sizes[2] : feature_sizes[0]
            + feature_sizes[1]
            + feature_sizes[2]
            + feature_sizes[3],
        ],
        axis=1,
    )
    scene_shap = np.sum(
        shap_values[
            :,
            feature_sizes[0]
            + feature_sizes[1]
            + feature_sizes[2]
            + feature_sizes[3] : feature_sizes[0]
            + feature_sizes[1]
            + feature_sizes[2]
            + feature_sizes[3]
            + feature_sizes[4],
        ],
        axis=1,
    )
    thamiris_scene_shap = np.sum(
        shap_values[
            :,
            feature_sizes[0]
            + feature_sizes[1]
            + feature_sizes[2]
            + feature_sizes[3]
            + feature_sizes[4] :,
        ],
        axis=1,
    )

    age_gender_shap_mean = np.abs(np.mean(age_gender_shap))
    ita_shap_mean = np.abs(np.mean(ita_shap))
    objects_shap_mean = np.abs(np.mean(objects_shap))
    nsfw_shap_mean = np.abs(np.mean(nsfw_shap))
    scene_shap_mean = np.abs(np.mean(scene_shap))
    thamiris_scene_shap_mean = np.abs(np.mean(thamiris_scene_shap))

    logging.info(f"Age Gender importance: {age_gender_shap_mean}")
    logging.info(f"Ita importance: {ita_shap_mean}")
    logging.info(f"Objects importance: {objects_shap_mean}")
    logging.info(f"NSFW importance: {nsfw_shap_mean}")
    logging.info(f"Scene importance: {scene_shap_mean}")
    logging.info(f"Thamiris Scene importance: {thamiris_scene_shap_mean}")
    plt.bar(
        ["Age Gender", "Ita", "Objects", "NSFW", "Scene", "Thamiris Scene"],
        [
            age_gender_shap_mean,
            ita_shap_mean,
            objects_shap_mean,
            nsfw_shap_mean,
            scene_shap_mean,
            thamiris_scene_shap_mean,
        ],
    )
    plt.title(f"SHAP Feature Importance, metric= {metric:.2f}")
    plt.savefig(f"results/{img_name}", dpi=300, bbox_inches="tight")
    plt.clf()
    logging.info(f"SHAP plot saved to results/{img_name}")
    return (
        age_gender_shap_mean,
        ita_shap_mean,
        objects_shap_mean,
        nsfw_shap_mean,
        scene_shap_mean,
        thamiris_scene_shap_mean,
    )


def save_data(data, filename):
    df = pd.DataFrame(
        data,
        columns=[
            "Metric",
            "Age Gender",
            "Ita",
            "Objects",
            "NSFW",
            "Scene",
            "Thamiris Scene",
        ],
    )
    df.to_csv(filename, index=True)
    logging.info(f"Data saved to {filename}")


def load_ensemble_model(path_to_model):
    bst = xgb.Booster()
    bst.load_model(path_to_model)
    return bst
