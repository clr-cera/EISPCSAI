import logging
from matplotlib import pyplot as plt
import sklearn
import xgboost as xgb
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import shap


def train_ensemble_sentiment(path_to_features, path_to_labels):
    data = [
        train_numeric_question(path_to_features, path_to_labels, 1),
        train_numeric_question(path_to_features, path_to_labels, 2),
        train_multi_option_question(path_to_features, path_to_labels, 3),
        train_multi_option_question(path_to_features, path_to_labels, 4),
        train_multi_option_question(path_to_features, path_to_labels, 5),
    ]

    save_data(data, "results/results.csv")


def train_numeric_question(path_to_features, path_to_labels, question_number):
    # Load features and labels
    X = np.load(path_to_features)

    # Average all Q1 labels
    dfy = pd.read_csv(path_to_labels, sep=";")
    col_idxs = [(question_number+1) + 26 * i for i in range(5)]
    y = dfy.iloc[:, col_idxs].mean(axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    logging.info(f"Labels shape: {y.shape}")
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Train shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Test shape: {X_test.shape}, {y_test.shape}")

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Set parameters for XGBoost
    params = {
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbosity": 3,
        "max_depth": 7,
        "eta": 0.1,
    }

    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=128)

    X_test = xgb.DMatrix(X_test)
    predictions = bst.predict(X_test)

    mse = sklearn.metrics.mean_squared_error(y_test, predictions)
    logging.info(f"Mse: {mse}")

    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_train)
    age_gender_shap, ita_shap, objects_shap, nsfw_shap, scene_shap, thamiris_scene_shap = save_shap_plot(
        shap_values, mse, f"shap_plot_q{question_number}_reg.png"
    )

    bst.save_model(f"models/ensemble/ensemble_sentiment_q{question_number}.json")
    return mse, age_gender_shap, ita_shap, objects_shap, nsfw_shap, scene_shap, thamiris_scene_shap

def train_multi_option_question(path_to_features, path_to_labels, question_number):
    X = np.load(path_to_features)

    dfy = pd.read_csv(path_to_labels, sep=";")
    columns = [c for c in dfy.columns if f'.Q{question_number}.' in c]
    grouped = dfy[columns].copy()
    grouped.columns = ["".join(col.split('.')[1:]) for col in columns]
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

    # Set parameters for XGBoost
    params = {
        "tree_method": "hist",
        "objective": "binary:logistic",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbosity": 3,
        "max_depth": 7,
        "eta": 0.1,
    }

    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=128)

    # Uncomment the following lines to load the model from a file

    X_test = xgb.DMatrix(X_test)
    predictions = bst.predict(X_test)
    predicted_classes = np.round(predictions)

    acc = []

    for i in range(y_test.shape[1]):
        acc.append(accuracy_score(y_test[:, i], predicted_classes[:, i]))

    acc = np.array(acc)
    logging.info(f"Acc Q{question_number}: {acc}")

    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_train)
    logging.info(f"SHAP values shape: {shap_values.shape}")

    shap_data = []
    for i in range(shap_values.shape[2]):
        logging.info(f"SHAP values for Q{question_number}.{i}: {shap_values[:, :, i].shape}")
        age_gender_shap, ita_shap, objects_shap, nsfw_shap, scene_shap = save_shap_plot(
            shap_values[:, :, i], acc[i], f"shap_plot_q{question_number}.{i}.png"
        )
        shap_data.append((age_gender_shap, ita_shap, objects_shap, nsfw_shap, scene_shap))

    age_gender_shap, ita_shap, objects_shap, nsfw_shap, scene_shap, thamiris_scene_shap = np.mean(shap_data, axis=0)
    bst.save_model(f"models/ensemble/ensemble_sentiment_q{question_number}.json")
    return acc.mean(), age_gender_shap, ita_shap, objects_shap, nsfw_shap, scene_shap, thamiris_scene_shap


def save_confusion_matrix(y_test, predicted_classes, img_name="confusion_matrix.png"):
    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig(f"results/{img_name}")
    plt.clf()

    logging.info(f"Confusion matrix saved to results/{img_name}")


def save_shap_plot(shap_values, metric, img_name="shap_plot.png"):
    age_gender_shap = np.sum(shap_values[:, :4096], axis=1)
    ita_shap = np.sum(shap_values[:, 4096:4097], axis=1)
    objects_shap = np.sum(shap_values[:, 4097 : 4097 + 768], axis=1)
    nsfw_shap = np.sum(shap_values[:, 4097 + 768 : 4097 + 768 + 768], axis=1)
    scene_shap = np.sum(shap_values[:, 4097 + 768 + 768 :], axis=1)
    thamiris_scene_shap = np.sum(shap_values[:, 4097 + 768 + 768 + 256 :], axis=1)

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
        thamiris_scene_shap_mean
    )


def save_data(data, filename):
    df = pd.DataFrame(
        data, columns=["Metric", "Age Gender", "Ita", "Objects", "NSFW", "Scene", "Thamiris Scene"]
    )
    df.to_csv(filename, index=True)
    logging.info(f"Data saved to {filename}")


def load_ensemble_model(path_to_model):
    bst = xgb.Booster()
    bst.load_model(path_to_model)
    return bst
