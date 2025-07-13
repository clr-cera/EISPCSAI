import logging
from matplotlib import pyplot as plt
import xgboost as xgb
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import shap


def train_ensemble_sentiment(path_to_features, path_to_labels):
    train_first_question(path_to_features, path_to_labels)


def train_first_question(path_to_features, path_to_labels):
    # Load features and labels
    X = np.load(path_to_features)

    # Define the feature sizes
    feature_sizes = [4096, 1, 768, 768, 256]

    # Average all Q1 labels
    dfy = pd.read_csv(path_to_labels, sep=";")
    col_idxs = [2 + 26 * i for i in range(5)]
    y = dfy.iloc[:, col_idxs].mean(axis=1).values
    y = np.round(y)

    logging.info(f"Labels shape: {y.shape}")
    logging.info(f"Features shape: {X.shape}")

    # 0 indexing
    y = y - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    logging.info(f"Train shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Test shape: {X_test.shape}, {y_test.shape}")

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Set parameters for XGBoost
    params = {
        "tree_method": "hist",
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 9,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbosity": 3,
        "max_depth": 7,
        "eta": 0.1,
    }

    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=128)

    # Uncomment the following lines to load the model from a file
    # bst = xgb.Booster()
    # bst.load_model("models/ensemble/ensemble_sentiment.json")

    X_test = xgb.DMatrix(X_test)
    predictions = bst.predict(X_test)
    predicted_classes = predictions.argmax(axis=1)

    acc = accuracy_score(y_test, predicted_classes)

    logging.info("Accuracy on test data: {:.1f}%".format(acc * 100))
    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig("confusion_matrix.png")
    plt.clf()

    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_train)

    age_gender_shap = np.sum(shap_values[:, :4096], axis=1)
    ita_shap = np.sum(shap_values[:, 4096:4097], axis=1)
    objects_shap = np.sum(shap_values[:, 4097 : 4097 + 768], axis=1)
    nsfw_shap = np.sum(shap_values[:, 4097 + 768 : 4097 + 768 + 768], axis=1)
    scene_shap = np.sum(shap_values[:, 4097 + 768 + 768 :], axis=1)

    age_gender_shap_mean = np.abs(np.mean(age_gender_shap))
    ita_shap_mean = np.abs(np.mean(ita_shap))
    objects_shap_mean = np.abs(np.mean(objects_shap))
    nsfw_shap_mean = np.abs(np.mean(nsfw_shap))
    scene_shap_mean = np.abs(np.mean(scene_shap))

    print("Age Gender importance:", age_gender_shap_mean)
    print("Ita importance:", ita_shap_mean)
    print("Objects importance:", objects_shap_mean)
    print("NSFW importance:", nsfw_shap_mean)
    print("Scene importance:", scene_shap_mean)
    plt.bar(
        ["Age Gender", "Ita", "Objects", "NSFW", "Scene"],
        [
            age_gender_shap_mean,
            ita_shap_mean,
            objects_shap_mean,
            nsfw_shap_mean,
            scene_shap_mean,
        ],
    )
    plt.title(f"SHAP Feature Importance, acc= {acc * 100:.1f}%")
    plt.savefig("shap_bar_1.png", dpi=300, bbox_inches="tight")

    bst.save_model("models/ensemble/ensemble_sentiment_1.json")
