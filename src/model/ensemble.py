import logging
import xgboost as xgb
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def train_ensemble_sentiment(path_to_features, path_to_labels):
    # Load features and labels
    X = np.load(path_to_features)

    # Define the feature sizes
    feature_sizes = [4096, 1, 768, 768, 256]

    y = pd.read_csv(path_to_labels, sep=";").iloc[:, 2].values
    logging.info(f"Labels shape: {y.shape}")
    logging.info(f"Features shape: {X.shape}")

    # 0 indexing
    y = y - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

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
    }

    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=128)

    # Uncomment the following lines to load the model from a file
    # bst = xgb.Booster()
    # bst.load_model("models/ensemble/ensemble_sentiment.json")

    X_test = xgb.DMatrix(X_test)
    predictions = bst.predict(X_test)
    predicted_classes = predictions.argmax(axis=1)

    logging.info(
        "Accuracy on test data: {:.1f}%".format(
            accuracy_score(y_test, predicted_classes) * 100
        )
    )
    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig("confusion_matrix.png")

    bst.save_model("models/ensemble/ensemble_sentiment.json")
