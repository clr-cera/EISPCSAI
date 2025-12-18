from eisp import visualization, proxy_tasks
import os
import pandas as pd
import numpy as np


def generate_visualizations():
    features: proxy_tasks.FeatureVectors = proxy_tasks.FeatureVectors.from_files(
        "./features"
    )
    features.get_all_features()['Age_Gender'] = np.nan_to_num(features.get_all_features()["Age_Gender"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    features.get_all_features()['ITA'] = np.nan_to_num(features.get_all_features()["ITA"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    for feature_name, feature_vector in features.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

    pca_features = features.apply_pca()
    features.get_all_features()['Age_Gender'] = np.nan_to_num(features.get_all_features()["Age_Gender"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    features.get_all_features()['ITA'] = np.nan_to_num(features.get_all_features()["ITA"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    for feature_name, feature_vector in features.get_all_features().items():
        print(f"Feature name after NaN handling: {feature_name}, Nan count: {np.isnan(feature_vector).sum()}")



    for feature_name, feature_vector in pca_features.get_all_features().items():
        print(f"Feature after PCA: {feature_name}, Shape: {feature_vector.shape}")

    if not os.path.exists("visualizations/"):
        os.makedirs("visualizations/")
        os.makedirs("visualizations/tsne_per_feature/")
        os.makedirs("visualizations/umap_per_feature/")

    labels = (pd.read_csv("rcpd/rcpd_annotation_fix.csv")["csam"] * 1).to_numpy()

    print("Generating visualizations...")
    print("Generating t-SNE visualization...")
    visualization.plot_tsne(
        features=pca_features, labels=labels, save_path="visualizations/rcpd_tsne.png"
    )
    print("Generating UMAP visualization...")
    visualization.plot_umap(
        features=pca_features, labels=labels, save_path="visualizations/rcpd_umap.png"
    )
    print("Generating per-feature visualizations...")
    print("Generating UMAP per-feature visualizations...")
    try:
        visualization.plot_umap_per_feature(
            features=pca_features,
            labels=labels,
            save_dir="visualizations/umap_per_feature/",
        )
    except Exception as e:
        print(f"UMAP per-feature visualization failed: {e}")
    print("Generating t-SNE per-feature visualizations...")
    try:
        visualization.plot_tsne_per_feature(
            features=pca_features,
            labels=labels,
            save_dir="visualizations/tsne_per_feature/",
        )
    except Exception as e:
        print(f"t-SNE per-feature visualization failed: {e}")
    print("Visualizations generated and saved in the visualizations/ directory.")
