from eisp import visualization, proxy_tasks
import os
import pandas as pd


def generate_visualizations():
    features: proxy_tasks.FeatureVectors = proxy_tasks.FeatureVectors.from_files(
        "./features"
    )

    for feature_name, feature_vector in features.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

    pca_features = features.apply_pca()

    for feature_name, feature_vector in pca_features.get_all_features().items():
        print(f"Feature: {feature_name}, Shape: {feature_vector.shape}")

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
