import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import umap
from utils import load_features, ensure_dir


def process_pca_features(path_to_features: str, n_components=256):
    (
        agegender_vector,
        ita_vector,
        objects_vector,
        nsfw_vector,
        scene_vector,
        thamiris_scene_vector,
    ) = load_features(path_to_features)

    # Ita vector is not used in PCA, because it is a single value

    np.nan_to_num(agegender_vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    pca_models = generate_pca_models(
        agegender_vector,
        objects_vector,
        nsfw_vector,
        scene_vector,
        thamiris_scene_vector,
        n_components,
    )

    logging.info("PCA models generated")

    pca_variance = get_pca_variance(pca_models)

    logging.info(f"Age gender PCA variance: {pca_variance[0]}")
    logging.info(f"Objects PCA variance: {pca_variance[1]}")
    logging.info(f"NSFW PCA variance: {pca_variance[2]}")
    logging.info(f"Scene PCA variance: {pca_variance[3]}")
    logging.info(f"Thamiris scene PCA variance: {pca_variance[4]}")

    pca_features = generate_pca_features(
        [
            agegender_vector,
            objects_vector,
            nsfw_vector,
            scene_vector,
            thamiris_scene_vector,
        ],
        pca_models,
    )
    pca_features.insert(1, ita_vector)

    save_pca_features(pca_features)

    save_pca_models(pca_models)


def generate_pca_models(
    agegender_vector,
    objects_vector,
    nsfw_vector,
    scene_vector,
    thamiris_scene_vector,
    n_components=32,
):
    agegender_pca_model = PCA(n_components=n_components).fit(agegender_vector)
    objects_pca_model = PCA(n_components=n_components).fit(objects_vector)
    nsfw_pca_model = PCA(n_components=n_components).fit(nsfw_vector)
    scene_pca_model = PCA(n_components=n_components).fit(scene_vector)
    thamiris_scene_pca_model = PCA(n_components=n_components).fit(thamiris_scene_vector)

    return (
        agegender_pca_model,
        objects_pca_model,
        nsfw_pca_model,
        scene_pca_model,
        thamiris_scene_pca_model,
    )


def generate_pca_features(vector_list: list[np.ndarray], pca_models: list[PCA]):
    return [pca.transform(vector) for vector, pca in zip(vector_list, pca_models)]


def save_pca_features(pca_features):
    path_to_store = "features/pca_features.npy"
    ensure_dir(path_to_store.rsplit("/", 1)[0])
    pca_features = np.concatenate(pca_features, axis=1)
    logging.info(f"PCA features shape: {pca_features.shape}")
    np.save(path_to_store, pca_features)
    logging.info(f"PCA features saved to {path_to_store}")


def save_pca_models(pca_models: list[PCA]):
    path_to_store = "models/pca/models.npz"
    ensure_dir(path_to_store.rsplit("/", 1)[0])
    (
        agegender_pca_model,
        objects_pca_model,
        nsfw_pca_model,
        scene_pca_model,
        thamiris_scene_pca_model,
    ) = pca_models
    np.savez(
        path_to_store,
        agegender=agegender_pca_model,
        objects=objects_pca_model,
        nsfw=nsfw_pca_model,
        scene=scene_pca_model,
        thamiris_scene=thamiris_scene_pca_model,
    )
    logging.info(f"PCA models saved to {path_to_store}")


def get_pca_variance(pca_model_list: list[PCA]):
    return [pca.explained_variance_ratio_.sum() for pca in pca_model_list]


def generate_tsne(
    perplexity, random_state=42, path_to_labels="sentiment-dataset/annotations.csv"
):
    pca_features = np.load("features/pca_features.npy")
    logging.info(f"PCA features shape: {pca_features.shape}")
    tsne_model = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
    )
    tsne_features = tsne_model.fit_transform(pca_features)
    np.save("features/tsne_features.npy", tsne_features)
    logging.info(f"TSNE features shape: {tsne_features.shape}")
    logging.info("KL divergence: " + str(tsne_model.kl_divergence_))
    logging.info("TSNE features saved to features/tsne_features.npy")

    for question_number in range(1, 6):
        dfy = pd.read_csv(path_to_labels, sep=";")
        columns = [c for c in dfy.columns if f"Q{question_number}" in c]
        grouped = dfy[columns].copy()
        grouped.columns = ["".join(col.split(".")[1:]) for col in columns]
        mean_by_answer = grouped.T.groupby(by=grouped.columns).mean().T
        y = mean_by_answer.values
        y = np.round(y).astype(int)

        for option, label in enumerate(y.T):
            plt.scatter(
                tsne_features[:, 0], tsne_features[:, 1], s=1, c=label, cmap="viridis"
            )
            plt.title(
                "TSNE Features, question "
                + str(question_number)
                + ", option "
                + str(option + 1)
            )
            ensure_dir("results/tsne")
            plt.savefig(f"results/tsne/tsne_features_Q{question_number}.{option+1}.png")
            plt.clf()
            logging.info(
                f"TSNE features plot saved to results/tsne/tsne_features_Q{question_number}.{option+1}.png"
            )

    return tsne_features


def generate_tsne_rcpd(
    perplexity, random_state=42, path_to_labels="rcpd/rcpd_annotation_processed.csv"
):
    pca_features = np.load("features/pca_features.npy")
    logging.info(f"PCA features shape: {pca_features.shape}")
    tsne_model = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
    )
    tsne_features = tsne_model.fit_transform(pca_features)
    np.save("features/tsne_features.npy", tsne_features)
    logging.info(f"TSNE features shape: {tsne_features.shape}")
    logging.info("KL divergence: " + str(tsne_model.kl_divergence_))
    logging.info("TSNE features saved to features/tsne_features.npy")

    dfy = pd.read_csv(path_to_labels)
    label = dfy["img_category_num"].values

    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], s=1, c=label, cmap="viridis")
    plt.title("TSNE Features")
    ensure_dir("results/tsne")
    plt.savefig(f"results/tsne/tsne_features.png")
    plt.clf()
    logging.info(f"TSNE features plot saved to results/tsne/tsne_features.png")

    return tsne_features


def generate_umap(
    n_neighbors, min_dist, path_to_labels="sentiment-dataset/annotations.csv"
):
    pca_features = np.load("features/pca_features.npy")
    logging.info(f"PCA features shape: {pca_features.shape}")
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    umap_features = umap_model.fit_transform(pca_features)
    np.save("features/umap_features.npy", umap_features)
    logging.info(f"UMAP features shape: {umap_features.shape}")
    logging.info("UMAP features saved to features/umap_features.npy")

    for question_number in range(1, 6):
        dfy = pd.read_csv(path_to_labels, sep=";")
        columns = [c for c in dfy.columns if f"Q{question_number}" in c]
        grouped = dfy[columns].copy()
        grouped.columns = ["".join(col.split(".")[1:]) for col in columns]
        mean_by_answer = grouped.T.groupby(by=grouped.columns).mean().T
        y = mean_by_answer.values
        y = np.round(y).astype(int)

        for option, label in enumerate(y.T):
            plt.scatter(
                umap_features[:, 0], umap_features[:, 1], s=1, c=label, cmap="viridis"
            )
            plt.title(
                "UMAP Features, question "
                + str(question_number)
                + ", option "
                + str(option)
            )
            ensure_dir("results/umap")
            plt.savefig(f"results/umap/umap_features_Q{question_number}.{option}.png")
            plt.clf()
            logging.info(
                f"UMAP features plot saved to results/umap/umap_features_Q{question_number}.{option}.png"
            )

    return umap_features


def generate_umap_rcpd(
    n_neighbors, min_dist, path_to_labels="rcpd/rcpd_annotation_processed.csv"
):
    pca_features = np.load("features/pca_features.npy")
    logging.info(f"PCA features shape: {pca_features.shape}")
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    umap_features = umap_model.fit_transform(pca_features)
    np.save("features/umap_features.npy", umap_features)
    logging.info(f"UMAP features shape: {umap_features.shape}")
    logging.info("UMAP features saved to features/umap_features.npy")

    dfy = pd.read_csv(path_to_labels)
    label = dfy["img_category_num"].values

    plt.scatter(umap_features[:, 0], umap_features[:, 1], s=1, c=label, cmap="viridis")
    plt.title("UMAP Features")
    ensure_dir("results/umap")
    plt.savefig(f"results/umap/umap_features.png")
    plt.clf()
    logging.info(f"UMAP features plot saved to results/umap/umap_features.png")

    return umap_features


def generate_tsne_per_feature(
    perplexity,
    random_state=42,
):
    pca_features = load_features(
        "features/pca_features.npy", feature_sizes=[256, 1, 256, 256, 256, 256]
    )
    feature_names = [
        "agegender",
        "ita",
        "objects",
        "nsfw",
        "scene",
        "thamiris_scene",
    ]

    for name, feature in zip(feature_names, pca_features):

        if feature.shape[1] == 1:
            feature = feature.repeat(
                2, axis=1
            )  # Repeat if single value for visualization

        tsne_model = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
        )
        tsne_features = tsne_model.fit_transform(feature)
        logging.info(f"TSNE features shape for {name}: {tsne_features.shape}")
        logging.info("KL divergence: " + str(tsne_model.kl_divergence_))

        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], s=1)
        plt.title(f"TSNE by {name} feature")
        plt.savefig(f"results/tsne/tsne_features_{name}.png")
        plt.clf()
        logging.info(
            f"TSNE features plot for {name} saved to results/tsne/tsne_features_{name}.png"
        )
    return tsne_features
