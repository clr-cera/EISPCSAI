import os
import numpy as np


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_features(path_to_features: str, feature_sizes=[4096, 1, 768, 768, 256, 384]):
    features = np.load(path_to_features)

    split_vectors = np.split(features, np.cumsum(feature_sizes)[:-1], axis=1)

    agegender_vector = split_vectors[0]
    ita_vector = split_vectors[1]
    objects_vector = split_vectors[2]
    nsfw_vector = split_vectors[3]
    scene_vector = split_vectors[4]
    thamiris_scene_vector = split_vectors[5]

    return (
        agegender_vector,
        ita_vector,
        objects_vector,
        nsfw_vector,
        scene_vector,
        thamiris_scene_vector,
    )
