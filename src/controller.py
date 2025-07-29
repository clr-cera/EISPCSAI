import argparse
import torch

import datasets
import model
import dataloader


class Controller:
    def __init__(self):
        self.args = _parseArguments()
        self.torchdevice = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def start(self):

        # Download for Reproducibility
        if self.args.download_dataset_name:
            datasets.download_dataset(self.args.download_dataset_name)

        if self.args.download_models:
            model.download.download_models()

        # Model Testing
        self.test_separate()

        # Feature Extraction
        if self.args.features_image:
            model.feature_separate.get_feature_vector(
                self.args.features_image, self.torchdevice
            )

        if self.args.sentiment_dataset_feature:
            sentiment_dataloader = dataloader.get_sentiment_dataloader(
                batch_size=32, shuffle=False
            )
            model.feature_multiple.get_feature_vector(
                sentiment_dataloader,
                path_to_store="features/sentiment",
                torch_device=self.torchdevice,
            )

        # Train Ensemble Sentiment
        if self.args.train_ensemble_sentiment:
            model.ensemble.train_ensemble_sentiment(
                path_to_features="features/sentiment.npy",
                path_to_labels="sentiment-dataset/annotations.csv",
                feature_sizes=[4096, 1, 768, 768, 256, 384],
            )
        if self.args.train_ensemble_sentiment_pca:
            model.ensemble.train_ensemble_sentiment(
                path_to_features="features/pca_features.npy",
                path_to_labels="sentiment-dataset/annotations.csv",
                feature_sizes=[256, 1, 256, 256, 256, 256],
            )

        if self.args.train_ensemble_sentiment_combinatorics:
            model.ensemble.train_ensemble_sentiment_combination(
                path_to_features="features/sentiment.npy",
                path_to_labels="sentiment-dataset/annotations.csv",
                feature_sizes=[4096, 1, 768, 768, 256, 384],
            )
        if self.args.train_ensemble_sentiment_combinatorics_pca:
            model.ensemble.train_ensemble_sentiment_combination(
                path_to_features="features/pca_features.npy",
                path_to_labels="sentiment-dataset/annotations.csv",
                feature_sizes=[256, 1, 256, 256, 256, 256],
            )

        # Feature Normalization
        if self.args.sentiment_features_pca:
            model.process_features.process_pca_features(
                path_to_features="features/sentiment.npy"
            )

        # Visualization
        if self.args.generate_tsne:
            model.process_features.generate_tsne(perplexity=30)

        if self.args.generate_tsne_per_feature:
            model.process_features.generate_tsne_per_feature(perplexity=30)

        if self.args.generate_umap:
            model.process_features.generate_umap(n_neighbors=50, min_dist=0.1)

    def test_separate(self):
        if self.args.test_object:
            model.separate.infer_objects(image=self.args.test_image)

        if self.args.test_nsfw:
            model.separate.infer_nsfw(
                image=self.args.test_image, device=self.torchdevice
            )

        if self.args.test_scene:
            model.separate.infer_scenes(image=self.args.test_image)

        if self.args.test_age_gender_ita:
            model.separate.infer_age_gender(image=self.args.test_image)


def _parseArguments():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.description = "Attacking CSAI detection with Proxy Tasks ensembling"

    # Dataset manipulation
    parser.add_argument(
        "--download_dataset",
        help="This is the dataset to be downloaded. Options: [sentiment]",
        dest="download_dataset_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--download_models",
        help="This is a boolean value to download all models",
        dest="download_models",
        action="store_true",
    )
    parser.add_argument(
        "--test_object",
        help="When this option is set the object detection model will be tested on image from argument --test_image",
        dest="test_object",
        action="store_true",
    )
    parser.add_argument(
        "--test_nsfw",
        help="When this option is set the nsfw model will be tested on image from argument --test_image",
        dest="test_nsfw",
        action="store_true",
    )
    parser.add_argument(
        "--test_scene",
        help="When this option is set the scene model will be tested on image from argument --test_image",
        dest="test_scene",
        action="store_true",
    )
    parser.add_argument(
        "--test_age_gender_ita",
        help="When this option is set the agegender and ita model will be tested on image from argument --test_image",
        dest="test_age_gender_ita",
        action="store_true",
    )
    parser.add_argument(
        "--test_image",
        help="Image to be tested by test arguments",
        dest="test_image",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--features_image",
        help="Image to extract feature vectors and store them",
        dest="features_image",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sentiment-dataset-feature"
        "When this option is set the sentiment dataset will be used to extract feature vectors",
        dest="sentiment_dataset_feature",
        action="store_true",
    )
    parser.add_argument(
        "--train_ensemble_sentiment",
        help="When this option is set the sentiment dataset will be used to train ensemble model",
        dest="train_ensemble_sentiment",
        action="store_true",
    )
    parser.add_argument(
        "--train_ensemble_sentiment_combinatorics",
        help="When this option is set the sentiment dataset will be used to train ensemble model with combinatorics to find optimal feature combination",
        dest="train_ensemble_sentiment_combinatorics",
        action="store_true",
    )
    parser.add_argument(
        "--train_ensemble_sentiment_combinatorics_pca",
        help="When this option is set the sentiment dataset will be used to train ensemble model with combinatorics on pca features to find optimal feature combination",
        dest="train_ensemble_sentiment_combinatorics_pca",
        action="store_true",
    )
    parser.add_argument(
        "--process_pca_sentiment",
        help="When this option is set the features extracted from sentiment dataset are going to be processed with PCA",
        dest="sentiment_features_pca",
        action="store_true",
    )
    parser.add_argument(
        "--train_ensemble_sentiment_pca",
        help="When this option is set the sentiment dataset processed with pca will be used to train ensemble model",
        dest="train_ensemble_sentiment_pca",
        action="store_true",
    )
    parser.add_argument(
        "--generate_tsne",
        help="When this option is set the sentiment dataset processed with pca will be used to generate tsne features",
        dest="generate_tsne",
        action="store_true",
    )
    parser.add_argument(
        "--generate_tsne_per_feature",
        help="When this option is set the sentiment dataset processed with pca will be used to generate tsne visualization per feature",
        dest="generate_tsne_per_feature",
        action="store_true",
    )
    parser.add_argument(
        "--generate_umap",
        help="When this option is set the sentiment dataset processed with pca will be used to generate umap features",
        dest="generate_umap",
        action="store_true",
    )
    return parser.parse_args()
