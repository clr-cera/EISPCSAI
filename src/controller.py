import argparse
import torch

import datasets
import model


class Controller:
    def __init__(self):
        self.args = _parseArguments()
        self.torchdevice = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def start(self):
        if self.args.download_dataset_name:
            datasets.download_dataset(self.args.download_dataset_name)

        if self.args.download_models:
            model.download.download_models()

        self.test_separate()

        if self.args.features_image:
            model.feature_single.get_feature_vector(
                self.args.features_image, self.torchdevice
            )

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
    return parser.parse_args()
