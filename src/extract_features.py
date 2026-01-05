import shutil
import eisp
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2 as transforms
import pandas as pd
import os

from proxy_tasks import (
    get_nsfw_vector,
    get_objects_vector,
    get_scene_vector,
    get_scene_thamiris_vector,
    get_age_gender_vector,
    get_ita_vector,
    get_pose_vector,
    get_dino_vector,
)
from proxy_tasks import (
    get_nsfw_model,
    get_object_model,
    get_scene_model,
    get_scene_thamiris_model,
    get_age_model,
    get_ita_model,
    get_pose_model,
    get_dino_model,
)


PROXY_FEATURES_FUNCTIONS = [
    get_pose_vector,
    get_nsfw_vector,
    get_objects_vector,
    get_scene_vector,
    get_scene_thamiris_vector,
    get_age_gender_vector,
    get_ita_vector,
]
PROXY_FEATURES_NAMES = [
    "Pose",
    "Nudity",
    "Objects",
    "Scenes_Places",
    "Scenes_Few_Shot",
    "Age_Gender",
    "ITA_Skin_Tone",
]
PROXY_FEATURES_ARGUMENTS_GENERATORS = [
    get_pose_model,
    get_nsfw_model,
    get_object_model,
    get_scene_model,
    get_scene_thamiris_model,
    get_age_model,
    get_ita_model,
]


class RCPDDataloader(Dataset):
    def __init__(self, prepend_img_dir: str, transform=None):
        self.data = pd.read_csv("rcpd/rcpd_annotation_fix.csv")
        self.prepend_img_dir = prepend_img_dir
        if not transform:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.prepend_img_dir, self.data.iloc[idx, 1][1:])
        image = decode_image(img_path)
        if image.shape[0] == 1:  # If grayscale, convert to RGB
            image = image.repeat(3, 1, 1)
        label = self.data.iloc[idx, -1]
        if image.shape[0] == 4:
            image = image[:3, :, :]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_rcpd_dataloader(
    prepend_img_dir: str, batch_size=32, shuffle=False, transform=None
):
    dataset = RCPDDataloader(prepend_img_dir=prepend_img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def extract_features():
    print("Extracting features...")

    prepend_img_dir = input("Input the path to the RCPD images directory: ")

    rcpd_dataloader = get_rcpd_dataloader(
        prepend_img_dir=prepend_img_dir, shuffle=False
    )

    proxy_features_arguments = [gen() for gen in PROXY_FEATURES_ARGUMENTS_GENERATORS]

    store_path = "./features"
    eisp.proxy_tasks.FeatureVectors.extract(
        rcpd_dataloader,
        PROXY_FEATURES_FUNCTIONS,
        PROXY_FEATURES_NAMES,
        proxy_features_arguments,
        store_path,
    )
    print(f"Features extracted and stored in {store_path}")


def extract_dino_features():
    print("Extracting DINO features...")

    if os.path.exists("features") and not os.path.exists("features_all"):
        shutil.move("features", "features_all")
    if os.path.exists("results") and not os.path.exists("results_all"):
        shutil.move("results", "results_all")
    if os.path.exists("visualizations") and not os.path.exists("visualizations_all"):
        shutil.move("visualizations", "visualizations_all")

    prepend_img_dir = input("Input the path to the RCPD images directory: ")
    transform_dino = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    rcpd_dataloader = get_rcpd_dataloader(
        prepend_img_dir=prepend_img_dir, transform=transform_dino, shuffle=False
    )

    store_path = "./features"
    eisp.proxy_tasks.FeatureVectors.extract(
        rcpd_dataloader,
        [get_dino_vector],
        ["DINO"],
        [get_dino_model()],
        store_path,
    )
    print(f"DINO Features extracted and stored in {store_path}")
