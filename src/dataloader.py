import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision import transforms
import numpy as np


class SentimentDataset(Dataset):
    def __init__(self, transform=None):
        self.img_labels = pd.read_csv("sentiment-dataset/annotations.csv", sep=";")
        self.img_dir = "sentiment-dataset/images"
        if not transform:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = decode_image(img_path)
        if image.shape[0] == 1:  # If grayscale, convert to RGB
            image = image.repeat(3, 1, 1)
        label = self.img_labels.iloc[idx, 2:]
        label = np.array(label, dtype=np.float32)
        if image.shape[0] == 4:
            image = image[:3, :, :]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_sentiment_dataloader(batch_size=32, shuffle=True):
    dataset = SentimentDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def test_dataloader():
    dataloader = get_sentiment_dataloader()
    for images, labels in dataloader:
        print(f"Batch size: {len(images)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Remove this line to iterate through the entire dataset


class RCPDDataloader(Dataset):
    def __init__(self, prepend_img_dir: str, transform=None):
        self.data = pd.read_csv("rcpd/rcpd_annotation_processed.csv")
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
        img_path = os.path.join(self.prepend_img_dir, self.data.iloc[idx, 2][1:])
        image = decode_image(img_path)
        if image.shape[0] == 1:  # If grayscale, convert to RGB
            image = image.repeat(3, 1, 1)
        label = self.data.iloc[idx, -1]
        if image.shape[0] == 4:
            image = image[:3, :, :]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_rcpd_dataloader(prepend_img_dir: str, batch_size=32, shuffle=True):
    dataset = RCPDDataloader(prepend_img_dir=prepend_img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def test_rcpd_dataloader():
    dataloader = get_rcpd_dataloader(prepend_img_dir="rcpd/images/")
    for images, labels in dataloader:
        print(f"Batch size: {len(images)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Remove this line to iterate through the entire dataset
