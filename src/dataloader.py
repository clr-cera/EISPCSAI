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
