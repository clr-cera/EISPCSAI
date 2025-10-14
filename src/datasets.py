import wget
import zipfile
import os
import ssl



# Apparently the dataset server has an invalid SSL certificate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Sentiment analysis


def download_dataset(name: str):
    if name == "sentiment":
        if not os.path.exists("sentiment-dataset"):
            wget.download(
                "https://datasets.simula.no/downloads/image-sentiment/sentiment.zip"
            )
            with zipfile.ZipFile("sentiment.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove("sentiment.zip")
