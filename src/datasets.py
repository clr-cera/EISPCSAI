import wget
import zipfile
import os
import ssl
import csv



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

            # This file is corrupted, so we remove it
            os.remove("sentiment-dataset/images/0c416a95-6e96-4e8c-bab3-56b682feafe9.jpg")
            csv_file = "sentiment-dataset/annotations.csv"
            # Remove file from csv
            rows = []
            with open(csv_file, "r") as f:
                reader = csv.reader(f, delimiter=";")
                for row in reader:
                    if row[1] != "0c416a95-6e96-4e8c-bab3-56b682feafe9.jpg":
                        rows.append(row)
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerows(rows)

            print("Sentiment dataset downloaded and extracted.")