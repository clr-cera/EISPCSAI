# native packages
import argparse

import torch

# local packages
import datasets
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.description='Attacking CSAI detection with Proxy Tasks ensembling'

# Dataset manipulation
parser.add_argument("--download_dataset", help='This is the dataset to be downloaded', dest='download_dataset_name', type=str, default=None)
parser.add_argument("--download_models", help="This is a boolean value to download all models", dest="download_models", action='store_true')

args = parser.parse_args()

if args.download_dataset_name:
    datasets.download_dataset(args.download_dataset_name)

if args.download_models:
    model.download_models()

#model.infer_objects(dataset="sentiment-dataset", device=device)