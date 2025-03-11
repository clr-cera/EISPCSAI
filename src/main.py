# native packages
import argparse

# local packages
import datasets

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.description='Attacking CSAI detection with Proxy Tasks ensembling'

# Dataset manipulation
parser.add_argument("--download_dataset", help='This is the dataset to be downloaded', dest='download_dataset_name', type=str, default=None)

args = parser.parse_args()

if args.download_dataset_name:
    datasets.download_dataset(args.download_dataset_name)