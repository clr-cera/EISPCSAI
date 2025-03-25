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
parser.add_argument("--test_object", help="When this option is set the object detection model will be tested in an image from the sentiment-dataset", dest="test_object", action='store_true')

args = parser.parse_args()

if args.download_dataset_name:
    datasets.download_dataset(args.download_dataset_name)

if args.download_models:
    model.download_models()


#model.infer_objects(dataset="sentiment-dataset", device=device, test=args.test_object)
#model.infer_nsfw(dataset="sentiment-dataset", device=device, test=True)
#model.infer_scenes(dataset='sentiment-dataset', device=device, test=True)
model.infer_age_gender(dataset='sentiment-dataset', device=device, test=True)