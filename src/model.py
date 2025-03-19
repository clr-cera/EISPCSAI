import wget, os
import gdown
import torch
from git import Repo

def download_models():
    # Yolov7.pt
    download_wget('https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt','models/yolov7.pt')
    Repo.clone_from('https://github.com/WongKinYiu/yolov7.git', 'models/yolov7/')


    # Whole_Wideresnet18_places365.pth
    download_gdown('1D6bGoJHuzXJhnr5KI70Zj1PkbGWWtWij', 'models/scenes/')
    # Nsfw_mobilenet2.224x224.h5
    download_gdown('1t8cAnS8rNBQU8vo16CDAiBL0RuTJdesi', 'models/nsfw_model/')
    # Shape_Predictor_68_face_landmarks.dat
    download_gdown('1AEtQ2s4k5R7IKdrK6vs_zqH_DvXIeFUK', 'models/fitzpatrick/')
    # Model_Age
    download_gdown('1ZF33ousEHhAwK8MmNXpuwmvVtXilVAJ_', 'models/model_age/')


def download_wget(url: str, out: str):
    if not os.path.exists(out):
        wget.download(url=url,out=out)

def download_gdown(id: str, out: str):
    if not os.path.exists(out):
        gdown.download(id=id, output=out)

def infer_objects(dataset: str, device: torch.device):
    path = 'models/yolov7.pt'
    model = torch.hub.load("WongKinYiu/yolov7", "custom", path, trust_repo=True)
    
    results = model("sentiment-dataset/images/0a4d4c7d-e520-4c11-b969-71ceee1b40d0.jpg")
    print(results)