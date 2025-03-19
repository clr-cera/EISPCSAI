import wget, os, sys
import gdown
import torch
from git import Repo

def download_models():
    ensure_dir('models/')
    
    # Yolov7.pt
    ensure_dir('models/objects/')

    download_wget('https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt','models/objects/yolov7.pt')
    if not os.path.exists('models/objects/yolov7'):
        Repo.clone_from('https://github.com/WongKinYiu/yolov7.git', 'models/objects/yolov7/')
    download_gdown('1SAUaYzNJdQeZ2r1gmf6tLFB6oRclB4RA', 'models/objects/coco_categories.csv')

    # Whole_Wideresnet18_places365.pth
    download_gdown('1D6bGoJHuzXJhnr5KI70Zj1PkbGWWtWij', 'models/scenes/')
    # Nsfw_mobilenet2.224x224.h5
    download_gdown('1t8cAnS8rNBQU8vo16CDAiBL0RuTJdesi', 'models/nsfw_model/')
    # Shape_Predictor_68_face_landmarks.dat
    download_gdown('1AEtQ2s4k5R7IKdrK6vs_zqH_DvXIeFUK', 'models/fitzpatrick/')
    # Model_Age
    download_gdown('1ZF33ousEHhAwK8MmNXpuwmvVtXilVAJ_', 'models/model_age/')

def ensure_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def download_wget(url: str, out: str):
    if not os.path.exists(out):
        wget.download(url=url,out=out)

def download_gdown(id: str, out: str):
    if not os.path.exists(out):
        gdown.download(id=id, output=out)

def infer_objects(dataset: str, device: torch.device):
    # I'm sorry for the crimes I am about to commit
    sys.path.append('./models/yolov7')
    from models.experimental import attempt_load # type: ignore
    from utils.general import check_img_size, non_max_suppression, scale_coords # type: ignore
    from utils.datasets import LoadImages # type: ignore
    from utils.torch_utils import select_device, TracedModel # type: ignore

