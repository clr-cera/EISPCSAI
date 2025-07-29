import os
import wget
import gdown

from utils import ensure_dir


def download_models():
    ensure_dir("models/")

    # Yolov11.pt
    ensure_dir("models/objects/")
    download_wget(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "models/objects/yolov11.pt",
    )

    # Places365
    ensure_dir("models/scenes/")
    download_wget(
        "http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar",
        "models/scenes/alexnet_places365.pth.tar",
    )
    # Shape_Predictor_68_face_landmarks.dat
    download_gdown("1AEtQ2s4k5R7IKdrK6vs_zqH_DvXIeFUK", "models/fitzpatrick/")

    # Model_Age
    download_gdown("1ZF33ousEHhAwK8MmNXpuwmvVtXilVAJ_", "models/model_age/")

    # Model Scene Thamiris
    download_gdown("1RLaTWt_YC6V_Pza6rTQJ2Qfzn3fOJdZ8", "models/scenes_thamiris/")

    # Nsfw_mobilenet2.224x224.h5
    # download_gdown('1t8cAnS8rNBQU8vo16CDAiBL0RuTJdesi', 'models/nsfw_model/')


def download_wget(url: str, out: str):
    if not os.path.exists(out):
        wget.download(url=url, out=out)


def download_gdown(id: str, out: str):
    if not os.path.exists(out):
        gdown.download(id=id, output=out)
