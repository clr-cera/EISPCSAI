import os
import wget
import gdown


def download_models():
    os.makedirs("models/", exist_ok=True)

    # Yolov11.pt
    os.makedirs("models/objects/", exist_ok=True)
    download_wget(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "models/objects/yolov11.pt",
    )

    # Yolov11Pose.pt
    os.makedirs("models/pose/", exist_ok=True)
    download_wget(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt",
        "models/pose/yolov11pose.pt",
    )

    # Places365
    os.makedirs("models/scenes/", exist_ok=True)
    download_wget(
        "http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar",
        "models/scenes/alexnet_places365.pth.tar",
    )
    download_gdown(
        "1boHawW-6hY7l0_jRqqdSt8b7kPbXzxDk", "models/scenes/categories_places365.txt"
    )
    # Shape_Predictor_68_face_landmarks.dat
    download_gdown("1AEtQ2s4k5R7IKdrK6vs_zqH_DvXIeFUK", "models/fitzpatrick/")

    # Model_Age
    os.makedirs("models/model_age/", exist_ok=True)
    download_gdown(
        "1ZF33ousEHhAwK8MmNXpuwmvVtXilVAJ_", "models/model_age/vgg16_agegender.hdf5"
    )
    download_gdown(
        "152W9ijPlnaQQOJsFBJtl5aMEyUKWf5gd",
        "models/model_age/vgg16_agegender_model.json",
    )

    # Model Scene Thamiris
    download_gdown("1RLaTWt_YC6V_Pza6rTQJ2Qfzn3fOJdZ8", "models/scenes_thamiris/")


def download_wget(url: str, out: str):
    if not os.path.exists(out):
        wget.download(url=url, out=out)


def download_gdown(id: str, out: str):
    if not os.path.exists(out):
        gdown.download(id=id, output=out)
