from ultralytics import YOLO
from transformers import ViTImageProcessor, AutoModelForImageClassification
import numpy as np
from facenet_pytorch import MTCNN
import logging
import torchvision
from torchvision import transforms as trn
import torch
from torch.autograd import Variable as V
from PIL import Image
from tqdm import tqdm
from .thamirismodel import vit_small

import keras
from keras.models import model_from_json

from tensorflow import compat as compat
from model.itamodel import SkinTone

tf = compat.v1


from model.feature_separate import get_age_gender_vector, get_ita_vector
from utils import ensure_dir


def get_feature_vector(dataloader, path_to_store=None, torch_device=None):
    arr = None
    iteration = 0

    device = torch_device

    objects_model = YOLO("models/objects/yolov11.pt", verbose=False)
    nsfw_processor = ViTImageProcessor.from_pretrained(
        "AdamCodd/vit-base-nsfw-detector"
    )
    nsfw_model = AutoModelForImageClassification.from_pretrained(
        "AdamCodd/vit-base-nsfw-detector"
    ).to(device)
    mtcnn_model = MTCNN(keep_all=True)

    tf.disable_v2_behavior()
    age_model = model_from_json(
        open("models/model_age/vgg16_agegender_model.json").read()
    )
    skin_model = SkinTone("models/fitzpatrick/shape_predictor_68_face_landmarks.dat")

    # th architecture to use
    arch = "alexnet"
    # load the pre-trained weights
    model_file = "models/scenes/alexnet_places365.pth.tar"
    scene_model = torchvision.models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    scene_model.load_state_dict(state_dict)
    scene_model.eval()

    scene_thamiris_model = vit_small(patch_size=16)
    scene_thamiris_state_dict = torch.load(
        "models/scenes_thamiris/thamiris_FSL_places600_best.pth", map_location=device
    )
    scene_thamiris_model.load_state_dict(scene_thamiris_state_dict, strict=False)

    for batch_images, _ in tqdm(dataloader):
        logging.info(f"Processing batch {iteration}")
        iteration += 1
        batch_arr = np.concatenate(
            (
                get_faces_vector(
                    batch_images,
                    mtcnn_model=mtcnn_model,
                    model_age=age_model,
                    skinModel=skin_model,
                ),
                get_objects_vector(batch_images, model=objects_model),
                get_nsfw_vector(
                    batch_images,
                    device=torch_device,
                    processor=nsfw_processor,
                    model=nsfw_model,
                ),
                get_scene_vector(batch_images, model=scene_model),
                get_scene_thamiris_vector(
                    batch_images, device=torch_device, model=scene_thamiris_model
                ),
            ),
            axis=1,
        )
        if arr is None:
            arr = batch_arr
        else:
            arr = np.concatenate((arr, batch_arr), axis=0)
        logging.info(
            f"Batch {iteration} processed, images processed: {arr.shape[0]}, feature size: {arr.shape[1]}"
        )

    ensure_dir(path_to_store.rsplit("/", 1)[0])
    np.save(path_to_store, arr)
    logging.info(f"Features saved to {path_to_store}")


def get_faces_vector(batch, mtcnn_model=None, model_age=None, skinModel=None):
    batch_face_vector = []
    for i, img in enumerate(batch):
        faces = get_face_imgs(img, mtcnn=mtcnn_model)
        if len(faces) != 0:
            age_gender_vector = get_age_gender_vector(faces, model_age=model_age)
            ita_vector = get_ita_vector(faces, skinModel=skinModel)
            face_vector = np.concatenate((ita_vector, age_gender_vector), axis=1)
        else:
            face_vector = np.zeros((1, 4097))  # Default vector if no faces detected
        # logging.info(f"Got face vector for {i} image: {face_vector.shape}")
        batch_face_vector.append(face_vector)
    batch_face_vector = np.array(batch_face_vector).squeeze()
    logging.info("AgeGender and Ita feature processed")
    logging.info(f"Batch face vector shape: {batch_face_vector.shape}")
    # logging.info(f"Batch face vector shape: {batch_face_vector.shape}")
    return batch_face_vector


def get_face_imgs(img, mtcnn=None):
    img = img.permute(1, 2, 0)  # Convert from CHW to HWC format
    if mtcnn is None:
        mtcnn = MTCNN(keep_all=True)

    faces = mtcnn(img)
    # logging.info("FACES")
    if faces == None:
        # logging.info("No faces detected")
        return []

    faces = (faces.numpy() * 255).astype(np.uint8)
    # logging.info(f"Detected {len(faces)} faces")

    return faces


def get_objects_vector(batch, model=None):
    if model is None:
        model = YOLO("models/objects/yolov11.pt", verbose=False)

    layer = model.model.model[8]
    hook_handles = []
    features = []

    def hook(_, __, output):
        features.append(output.detach())

    hook_handles.append(layer.register_forward_hook(hook))
    batch = batch.float() / 255.0  # Normalize the batch

    model(batch)

    for handle in hook_handles:
        handle.remove()
    logging.info("Object detection feature processed")
    feature = features[-1]
    feature = feature.reshape((feature.shape[0], feature.shape[1], -1)).mean(dim=(2))
    feature = feature.cpu().numpy()
    logging.info(f"Object feature shape: {feature.shape}")
    return feature


def get_nsfw_vector(batch, device=None, processor=None, model=None):
    if processor is None or model is None:
        processor = ViTImageProcessor.from_pretrained("AdamCodd/vit-base-nsfw-detector")
        model = AutoModelForImageClassification.from_pretrained(
            "AdamCodd/vit-base-nsfw-detector"
        ).to(device)

    features = []

    def hook(_, __, output):
        features.append(output.detach())

    handle = model.vit.layernorm.register_forward_hook(hook)

    inputs = processor(images=batch, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)

    handle.remove()
    logging.info("NSFW feature processed")
    feature = features[0][:, 0, :]
    feature = feature.cpu().numpy()
    logging.info(f"NSFW feature shape: {feature.shape}")
    return feature


def get_scene_vector(batch, model=None):
    features = []

    def hook(_, __, output):
        features.append(output.detach())

    if model is None:
        # th architecture to use
        arch = "alexnet"
        # load the pre-trained weights
        model_file = "models/scenes/alexnet_places365.pth.tar"
        model = torchvision.models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)

    model.eval()

    # load the image transformer
    centre_crop = trn.Compose(
        [
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # load the class label
    file_name = "models/scenes/categories_places365.txt"
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(" ")[0][3:])
    classes = tuple(classes)

    handle = model.features[-1].register_forward_hook(hook)
    for img in batch:
        img = img.permute(1, 2, 0)  # Convert from CHW to HWC format
        img = Image.fromarray(img.numpy())
        img = centre_crop(img).unsqueeze(0)
        input_img = V(img)

        # forward pass
        model.forward(input_img)
    handle.remove()
    logging.info("Scene Classification feature processed")
    feature = np.array(features)
    if feature.shape[1] == 1:
        feature = feature.squeeze()
    feature = feature.reshape((feature.shape[0], feature.shape[1], -1))
    feature = feature.mean(axis=(2))
    logging.info(feature.shape)

    return feature


def get_scene_thamiris_vector(batch, device, model=None):
    if model is None:
        model = vit_small(patch_size=16)
        state_dict = torch.load(
            "models/scenes_thamiris/thamiris_FSL_places600_best.pth",
            map_location=device,
        )
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    batch = batch.float() / 255.0  # Normalize the batch
    batch = batch.to(device)
    with torch.no_grad():
        feature = model(batch)
    logging.info("Scene Classification with Thamiris Few Shot Model feature processed")
    logging.info(feature.shape)
    feature = feature.cpu().numpy()
    return feature
