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
from model_scripts.thamirismodel import vit_small

import keras
from keras.models import model_from_json

from tensorflow import compat as compat
from model_scripts.itamodel import SkinTone
from tensorflow.python.keras import backend as K
from skimage import transform

tf = compat.v1


mtcnn_model = MTCNN(keep_all=True)


def get_object_model():
    objects_model = YOLO("models/objects/yolov11.pt", verbose=False)
    return {"model": objects_model}


def get_nsfw_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nsfw_processor = ViTImageProcessor.from_pretrained(
        "AdamCodd/vit-base-nsfw-detector"
    )
    nsfw_model = AutoModelForImageClassification.from_pretrained(
        "AdamCodd/vit-base-nsfw-detector"
    ).to(device)
    return {"device": device, "processor": nsfw_processor, "model": nsfw_model}


def get_scene_model():
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
    return {"model": scene_model}


def get_scene_thamiris_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_thamiris_model = vit_small(patch_size=16)
    scene_thamiris_state_dict = torch.load(
        "models/scenes_thamiris/thamiris_FSL_places600_best.pth", map_location=device
    )
    scene_thamiris_model.load_state_dict(scene_thamiris_state_dict, strict=False)
    return {"device": device, "model": scene_thamiris_model}


def get_age_model():
    tf.disable_v2_behavior()
    age_model = model_from_json(
        open("models/model_age/vgg16_agegender_model.json").read()
    )
    return {"mtcnn": mtcnn_model, "model": age_model}


def get_ita_model():
    skin_model = SkinTone("models/fitzpatrick/shape_predictor_68_face_landmarks.dat")
    return {"mtcnn": mtcnn_model, "model": skin_model}


def get_age_gender_vector(batch, mtcnn, model=None):

    if model is None:
        tf.disable_v2_behavior()
        model = model_from_json(
            open("models/model_age/vgg16_agegender_model.json").read()
        )

    batch_age_gender_vector = []

    config = tf.ConfigProto(device_count={"GPU": 0})
    sess = tf.Session(config=config)
    K.set_session(sess)

    with sess:
        model.load_weights("models/model_age/vgg16_agegender.hdf5")
        feature_model = keras.Model(
            inputs=model.input,
            outputs=model.get_layer("fc2").output,
        )

        for i, image in enumerate(batch):
            faces = get_face_imgs(image, mtcnn=mtcnn)
            features = []
            if len(faces) != 0:
                for face in faces:
                    if face.shape[0] == 3:
                        face = face.transpose((1, 2, 0))
                    face = transform.resize(face, (128, 128))
                    preds = model.predict(face[None, :, :, :])
                    age = preds[0][0].tolist()
                    index_with_max_prob = np.argmax(age)
                    features.append(
                        (
                            index_with_max_prob,
                            feature_model.predict(face[None, :, :, :]),
                        )
                    )
                features = sorted(features, key=lambda x: x[0])
                batch_age_gender_vector.append(features[0][1])

            else:
                batch_age_gender_vector.append(np.zeros((1, 4096)))
    features = np.array(batch_age_gender_vector).squeeze()

    return features


def get_ita_vector(batch, mtcnn, model=None):
    if model is None:
        model = SkinTone("models/fitzpatrick/shape_predictor_68_face_landmarks.dat")

    batch_ita_vector = []

    config = tf.ConfigProto(device_count={"GPU": 0})
    sess = tf.Session(config=config)
    K.set_session(sess)

    with sess:
        for i, image in enumerate(batch):
            faces = get_face_imgs(image, mtcnn=mtcnn)
            skin_ita = []
            if len(faces) != 0:
                for face in faces:
                    if face.shape[0] == 3:
                        face = face.transpose((1, 2, 0))
                    ita, patch = model.ITA(face)
                    skin_ita.append(ita)
                skin_ita = np.array(skin_ita)
                batch_ita_vector.append(
                    np.mean(skin_ita, axis=0, keepdims=True).reshape((1, 1))
                )
            else:
                batch_ita_vector.append(np.zeros((1, 1)))
    features = np.array(batch_ita_vector).squeeze(axis=2)

    return features


def get_face_imgs(img, mtcnn=None):
    img = img.permute(1, 2, 0)  # Convert from CHW to HWC format
    if mtcnn is None:
        mtcnn = MTCNN(keep_all=True)

    faces = mtcnn(img)
    # logging.info("FACES")
    if faces is None:
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
