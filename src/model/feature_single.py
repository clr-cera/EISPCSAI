from ultralytics import YOLO
from transformers import ViTImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torchvision
from torchvision import transforms as trn
from torch.nn import functional as F
from torch.autograd import Variable as V
from mtcnn import MTCNN
import cv2
import keras
from keras.models import model_from_json
from tensorflow import compat as compat
import numpy as np

tf = compat.v1

from tensorflow.python.keras import backend as K
from skimage import transform


def get_feature_vector(image, device):
    object_feature = get_objects_vector(image)
    nsfw_feature = get_nsfw_vector(image)
    scene_feature = get_scene_vector(image)
    faces = get_face_imgs(image)
    age_gender_feature = get_age_gender_vector(faces)


def get_objects_vector(image):
    model = YOLO("models/objects/yolov11.pt", verbose=False)
    layer = model.model.model[-2].m
    hook_handles = []
    features = []

    def hook(_, __, output):
        features.append(output.detach())

    for block in layer:
        hook_handles.append(block.cv3.register_forward_hook(hook))

    model(image)

    for handle in hook_handles:
        handle.remove()
    print("OBJECT")
    feature = features[-1]
    feature = feature.reshape((feature.shape[0], feature.shape[1], -1)).mean(dim=(2))
    print(feature.shape)
    return feature


def get_nsfw_vector(image):
    image = Image.open(image)
    processor = ViTImageProcessor.from_pretrained("AdamCodd/vit-base-nsfw-detector")
    model = AutoModelForImageClassification.from_pretrained(
        "AdamCodd/vit-base-nsfw-detector"
    )

    features = []

    def hook(_, __, output):
        features.append(output.detach())

    handle = model.vit.layernorm.register_forward_hook(hook)

    inputs = processor(images=image, return_tensors="pt")
    model(**inputs)

    handle.remove()
    print("NSFW")
    feature = features[0][:, 0, :]
    print(feature.shape)
    return feature


def get_scene_vector(image):
    features = []

    def hook(_, __, output):
        features.append(output.detach())

    # th architecture to use
    arch = "alexnet"

    # load the pre-trained weights
    model_file = "models/scenes/alexnet_places365.pth.tar"

    model = torchvision.models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
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

    # load the test image
    img_name = image

    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    print("SCENE")
    print(model.features[-1])
    handle = model.features[-1].register_forward_hook(hook)

    model.forward(input_img)
    handle.remove()
    feature = features[0]
    feature = feature.reshape((feature.shape[0], feature.shape[1], -1)).mean(dim=(2))
    print(feature.shape)

    return feature


def get_age_gender_vector(faces):
    tf.disable_v2_behavior()
    model_age = model_from_json(
        open("models/model_age/vgg16_agegender_model.json").read()
    )

    config = tf.ConfigProto(device_count={"GPU": 0})
    sess = tf.Session(config=config)
    K.set_session(sess)

    features = []
    with sess:
        model_age.load_weights("models/model_age/vgg16_agegender.hdf5")

        feature_model = keras.Model(
            inputs=model_age.input,
            outputs=model_age.get_layer("fc2").output,
        )

        for face in faces:
            face = transform.resize(face, (128, 128))
            features.append(feature_model.predict(face[None, :, :, :]))

    print("AGE GENDER")
    print(len(features))
    print([x.shape for x in features])
    feature = tf.stack(features, 1)
    feature = tf.reduce_mean(feature, 1)
    print(feature.shape)

    return feature


def get_face_imgs(img_path: str):
    detector = MTCNN()
    image = cv2.imread(img_path)
    faces = detector.detect_faces(image)

    return list(
        map(
            lambda face: image.copy()[
                face["box"][1] : face["box"][1] + face["box"][3],
                face["box"][0] : face["box"][0] + face["box"][2],
            ],
            faces,
        )
    )
