import torch
import torchvision
import cv2
from torchvision import transforms as trn
from torch.nn import functional as F
from torch.autograd import Variable as V
import transformers
from PIL import Image
from ultralytics import YOLO
from mtcnn import MTCNN
from keras.models import model_from_json
from tensorflow import compat as compat

tf = compat.v1

from tensorflow.python.keras import backend as K
from skimage import transform

from model.itamodel import SkinTone


def infer_objects(image: str):
    model = YOLO("models/objects/yolov11.pt")

    results = model(image)
    # Access the results
    for result in results:
        xyxy = (
            result.boxes.xyxy
        )  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        names = [
            result.names[cls.item()] for cls in result.boxes.cls.int()
        ]  # class name of each box

        image = cv2.imread(image)
        color = (256, 0, 0)
        thickness = 2
        for box in xyxy:
            start_point = (int(box[0].item()), int(box[1].item()))
            end_point = (int(box[2].item()), int(box[3].item()))
            image = draw_rectangle(image, start_point, end_point)

        cv2.imwrite("test.jpg", image)
        print(names)


def infer_nsfw(image: str, device: torch.device):
    predict = transformers.pipeline(
        "image-classification",
        model="AdamCodd/vit-base-nsfw-detector",
        device=device,
        use_fast=True,
    )

    img = Image.open(image)
    results = predict(img)
    print(results)


def infer_scenes(image: str):

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
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print("{} prediction on {}".format(arch, img_name))
    # output the prediction
    for i in range(0, 5):
        print("{:.3f} -> {}".format(probs[i], classes[idx[i]]))


def infer_age_gender(image: str):
    faces: list = get_face_imgs(image)
    age_info = get_age_gender(faces)
    ita_info = get_ita(faces)


def get_age_gender(faces: list):
    tf.disable_v2_behavior()
    model_age = model_from_json(
        open("models/model_age/vgg16_agegender_model.json").read()
    )

    config = tf.ConfigProto(device_count={"GPU": 0})
    sess = tf.Session(config=config)
    K.set_session(sess)

    age = []
    child = []
    gender = []
    with sess:
        model_age.load_weights("models/model_age/vgg16_agegender.hdf5")

        for index, face in enumerate(faces):
            cv2.imwrite(f"testface{index}.jpg", face)
            face = transform.resize(face, (128, 128))
            prediction = model_age.predict(face[None, :, :, :])

            age.append(prediction[0][0].tolist())
            child.append(prediction[1][0][0].item())
            gender.append(prediction[2][0][0].item())

        print(
            *list(
                map(
                    lambda l: list(
                        map(lambda tup: f"{tup[0]} probability: {tup[1]}", l)
                    ),
                    map(
                        lambda l: sorted(
                            enumerate(l), key=lambda x: x[1], reverse=True
                        ),
                        age,
                    ),
                )
            )
        )

        print(*list(map(lambda prob: f"Probability of being a child: {prob}", child)))
        print(*list(map(lambda prob: f"Probability of being a man: {prob}", gender)))

    return age, child, gender


def get_ita(faces: list):
    skinModel = SkinTone("models/fitzpatrick/shape_predictor_68_face_landmarks.dat")

    config = tf.ConfigProto(device_count={"GPU": 0})
    sess = tf.Session(config=config)
    K.set_session(sess)

    skin_ita = []

    with sess:
        for face in faces:
            ita, patch = skinModel.ITA(face)
            skin_ita.append(ita)
    print(
        *list(
            map(
                lambda ita: f"Ita value: {ita}, Ita description: {skinModel.ita2str(ITA=ita)[0]}",
                skin_ita,
            )
        )
    )

    return skin_ita


def draw_rectangle(image, start, end):
    color = (256, 0, 0)
    thickness = 2
    return cv2.rectangle(image, start, end, color, thickness)


def get_face_imgs(img_path: str):
    detector = MTCNN()
    image = cv2.imread(img_path)
    faces = detector.detect_faces(image)

    for face in faces:
        x, y, w, h = face["box"]

        rec_image = draw_rectangle(image, (x, y), (x + w, y + h))

    cv2.imwrite("test.jpg", rec_image)

    return list(
        map(
            lambda face: image.copy()[
                face["box"][1] : face["box"][1] + face["box"][3],
                face["box"][0] : face["box"][0] + face["box"][2],
            ],
            faces,
        )
    )
