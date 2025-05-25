from ultralytics import YOLO
import transformers
from PIL import Image


def get_feature_vector(image, device):
    get_objects_vector(image)
    get_nsfw_vector(image, device)


def get_objects_vector(image):
    model = YOLO("models/objects/yolov11.pt")
    detect_layer = model.model.model[-1]
    hook_handles = []
    features = []

    def hook(_, __, output):
        features.append(output.detach())

    for block in detect_layer.cv3:
        hook_handles.append(block[-1].register_forward_hook(hook))
        print(block[-1])

    model(image)

    for handle in hook_handles:
        handle.remove()
    print(len(features))
    feature_ex = features[0].cpu().numpy()
    print(feature_ex.shape)


def get_nsfw_vector(image, device):
    predict = transformers.pipeline(
        "image-classification",
        model="AdamCodd/vit-base-nsfw-detector",
        device=device,
        use_fast=True,
    )
    vit_nsfw = transformers.AutoModel.from_pretrained(
        "AdamCodd/vit-base-nsfw-detector", output_hidden_states=True
    )
    vit_nsfw.to(device)
    img = Image.open(image)
    results = predict(img)
