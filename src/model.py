import os

import gdown
import wget
import torch
import torchvision
import cv2
from torchvision import transforms as trn
from torch.nn import functional as F
from torch.autograd import Variable as V
import transformers
from PIL import Image
from ultralytics import YOLO

TEST_PATH = "sentiment-dataset/images/0d8a7158-b805-4ce5-967b-7ec631547e71.jpg"

def download_models():
    ensure_dir('models/')
    
    # Yolov11.pt
    ensure_dir('models/objects/')
    download_wget('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt','models/objects/yolov11.pt')

    # Places365
    download_wget('http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar','models/scenes/alexnet_places365.pth.tar')
    # Shape_Predictor_68_face_landmarks.dat
    download_gdown('1AEtQ2s4k5R7IKdrK6vs_zqH_DvXIeFUK', 'models/fitzpatrick/')

    # Model_Age
    download_gdown('1ZF33ousEHhAwK8MmNXpuwmvVtXilVAJ_', 'models/model_age/')

    # Nsfw_mobilenet2.224x224.h5
    #download_gdown('1t8cAnS8rNBQU8vo16CDAiBL0RuTJdesi', 'models/nsfw_model/')
    

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def download_wget(url: str, out: str):
    if not os.path.exists(out):
        wget.download(url=url,out=out)

def download_gdown(id: str, out: str):
    if not os.path.exists(out):
        gdown.download(id=id, output=out)

def infer_objects(dataset: str, device: torch.device, test: bool = False):
    model = YOLO('models/objects/yolov11.pt')
   
    if test:
        
        results = model(TEST_PATH)
        # Access the results
        for result in results: 
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box

            image = cv2.imread(TEST_PATH)
            color = (256, 0, 0)
            thickness = 2
            for box in xyxy:
                start_point = (int(box[0].item()), int(box[1].item()))
                end_point = (int(box[2].item()), int(box[3].item()))
                image = cv2.rectangle(image, start_point, end_point, color, thickness)

            cv2.imwrite('test.jpg', image)
            print(names)

def infer_nsfw(dataset: str, device: torch.device, test: bool = False):
    predict = transformers.pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector",device=device, use_fast=True)

    img = Image.open(TEST_PATH)
    results = predict(img)
    print(results)

def infer_scenes(dataset: str, device: torch.device, test: bool = False):
        
    # th architecture to use
    arch = 'alexnet'

    # load the pre-trained weights
    model_file = 'models/scenes/alexnet_places365.pth.tar'

    model = torchvision.models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()


    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'models/scenes/categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    if test:
        # load the test image
        img_name = TEST_PATH

        img = Image.open(img_name)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print('{} prediction on {}'.format(arch,img_name))
        # output the prediction
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

def infer_age_gender(dataset: str, device: torch.device, test: bool = False):
    pass
