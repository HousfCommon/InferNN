import os
import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import utils
from torch.utils import model_zoo

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

resnet_pretrained = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
                     }

# yolov5_pretrained = {'small': 'yolov5s',
#                      'mi'}

def load_csv():
    return 0


def load_img(dataset=''):
    img_list = []
    for dirname, _, filenames in os.walk('../dataset/'+dataset):
        for filename in filenames:
            img_list.append(os.path.join(dirname, filename))

    idx_list = np.random.choice(len(img_list),3)
    return img_list, idx_list

# def load_dataset_transform():
#     imgs = datasets.ImageFolder('../dataset', transform)
#     idx_list = np.random.choice(len(imgs), 3)

#     return imgs, idx_list

def load_dataset():
    imgs = datasets.ImageFolder('../dataset')
    idx_list = np.random.choice(len(imgs), 3)

    return imgs, idx_list


# def resnet18(pretrained=False, **kwargs):
# 	if pretrained:
# 		model = models.resnet18().load_state_dict(model_zoo.load_url(resnet_pretrained['resnet18']))
# 	return model


def detect(model='yolo'):
    img_list, idx_list = load_dataset()

    if model == 'yolo':
        model = torch.hub.load("ultralytics/yolov5", "yolov5m")
    # elif model == 'resnet':
    #     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    # DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(DEV)
    
    for i in idx_list:
        img = img_list[i][0]
        results = model(img)

        results.save()

def inference():
    img_list, idx_list = load_img('others')
    model = torch.hub.load("ultralytics/yolov5", "yolov5m")

    for i in img_list:
        # img = img_list[i]
        results = model(i)

        results.save()
    

if __name__=='__main__':
    # load_csv()
    # load_dataset()
    detect('yolo')
    # inference()