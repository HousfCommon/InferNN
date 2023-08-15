import os
import torch 
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch import nn


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

classes = ('cat', 'dog')
batch_size = 5

def load_img(dataset=''):
    img_list = []
    for dirname, _, filenames in os.walk('../dataset/'+dataset):
        for filename in filenames:
            img_list.append(os.path.join(dirname, filename))

    idx_list = np.random.choice(len(img_list),3)
    return img_list, idx_list

def load_dataset():
    imgs = datasets.ImageFolder('../dataset', transform)
    idx_list = np.random.choice(len(imgs), 10)

    return imgs, idx_list


def detect():
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model = models.resnet18(pretrained=True)
    dim = model.fc.in_features
    model.fc = nn.Linear(dim,2)
    model.load_state_dict(torch.load('weights.pt',map_location=torch.device('cpu')))
    model.eval()

    img_list, idx_list=load_dataset()

    for i in idx_list:
        img = img_list[i][0].unsqueeze_(0)
        data = Variable(img)
        output = model(data)
        _,indices=torch.sort(output, descending=True)

        percentage = torch.nn.functional.softmax(output, dim=1)[0]*100

        if percentage[0] > percentage[1]:
            out = 'cat|{:>4.1f}%'.format(percentage[0])
        else:
            out = 'dog|{:>4.1f}%'.format(percentage[1])

        print('Image Name:{},predict:{},'.format(img_list.imgs[i][0], out))



    


    
    

    



if __name__=='__main__':
    # load_csv()
    # load_dataset()
    detect()
    # inference()