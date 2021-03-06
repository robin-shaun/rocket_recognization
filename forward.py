import torch
from torch.utils.data import DataLoader
from rocketDataset import RocketDataSet
from torchvision import transforms
import numpy as np
import pandas as pd
resnet34=torch.load('resnet34_rocket100.pkl')
transform = transforms.Compose([transforms.Resize((640,480)),transforms.ToTensor()]) 
labels = pd.read_csv('labels.csv')

def net_foward(img):
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    img_tensor = transform(img)
    img_tensor=img_tensor.unsqueeze(0)
    output = resnet34(img_tensor.cuda())
    pred_y = int(torch.Tensor.max(output,1)[1].data.cpu().numpy())
    title = '这是'+labels.loc[:,'class'][pred_y]
    print(title)
    return title