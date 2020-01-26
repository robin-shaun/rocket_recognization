# 导入所需要的包
import torch # 1.1.0 版本
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#import cv2

# 创建一个数据集类：继承 Dataset
class My_DataSet(Dataset):
    def __init__(self, img_dir, transform=None):
        super(My_DataSet, self).__init__()
        self.img_dir = img_dir
        class_dir = [self.img_dir + '/' + i for i in os.listdir(self.img_dir)] # 10 个数字的路径
        img_list = []
        for num in range(len(class_dir)):
            img_list += [class_dir[num]+'/'+img_name for img_name in os.listdir(class_dir[num]) if (img_name.endswith("png")or img_name.endswith("jpeg"))]
        self.img_list = img_list # 得到所有图片的路径
        print(self.img_list)
        self.transform = transform

    def __getitem__(self, index):
        label = self.img_list[index].split("/")[-2]
        img = Image.open(self.img_list[index])

        if self.transform is not None:
            img = self.transform(img)
        return img, int(label) # 得到的是字符串，故要进行类型转换

    def __len__(self):
        return len(self.img_list)

transform = transforms.Compose(
    [transforms.Resize((640,480)),transforms.ToTensor()]) 

# 这里可视化就没有进行 transform 操作
#i=0
trainloader = DataLoader(My_DataSet("data/processed/train/",transform), batch_size=13, shuffle=True)
'''
for data, label in DataLoader(My_DataSet("data/processed/train/",transform), batch_size=4, shuffle=True):
    print(data[0].shape)
    img = data[0].numpy()
    img = np.transpose(img, (1, 2, 0)) 
    plt.imshow(img)
    plt.show()
'''
#print(i)