import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import shutil
import torch.nn.functional as F
EPOCH = 10
BATCH_SIZE = 13
learning_rate = 0.001
resnet18 = models.resnet18(pretrained=False,num_classes=2).cuda()
optimzier = torch.optim.Adam(resnet18.parameters(),lr=learning_rate)
loss_func = nn.CrossEntropyLoss().cuda()

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
        #print(self.img_list)
        self.transform = transform

    def __getitem__(self, index):
        label = self.img_list[index].split("/")[-2]
        img = Image.open(self.img_list[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, int(label) # 得到的是字符串，故要进行类型转换

    def __len__(self):
        return len(self.img_list)

transform = transforms.Compose(
    [transforms.Resize((640,480)),transforms.ToTensor()]) 

# 这里可视化就没有进行 transform 操作
#i=0
train_loader = DataLoader(My_DataSet("data/processed/train/",transform), batch_size=13, shuffle=True)
valid_loader = DataLoader(My_DataSet("data/processed/train/",transform), batch_size=1, shuffle=True)
'''
for data, label in DataLoader(My_DataSet("data/processed/train/",transform), batch_size=4, shuffle=True):
    print(data[0].shape)
    img = data[0].numpy()
    img = np.transpose(img, (1, 2, 0)) 
    plt.imshow(img)
    plt.show()
'''
#print(i)

for epoch in range(EPOCH):
    for step, (b_x,b_y) in enumerate(train_loader):
        output = resnet18(b_x.cuda())
        #print(output)
        #print(b_y)
        loss = loss_func(output,b_y.cuda())
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        print('Loss:', loss.item())
        print(step)
'''
        if step % 50 == 0: 
            test_output, last_layer = resnet18(test_x)
            pred_y = torch.Tensor.max(test_output,1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
'''
