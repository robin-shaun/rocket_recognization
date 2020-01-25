import torch
import torchvision.models as models
import torch.nn as nn
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
EPOCH = 10
BATCH_SIZE = 50
learning_rate = 0.001
resnet18 = models.resnet18(pretrained=True).cuda()
optimzier = torch.optim.Adam(resnet18.parameters(),lr=learning_rate)
loss_func = nn.CrossEntropyLoss().cuda()


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # 读取 csv 文件
        self.data_info = pd.read_csv(csv_path, header=None)
        # 文件第一列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # 第三列是决定是否进行额外操作
        self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # 计算 length
        self.data_len = len(self.data_info.index)
 
    def __getitem__(self, index):
        # 从 pandas df 中得到文件名
        single_image_name = self.image_arr[index]
        # 读取图像文件
        img_as_img = Image.open(single_image_name)
 
        # 检查需不需要额外操作
        some_operation = self.operation_arr[index]
        # 如果需要额外操作
        if some_operation:
            # ...
            # ...
            pass
        # 把图像转换成 tensor
        img_as_tensor = self.to_tensor(img_as_img)
 
        # 得到图像的 label
        single_image_label = self.label_arr[index]
 
        return (img_as_tensor, single_image_label)
 
    def __len__(self):
        return self.data_len
 

custom_mnist_from_images = CustomDatasetFromImages('data/mnist_labels.csv')

for epoch in range(EPOCH):
    for step, (b_x,b_y) in enumerate(train_loader):
        output = resnet18(b_x)[0]
        loss = loss_func(output,b_y)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

        if step % 50 == 0: 
            test_output, last_layer = resnet18(test_x)
            pred_y = torch.Tensor.max(test_output,1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

