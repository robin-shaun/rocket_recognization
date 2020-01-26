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

def path_init(src_path, dst_path, rate=(0.6, 0.2, 0.2)):
    """
    将原始数据按比较分配成 train validation test
    :param src_path: 原始数据路径，要求格式如下
    - src_path
        - class_1
        - class_2
        ...
    :param dst_path: 目标路径
    :param rate: 分配比例，加起来一定要等于 1
    :return:
    """
    # 以下几行是创建如下格式的文件夹
    """
    - img_data
        - train
            - class_1
            - class_2
            ...
        - validation
            - class_1
            - class_2
            ...
        - test
            - class_1
            - class_2
            ...
    """
    try:
        class_names = os.listdir(src_path)  # 获取原始数据所有类别的纯文件名
        print(1)
        dst_path = dst_path + '/' + 'processed'
        os.mkdir(dst_path)  # 创建目标文件夹
        print(2)
        three_paths = [dst_path + '/' +
                       i for i in ['train', 'validation', 'test']]  # 三个文件夹的路径
        for three_path in three_paths:
            os.mkdir(three_path)
            for class_name in class_names:
                os.mkdir(three_path+'/'+class_name)
        # -----------------------------

        dst_train = dst_path + '/' + 'train'
        dst_validation = dst_path + '/' + 'validation'
        dst_test = dst_path + '/' + 'test'

        class_names_list = [src_path + '/' +
                            class_name for class_name in class_names]  # 获取原始数据所有类别的路径

        for class_li in class_names_list:
            imgs = os.listdir(class_li)  # 当前类别所有图片的文件名，不包括路径
            # 得到当前类别的所有图片的路径，指定后缀
            imgs_list = [class_li + '/' +
                         img for img in imgs if (img.endswith("jpeg") or img.endswith("png"))]
            print(len(imgs_list))
            img_num = len(imgs_list)  # 当前类别的图片数量
            # 三个文件夹的数量
            train_num = int(rate[0]*img_num)
            validation_num = int(rate[1]*img_num)
            # test_num = int(rate[2]*img_num)

            for img in imgs_list[0:train_num]:
                # 训练集复制
                src = img
                dst = dst_train + '/' + \
                    img.split('/')[-2] + '/' + img.split('/')[-1]
                # print(src, " ", dst)
                shutil.copy(src=img, dst=dst)
            print("训练集数量：", len(imgs_list[0:train_num]))

            for img in imgs_list[train_num:train_num+validation_num]:
                # 验证集复制
                src = img
                dst = dst_validation + '/' + \
                    img.split('/')[-2] + '/' + img.split('/')[-1]
                # print(src, " ", dst)
                shutil.copy(src=img, dst=dst)
            print("验证集数量：", len(imgs_list[train_num:train_num+validation_num]))

            for img in imgs_list[train_num + validation_num:]:
                # 测试集复制
                src = img
                dst = dst_test + '/' + \
                    img.split('/')[-2] + '/' + img.split('/')[-1]
                # print(src, " ", dst)
                shutil.copy(src=img, dst=dst)
            print("测试集数量：", len(imgs_list[train_num + validation_num:]))

    except:
        print("目标文件夹已经存在或原始文件夹不存在，请检查！")


# # 例程
src_path = 'data/src'
dst_path = 'data/'    
path_init(src_path, dst_path, rate=(0.6, 0.2, 0.2))