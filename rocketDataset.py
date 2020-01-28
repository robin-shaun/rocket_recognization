from torch.utils.data import Dataset
import os
from PIL import Image
# 创建一个火箭数据集类：继承 Dataset
class RocketDataSet(Dataset):
    def __init__(self, img_dir, transform=None):
        super(RocketDataSet, self).__init__()
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