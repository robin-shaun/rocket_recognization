import torch
from torch.utils.data import DataLoader
from rocketDataset import RocketDataSet
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PyQt5
from PyQt5 import QtMultimedia
from PyQt5.QtWidgets import QApplication, QWidget
import sys
import text2speech
from text2speech import Ws_Param
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

app = QApplication(sys.argv)
format = QtMultimedia.QAudioFormat()
format.setChannelCount(2)
format.setSampleRate(8000)
format.setSampleSize(16)
format.setCodec("audio/pcm")
format.setByteOrder(QtMultimedia.QAudioFormat.LittleEndian)
format.setSampleType(QtMultimedia.QAudioFormat.UnSignedInt)
audio_output = QtMultimedia.QAudioOutput(format)

rfile = PyQt5.QtCore.QFile()



resnet34=torch.load('resnet34_rocket100.pkl')
transform = transforms.Compose([transforms.Resize((640,480)),transforms.ToTensor()]) 
test_loader = DataLoader(RocketDataSet("data/processed/train/",transform), batch_size=1, shuffle=True)
labels = pd.read_csv('labels.csv')
loss_func = torch.nn.CrossEntropyLoss().cuda()
#print(resnet34)
#print(test_loader)
#print(labels)


for step, (b_x,b_y) in enumerate(test_loader):
    img = b_x[0].numpy()
    img = np.transpose(img, (1, 2, 0)) 
    plt.imshow(img)
    output = resnet34(b_x.cuda())
    print(output)
    print(loss_func(output,b_y.cuda()))
    pred_y = int(torch.Tensor.max(output,1)[1].data.cpu().numpy())
    title = labels.loc[:,'class'][pred_y]
    print(title)
    plt.title('这是'+title)

    wsParam = Ws_Param(APPID='5e2faa83', APIKey='58c05763b09a8d85d9a2f5645f981824',
                       APISecret='1d83a8338cc3e0188c880b9ab514770e',
                       Text='这是'+title)
    text2speech.tts(wsParam)
    rfile.setFileName("/home/robin/rocket_recognization/tmp.pcm")
    rfile.open(PyQt5.QtCore.QIODevice.ReadOnly)
    audio_output.start(rfile)
    app.exec()
    plt.show()