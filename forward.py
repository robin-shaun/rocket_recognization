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

import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os
import ctypes
from sdl2 import *

import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os
import ctypes
from sdl2 import *

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": "xiaoyan", "tte": "utf8","ent":"aisound"}
        self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}
        #使用小语种须使用以下方式，此处的unicode指的是 utf16小端的编码方式，即"UTF-16LE"”
        #self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-16')), "UTF8")}

    # 生成url
    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url

def on_message(ws, message):
    try:
        message =json.loads(message)
        code = message["code"]
        sid = message["sid"]
        audio = message["data"]["audio"]
        audio = base64.b64decode(audio)
        status = message["data"]["status"]
        print(message)
        if status == 2:
            print("ws is closed")
            ws.close()
        if code != 0:
            errMsg = message["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
        else:

            with open('./demo.pcm', 'ab') as f:
                f.write(audio)

    except Exception as e:
        print("receive msg,but parse exception:", e)



# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        d = {"common": wsParam.CommonArgs,
             "business": wsParam.BusinessArgs,
             "data": wsParam.Data,
             }
        d = json.dumps(d)
        print("------>开始发送文本数据")
        ws.send(d)
        if os.path.exists('./demo.pcm'):
            os.remove('./demo.pcm')

    thread.start_new_thread(run, ())



class audio_ctx:  # Context

    def __init__(self, fid, flag):
        self.f = open(fid, 'rb')
        self.runflag = flag

    def __del__(self):
        self.f.close


def audio_cb(udata, stream, len):
    c = ctypes.cast(udata, ctypes.py_object).value
    buf = c.f.read(2048)
    if not buf:
        SDL_PauseAudio(1)
        c.runflag = 0
        return
    SDL_memset(stream, 0, len)
    SDL_MixAudio(
        stream, ctypes.cast(
            buf, POINTER(ctypes.c_ubyte)), len, SDL_MIX_MAXVOLUME)


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
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    rfile.setFileName("/home/robin/rocket_recognization/demo.pcm")
    rfile.open(PyQt5.QtCore.QIODevice.ReadOnly)
    audio_output.start(rfile)
    app.exec()
    plt.show()