import sys
from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import text2speech
from text2speech import Ws_Param
import forward
from PIL import Image

class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()

        self.resize(1000, 700)
        self.setWindowTitle("航宝儿识火箭")
        self.move(500,200)

        self.label_img = QLabel(self)
        self.label_vid = QLabel(self)
        self.label_txt = QLabel(self)
        self.label_img.setFixedSize(480, 640)
        self.label_img.move(30, 10)
        self.label_vid.setFixedSize(480, 640)
        self.label_vid.move(520,10)
        self.label_txt.setFixedSize(480, 50)
        self.label_txt.move(520,650)
        self.label_img.setStyleSheet("QLabel{background:white;}")
        self.label_vid.setStyleSheet("QLabel{background:white;}")
        self.timer_camera = QTimer()     #定义定时器
        video = 'hangbaoer.mp4'     #加载视频文件
        self.cap = cv2.VideoCapture(video)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,50)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.timer_camera.start(1000/self.fps)  
        self.timer_camera.timeout.connect(self.openFrame)
        self.count = 0
        
        self.rocket_file = "/home/robin/rocket_recognization/rocket.pcm"
        
        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(200, 660)
        btn.clicked.connect(self.openimage)

    def openFrame(self):     
        """ Slot function to capture frame and process it
        """

        ret,frame = self.cap.read()
        if(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame.data,  width, height, bytesPerLine, 
                                 QImage.Format_RGB888).scaled(self.label_vid.width(), self.label_vid.height())
                self.label_vid.setPixmap(QPixmap.fromImage(q_image)) 
                self.count += 1
                if(self.count==50):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES,50)
                    self.count = 0
            else:
                self.cap.release()
                self.timer_camera.stop()   # 停止计时器

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpeg;;*.png;;All Files(*)")

        jpg = QtGui.QPixmap(imgName).scaled(self.label_img.width(), self.label_img.height())
        self.label_img.setPixmap(jpg)
        img = Image.open(imgName).convert('RGB')
        rocket = forward.net_foward(img)
        wsParam = Ws_Param(APPID='5e2faa83', APIKey='58c05763b09a8d85d9a2f5645f981824',
                    APISecret='1d83a8338cc3e0188c880b9ab514770e',
                    Text=rocket)
        self.label_txt.setText(rocket)
        text2speech.tts(wsParam,self.rocket_file)
        rfile.setFileName(self.rocket_file)
        rfile.open(QIODevice.ReadOnly)
        audio_output.start(rfile)
        

if __name__ == "__main__":
    welcoming = "/home/robin/rocket_recognization/welcoming.pcm"
    format = QtMultimedia.QAudioFormat()
    format.setChannelCount(2)
    format.setSampleRate(8000)
    format.setSampleSize(16)
    format.setCodec("audio/pcm")
    format.setByteOrder(QtMultimedia.QAudioFormat.LittleEndian)
    format.setSampleType(QtMultimedia.QAudioFormat.UnSignedInt)
    audio_output = QtMultimedia.QAudioOutput(format)
    rfile = QFile()
    app = QApplication(sys.argv)
    my = picture()
    my.show()
    rfile.setFileName(welcoming)
    rfile.open(QIODevice.ReadOnly)
    audio_output.start(rfile)
    my.label_txt.setText('我会识别火箭～')
    my.label_txt.setAlignment(Qt.AlignCenter)

    sys.exit(app.exec_())