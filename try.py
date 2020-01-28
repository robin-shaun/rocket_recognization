import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()

        self.resize(1000, 700)
        self.setWindowTitle("航宝儿识火箭")

        self.label_img = QLabel(self)
        self.label_vid = QLabel(self)
        #self.label.setText("显示图片")
        self.label_img.setFixedSize(480, 640)
        self.label_img.move(30, 10)
        self.label_vid.setFixedSize(480, 640)
        self.label_vid.move(520,10)
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
                print(self.count)
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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())