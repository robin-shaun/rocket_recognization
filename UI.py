import matplotlib.pyplot as plt
import PyQt5
from PyQt5 import QtMultimedia
from PyQt5.QtWidgets import QApplication, QWidget
import sys
import text2speech
from text2speech import Ws_Param
import forward
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