
import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from main import FaceRecognition
from photoshot import GetImage
#*******************************************************************************************************************#
#--------------------------------------------------------***--------------------------------------------------------#
#*******************************************************************************************************************#

class Stream_Cam(QThread):
    changePixmap = pyqtSignal(QImage)
    def __init__(self):
        super().__init__()
        self.threadactive = True

    def run(self):
        cap = cv2.VideoCapture(0)
        recognition = FaceRecognition()
        while True:
            image = recognition.run(cap)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(500, 500, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
            if not self.threadactive:
                break

    def restart(self):
        self.threadactive = True

    def stop(self):
        self.threadactive = False
        # self.wait()

#*******************************************************************************************************************#
#--------------------------------------------------------***--------------------------------------------------------#
#*******************************************************************************************************************#

class Photo(QThread):
    hideflag = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.active = True

    def runn(self,name):
        done = 0
        cap = cv2.VideoCapture(0)
        getimg = GetImage()
        while True:
            done = getimg.run(cap,name=name)
            if done != 0:
                break
        self.hideflag.emit(done)
        cap.release()
        cv2.destroyAllWindows()


#*******************************************************************************************************************#
#--------------------------------------------------------***--------------------------------------------------------#
#*******************************************************************************************************************#


class Ui_MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.i = 0

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_show = QtWidgets.QLabel(self.centralwidget)
        self.label_show.setGeometry(QtCore.QRect(320, 50, 441, 431))
        self.label_show.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show.setFrameShape(QtWidgets.QFrame.Box)
        self.label_show.setText("")
        self.label_show.setObjectName("label_show")
        self.Verify_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Verify_Button.setGeometry(QtCore.QRect(60, 180, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.Verify_Button.setFont(font)
        self.Verify_Button.clicked.connect(self.verify_button)
        self.Verify_Button.setStyleSheet("background-color: rgb(136, 138, 133);")
        self.Verify_Button.setObjectName("Verify_Button")
        ############################################################################
        self.Register_Button = QtWidgets.QPushButton(self.centralwidget)
        self.Register_Button.setGeometry(QtCore.QRect(60, 280, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.Register_Button.setFont(font)
        self.Register_Button.clicked.connect(self.register_button)
        self.Register_Button.setStyleSheet("background-color: rgb(136, 138, 133);")
        self.Register_Button.setObjectName("Register_Button")
        #############################################################################
        self.RegisterFrame = QtWidgets.QFrame(self.centralwidget)
        self.RegisterFrame.setGeometry(QtCore.QRect(420, 180, 241, 171))
        self.RegisterFrame.setStyleSheet("background-color: rgb(85, 87, 83);")
        self.RegisterFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.RegisterFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.RegisterFrame.setObjectName("RegisterFrame")
        self.RegisterFrame.hide()
        #******************************************************************************
        self.label_name = QtWidgets.QLabel(self.RegisterFrame)
        self.label_name.setGeometry(QtCore.QRect(10, 20, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_name.setFont(font)
        self.label_name.setStyleSheet("color: rgb(243, 243, 243);")
        self.label_name.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_name.setAlignment(QtCore.Qt.AlignCenter)
        self.label_name.setObjectName("label_name")
        #*******************************************************************************
        self.Ok_Button = QtWidgets.QPushButton(self.RegisterFrame)
        self.Ok_Button.setGeometry(QtCore.QRect(80, 130, 89, 25))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.Ok_Button.setFont(font)
        self.Ok_Button.clicked.connect(self.ok_button)
        self.Ok_Button.setStyleSheet("background-color: rgb(115, 210, 22);")
        self.Ok_Button.setObjectName("Ok_Button")
        #*******************************************************************************
        self.NameBox = QtWidgets.QLineEdit(self.RegisterFrame)
        self.NameBox.setGeometry(QtCore.QRect(32, 80, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(15)
        self.NameBox.setFont(font)
        self.NameBox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.NameBox.setAlignment(QtCore.Qt.AlignCenter)
        self.NameBox.setObjectName("NameBox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.thread_1 = Stream_Cam()
        self.thread_1.changePixmap.connect(self.setImage)
        self.thread_2 = Photo()
        self.thread_2.hideflag.connect(self.hide_register)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label_show.setPixmap(QPixmap.fromImage(image))
        self.label_show.setScaledContents(True)

    # def killthread(self):
    #     self.thread_1.stop()

    def verify_button(self):
        self.thread_1.restart()
        self.thread_1.start()
        self.RegisterFrame.hide()

    def ok_button(self):
        self.thread_2.start()
        self.thread_2.runn(self.NameBox.text())

    def hide_register(self,hideflag):
        if hideflag:
            self.RegisterFrame.hide()

    def register_button(self):
        # self.i += 1
        # if self.i % 2 == 0:
        #     self.thread_1.restart()
        #     self.thread_1.start()
            
        # else:
        self.thread_1.stop()
        self.RegisterFrame.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Verify_Button.setText(_translate("MainWindow", "Verify"))
        self.Ok_Button.setText(_translate("MainWindow", "OK"))
        self.Register_Button.setText(_translate("MainWindow", "Register"))
        self.label_name.setText(_translate("MainWindow", "Enter your name"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())