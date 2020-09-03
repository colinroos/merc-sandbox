import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
import cv2.cv2 as cv
from util import constants as c
from box_corners import BoxCorners
from util.files import *


# noinspection PyTypeChecker
class MainUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        uic.loadUi('user_interface/test_window.ui', self)

        files = find_files_glob('data/2017/images/', '*.jpg')
        self.bc = BoxCorners(files[0])

        # Define image label
        self.img = self.findChild(QtWidgets.QLabel, 'label')

        # Define Adjustments
        self.slider_1 = self.findChild(QtWidgets.QSlider, 'horizontalSlider_2')
        self.spinbox_1 = self.findChild(QtWidgets.QSpinBox, 'spinBox_2')
        self.slider_2 = self.findChild(QtWidgets.QSlider, 'horizontalSlider_3')
        self.spinbox_2 = self.findChild(QtWidgets.QSpinBox, 'spinBox_3')
        self.slider_3 = self.findChild(QtWidgets.QSlider, 'horizontalSlider')
        self.spinbox_3 = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox')
        self.slider_4 = self.findChild(QtWidgets.QSlider, 'horizontalSlider_4')
        self.spinbox_4 = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox_2')

        # Define the get image pushbutton and connect its callback function
        self.btn_image = self.findChild(QtWidgets.QPushButton, 'pushButton')
        self.btn_image.clicked.connect(self.render_image)

        # Define staring values
        self.spinbox_1.setValue(self.bc.block_size)
        self.spinbox_2.setValue(self.bc.kernel_size)
        self.spinbox_3.setValue(self.bc.free_parameter)
        self.spinbox_4.setValue(self.bc.threshold)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.render_image()

    def render_image(self):
        self.bc.block_size = self.spinbox_1.value()
        self.bc.kernel_size = self.spinbox_2.value()
        self.bc.free_parameter = self.spinbox_3.value()
        self.bc.threshold = self.spinbox_4.value()

        self.bc.run()

        height, width, channels = self.bc.out_image.shape
        bytes_per_line = 3 * width

        qimg = QImage(self.bc.out_image.data, width, height, bytes_per_line,
                            QImage.Format_RGB888).rgbSwapped()

        qimg = qimg.scaled(int(self.img.frameGeometry().width() * 0.99),
                           int(self.img.frameGeometry().height() * 0.99),
                           1)

        self.img.setPixmap(QPixmap.fromImage(qimg))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    window.show()
    window.render_image()
    app.exec_()
