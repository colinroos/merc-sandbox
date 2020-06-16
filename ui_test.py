import sys
from PyQt5 import QtWidgets, QtCore, uic, QtGui
import cv2
from time import sleep
from box_locate import BoxLocate

class MainUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        uic.loadUi('user_interface/CropWindow.ui', self)

        # Define spinboxes
        self.spinbox_rows = self.findChild(QtWidgets.QSpinBox, 'spinBox')
        self.spinbox_rows.setValue(10)
        self.spinbox_spacing = self.findChild(QtWidgets.QSpinBox, 'spinBox_2')
        self.spinbox_spacing.setValue(100)
        self.spinbox_height = self.findChild(QtWidgets.QSpinBox, 'spinBox_3')
        self.spinbox_height.setValue(234)
        self.spinbox_wstart = self.findChild(QtWidgets.QSpinBox, 'spinBox_4')
        self.spinbox_wstart.setValue(200)
        self.spinbox_wend = self.findChild(QtWidgets.QSpinBox, 'spinBox_5')
        self.spinbox_wend.setValue(2500)
        self.spinbox_hstart = self.findChild(QtWidgets.QSpinBox, 'spinBox_6')
        self.spinbox_hstart.setValue(200)

        # Define image label
        self.img = self.findChild(QtWidgets.QLabel, 'label')

        # Connect callbacks
        self.spinbox_rows.valueChanged.connect(self.value_ch)
        self.spinbox_spacing.valueChanged.connect(self.value_ch)
        self.spinbox_height.valueChanged.connect(self.value_ch)
        self.spinbox_wstart.valueChanged.connect(self.value_ch)
        self.spinbox_wend.valueChanged.connect(self.value_ch)
        self.spinbox_hstart.valueChanged.connect(self.value_ch)

        self.edge_detect = BoxLocate()

        self._flag = False

    def value_ch(self):
        self.render_image()

    def resizeEvent(self, e):
    #     if not self._flag:
    #         self._flag = True
    #         self.render_image()
    #         QtCore.QTimer.singleShot(100, lambda: setattr(self, '_flag', False))
        super().resizeEvent(e)

    def render_image(self):
        hstart = self.spinbox_hstart.value()
        rows = self.spinbox_rows.value()
        row_height = self.spinbox_height.value()
        wstart = self.spinbox_wstart.value()
        wend = self.spinbox_wend.value()
        spacing = self.spinbox_spacing.value()

        # Draw boxes on the image
        # Draw 1st row

        temp_img = self.edge_detect.run()
        row_offset = hstart
        for row in range(rows):
            temp_img = cv2.rectangle(temp_img, (wstart, row_offset), (wend, row_offset + row_height), (0, 255, 0), 7)
            row_offset += spacing + row_height

        if len(temp_img.shape) != 3:
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)

        height, width, channels = temp_img.shape
        bytes_per_line = 3 * width

        # cv2.circle(self.edge_detector1.out_img, tuple(self.template_pts[2]), 5, c.BGR_GREEN, -1)
        qimg = QtGui.QImage(temp_img.data, width, height, bytes_per_line,
                            QtGui.QImage.Format_RGB888).rgbSwapped()

        qimg = qimg.scaled(self.img.frameGeometry().width(), self.img.frameGeometry().height(), 1)

        self.img.setPixmap(QtGui.QPixmap.fromImage(qimg))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    window.show()
    window.render_image()
    app.exec_()
