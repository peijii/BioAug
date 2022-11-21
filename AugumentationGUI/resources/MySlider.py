from PyQt5.Qt import *
from PyQt5 import QtWidgets, QtCore
from PyQt5.Qt import QLabel

class MySlider1(QtWidgets.QSlider):

    def __init__(self, parent=None, *args, **kwargs):
        super(MySlider1, self).__init__(parent, *args, **kwargs)
        self.setupUi()

    def setupUi(self):
        self.label = QLabel(self)

    def mousePressEvent(self, evt):
        super().mousePressEvent(evt)
        y = (self.height() - self.label.height()) / 2
        x = self.width() - ((1 - self.value() / (self.maximum() - self.minimum())) * self.width() - self.label.width())
        self.label.move(x, y)

    def mouseMoveEvent(self, evt):
        super().mouseMoveEvent(evt)
        self.true_value = self.value()
        y = (self.height() - self.label.height()) / 2
        x = self.width() - ((1 - self.value() / (self.maximum() - self.minimum())) * (self.width() - self.label.width())) - 13
        #self.label.show()
        self.label.move(x, y)
        self.label.setText(str(self.true_value))
        self.label.adjustSize()

    def mouseReleaseEvent(self, evt):
        super().mouseReleaseEvent(evt)
        #self.label.hide()

class MySlider2(QtWidgets.QSlider):

    def __init__(self, parent=None, *args, **kwargs):
        super(MySlider2, self).__init__(parent, *args, **kwargs)
        self.setupUi()

    def setupUi(self):
        self.label = QLabel(self)

    def mousePressEvent(self, evt):
        super().mousePressEvent(evt)
        y = (self.height() - self.label.height()) / 2
        x = self.width() - ((1 - self.value() / (self.maximum() - self.minimum())) * self.width() - self.label.width())
        self.label.move(x, y)

    def mouseMoveEvent(self, evt):
        super().mouseMoveEvent(evt)
        self.true_value = self.value() * 5
        y = (self.height() - self.label.height()) / 2
        x = self.width() - ((1 - self.value() / (self.maximum() - self.minimum())) * (self.width() - self.label.width())) - 13
        #self.label.show()
        self.label.move(x, y)
        self.label.setText(str(self.true_value))
        self.label.adjustSize()

    def mouseReleaseEvent(self, evt):
        super().mouseReleaseEvent(evt)
        #self.label.hide()

class MySlider3(QtWidgets.QSlider):

    def __init__(self, parent=None, *args, **kwargs):
        super(MySlider3, self).__init__(parent, *args, **kwargs)
        self.setupUi()

    def setupUi(self):
        self.label = QLabel(self)

    def mousePressEvent(self, evt):
        super().mousePressEvent(evt)
        y = (self.height() - self.label.height()) / 2
        x = self.width() - ((1 - self.value() / (self.maximum() - self.minimum())) * self.width() - self.label.width())
        self.label.move(x, y)

    def mouseMoveEvent(self, evt):
        super().mouseMoveEvent(evt)
        self.true_value = self.value()
        y = (self.height() - self.label.height()) / 2
        x = self.width() - ((1 - self.value() / (self.maximum() - self.minimum())) * (self.width() - self.label.width())) - 13
        #self.label.show()
        self.label.move(x, y)
        self.label.setText(str(self.true_value))
        self.label.adjustSize()

    def mouseReleaseEvent(self, evt):
        super().mouseReleaseEvent(evt)
        #self.label.hide()
