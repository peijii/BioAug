# -*- coding: utf-8 -*-
"""
# @file name  : demo_gui.py
# @author     : Peiji, Chen
# @date       : 2022/04/08
# @brief      :
"""
from PyQt5.Qt import *
from PyQt5 import QtWidgets
from PyQt5.Qt import QDialog, QApplication, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.io import loadmat, savemat

SOURCE1_PATH = os.path.join(os.path.dirname(__file__), 'resources')
SOURCE2_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils')

sys.path.append(SOURCE1_PATH)
sys.path.append(SOURCE2_PATH)

from augmentations import *
from augmentation_ui import Ui_Dialog


class AugmentationUI(QDialog, Ui_Dialog):
    App = QApplication(sys.argv)

    def __init__(self, parent=None, *args, **kwargs) -> None:
        super(AugmentationUI, self).__init__(parent, *args, **kwargs)
        self.setupUi(self)
        self.init_parameter()
        self.init_widget()
        self.init_widget2()

    def init_parameter(self):
        self.wSize = 500
        self.channels = 6
        # 
        self.jittering_flag = 0; self.scaling_flag = 0; self.permutation_flag = 0; self.magnitudewarping_flag = 0; 
        self.timewarping_flag = 0; self.randomsampling_flag = 0; self.randomcutout_flag = 0

        # Augmentation method
        self.do_jitter = False; self.do_scaling = False; self.do_permutation = False; self.do_magitudewarping = False; 
        self.do_timewarping = False; self.do_randomsampling = False; self.do_randomcutout = False
        # initialize file and folder flag
        self.file_names = None
        self.folder_name = None

    def init_widget(self):
        # TODO
        self.fig = plt.Figure()
        self.canvas = FC(self.fig)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.rawDataArea.setLayout(layout)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_yticks([])
        self.ax.set_xticks([0, self.wSize])

    def init_widget2(self):
        # TODO
        self.fig2 = plt.Figure()
        self.canvas2 = FC(self.fig2)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas2)
        self.augmentatedArea.setLayout(layout)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_yticks([])
        self.ax2.set_xticks([0, self.wSize])

    def set_loading(self, value, maximum=1):
        self.progressBar.setValue(value * 100)
        self.progressBar.setMaximum(maximum * 100)
        self.progressBar.setTextVisible(value != 0)
        AugmentationUI.App.processEvents()

    def select_file(self):
        fd = QFileDialog(self)
        self.file_names, _ = fd.getOpenFileName(self, '选择文件', './', 'All(*.*)')

    def select_folder(self):
        fd = QFileDialog(self)
        self.folder_name = fd.getExistingDirectory(self, '选择文件夹', './')
        print(self.folder_name)

    def jittering(self):
        self.jittering_flag += 1
        if self.jittering_flag % 2 == 1:
            print('enter gaussianNoise')
            self.do_jitter = True
        else:
            self.do_jitter = False

    def scaling(self):
        self.scaling_flag += 1
        if self.scaling_flag % 2 == 1:
            print('enter scaling')
            self.do_scaling = True
        else:
            self.do_scaling = False

    def permutation(self):
        self.permutation_flag += 1
        if self.permutation_flag % 2 == 1:
            print('enter permutation')
            self.do_permutation = True
        else:
            self.do_permutation = False

    def magnitudewarping(self):
        self.magnitudewarping_flag += 1
        if self.magnitudewarping_flag % 2 == 1:
            print('enter magnitudewarping')
            self.do_magitudewarping = True
        else:
            self.do_magitudewarping = False

    def timewarping(self):
        self.timewarping_flag += 1
        if self.timewarping_flag % 2 == 1:
            print('enter timewarping')
            self.do_timewarping = True
        else:
            self.do_timewarping = False

    def randomsampling(self):
        self.randomsampling_flag += 1
        if self.randomsampling_flag % 2 == 1:
            print('enter randomsampling')
            self.do_randomsampling = True
        else:
            self.do_randomsampling = False

    def randomcutout(self):
        self.randomcutout_flag += 1
        if self.randomcutout_flag % 2 == 1:
            print('enter randomcutout')
            self.do_randomcutout = True
        else:
            self.do_randomcutout = False

    def plot_raw(self):
        selectChannel = int(self.selectChannelCB.currentText())
        self.ax.cla() 
        if self.file_names:
            self.data = self.fileIO(self.file_names)['data'][:, selectChannel-1]
            self.ax.plot(range(self.wSize), self.data, 'r')
            self.canvas.draw()
                  
    def plot_aug(self):
        try:
            self.ax2.cla()
            a = self.data
            a = self.perform_augmentation(a)
            self.ax2.plot(range(self.wSize), a, 'b')
            self.canvas2.draw()
        except AttributeError:
            pass

    @staticmethod
    def fileIO(fileName):
        if fileName:
            data = loadmat(fileName)
            return data
        return

    def perform_augmentation(self, a):
        if self.do_jitter is True:
            print('-----Do GaussianNoise-----')
            snr = self.JitteringSlider.true_value
            print(snr)
            fn = GaussianNoise(p=1, SNR=snr)
            a = fn(a)

        if self.do_scaling is True:
            print('-----Do Scaling-----')
            sigma = self.ScalingSlider.true_value
            print(sigma)
            fn = Scaling(sigma=sigma, p=1, wSize=self.wSize, channels=1)
            a = fn(a)

        if self.do_permutation is True:
            print('-----Do Permutation-----')
            nPerm = self.nPermBox.value()
            Length = self.LengthBox.value()
            print(nPerm, Length)
            fn = Permutation(nPerm=nPerm, minSegLength=Length, p=1, wSize=self.wSize, channels=1)
            a = fn(a)

        if self.do_magitudewarping is True:
            print('-----Do MagnitudeWarping-----')
            sigma = self.MWSigmaSlider.true_value
            mw_knot = self.MWKnotBox.value()
            print(sigma, mw_knot)
            fn = MagnitudeWarping(sigma=sigma, knot=mw_knot, p=1, wSize=self.wSize, channels=1)
            a = fn(a)

        if self.do_timewarping is True:
            print('-----Do TimeWarping-----')
            sigma = self.TWSigmaSlider.true_value
            tw_knot = self.TWKnotBox.value()
            print(sigma, tw_knot)
            fn = TimeWarping(sigma=sigma, knot=tw_knot, p=1, wSize=self.wSize, channels=1)
            a = fn(a)

        if self.do_randomsampling is True:
            print('-----Do RandomSampling-----')
            nSample = self.RSSlider.true_value
            print(nSample)
            fn = RandomSampling(nSample=nSample, p=1, wSize=self.wSize, channels=1)
            a = fn(a)

        if self.do_randomcutout is True:
            print('-----Do RandomCutout-----')
            area = self.AreaLengthSlider.true_value
            num = self.RCNumBox.value()
            print(area, num)
            fn = RandomCutout(p=1, area=area, num=num, wSize=500, channels=1, default=0)
            a = fn(a)

            if len(a.shape) == 1:
                a = a[:, np.newaxis]

            for n in range(a.shape[1]):
                for m in range(a.shape[0]):
                    print(a[m, n])
                    if a[m, n] == 0:
                        a[m, n] = np.nan

        return a

    def process_save(self):
        dataList = self.getDataPath()
        print(dataList[0])
        base_path = os.path.join(self.folder_name, '..')
        for i in range(len(dataList)):
            source_path = dataList[i]
            data_name = source_path.split('/')[-1]
            day_name = source_path.split('/')[-2]
            subject_name = source_path.split('/')[-3]
            wSize_name = source_path.split('/')[-4]
            a = loadmat(source_path)['data']

            if self.do_jitter is True:
                sigma = self.JitteringSlider.true_value
                fn = Jittering(sigma=sigma, p=1, wSize=self.wSize, channels=self.channels)
                a = fn(a)
                wSize_name = wSize_name + '_J'

            if self.do_scaling is True:
                sigma = self.ScalingSlider.true_value
                fn = Scaling(sigma=sigma, p=1, wSize=self.wSize, channels=self.channels)
                a = fn(a)
                wSize_name = wSize_name + '_S'

            if self.do_permutation is True:
                nPerm = self.nPermBox.value()
                Length = self.LengthBox.value()
                fn = Permutation(nPerm=nPerm, minSegLength=Length, p=1, wSize=self.wSize, channels=self.channels)
                a = fn(a)
                wSize_name = wSize_name + '_P'

            if self.do_magitudewarping is True:
                sigma = self.MWSigmaSlider.true_value
                mw_knot = self.MWKnotBox.value()
                fn = MagnitudeWarping(sigma=sigma, knot=mw_knot, p=1, wSize=self.wSize, channels=self.channels)
                a = fn(a)
                wSize_name = wSize_name + '_MW'

            if self.do_timewarping is True:
                sigma = self.TWSigmaSlider.true_value
                tw_knot = self.TWKnotBox.value()
                fn = TimeWarping(sigma=sigma, knot=tw_knot, p=1, wSize=self.wSize, channels=self.channels)
                a = fn(a)
                wSize_name = wSize_name + '_TW'

            if self.do_randomsampling is True:
                nSample = self.RSSlider.true_value
                fn = RandomSampling(nSample=nSample, p=1, wSize=self.wSize, channels=self.channels)
                a = fn(a)
                wSize_name = wSize_name + '_RS'

            if self.do_randomcutout is True:
                area = self.AreaLengthSlider.true_value
                num = self.RCNumBox.value()
                fn = RandomCutout(p=1, area=area, num=num, wSize=500, channels=self.channels, default=0)
                a = fn(a)
                wSize_name = wSize_name + '_RC'

            save_path = os.path.join(base_path, wSize_name, subject_name, day_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            save_path = os.path.join(save_path, data_name)

            savemat(save_path, {'data': a})

            self.set_loading(i+1, len(dataList))

    def getDataPath(self, subject='all', day='all'):
        """
        Return the absoluate path of every input data    
        """
        if subject == 'all' and day == 'all':
            dataPathList = [os.path.join(self.folder_name, subject_path, day_path, gesture_path)
                            for subject_path in os.listdir(self.folder_name)
                            for day_path in os.listdir(os.path.join(self.folder_name, subject_path))
                            for gesture_path in os.listdir(os.path.join(self.folder_name, subject_path, day_path))]    

        return dataPathList


if __name__ == '__main__':
    app = AugmentationUI.App
    window = AugmentationUI()
    window.show()
    sys.exit(app.exec_())