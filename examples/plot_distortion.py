from scipy.io import loadmat
import matplotlib.pyplot as plt
from bioaug.Distortion import *

# 读取MAT文件，假设文件名为'emg_example1.mat'
data = loadmat('emg_example1.mat')

# 确认数据只有一列，并提取该列数据
emg_data = data['emg'][0][:, np.newaxis]
nonlinearDis = Distortion(p=1.0, phase_shift=0.05, distortion_type='phase')

emg_data_add_nonlinear = nonlinearDis(emg_data)

# 设置全局字体大小
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 20})

# 绘制带坐标轴的图像
plt.figure(figsize=(10, 6))
plt.plot(emg_data[:, 0], linestyle='-', color='blue', label='Original Signal')
plt.plot(emg_data_add_nonlinear[:, 0], linestyle='-', color='red', label='Augmented Signal')

# 添加图例并设置位置为右上角
plt.legend(loc='upper right', fontsize=20, frameon=True)

plt.title('Phase Distortion')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('with_axes_Phase_Distortion.png')  # 保存带坐标轴的图像

# 绘制不带坐标轴的图像
plt.figure(figsize=(10, 6))
plt.plot(emg_data[:, 0], linestyle='-', color='blue', label='Original Signal')
plt.plot(emg_data_add_nonlinear[:, 0], linestyle='-', color='red', label='Augmented Signal')
# 添加图例并设置位置为右上角
plt.legend(loc='upper right', fontsize=20, frameon=True)

plt.title('Phase Distortion')
plt.xticks([])  # 隐藏x轴刻度标签
plt.yticks([])  # 隐藏y轴刻度标签
plt.savefig('without_axes_Phase_Distortion.png')  # 保存不带坐标轴的图像

plt.show()
