from scipy.io import loadmat
import matplotlib.pyplot as plt
from bioaug.Drift import *

# 读取MAT文件，假设文件名为'emg_example1.mat'
data = loadmat('emg_example1.mat')

# 确认数据只有一列，并提取该列数据
emg_data = data['emg'][0][:, np.newaxis]
drift = SignalDrift(p=1.0, func='exp', drift_rate=0.2, seed=42)

emg_data_add_drift = drift(emg_data)

# 设置全局字体大小
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 20})

# 绘制带坐标轴的图像
plt.figure(figsize=(10, 6))
plt.plot(emg_data_add_drift[:, 0], linestyle='-', color='red', label='Augmented Signal')
plt.plot(emg_data[:, 0], linestyle='-', color='blue', label='Original Signal')


# 添加图例并设置位置为右上角
plt.legend(loc='upper right', fontsize=20, frameon=True)

plt.title('Drift (exp, drift rate=0.2)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('with_axes_drift(exp, drift rate=0.2).png')  # 保存带坐标轴的图像

# 绘制不带坐标轴的图像
plt.figure(figsize=(10, 6))
plt.plot(emg_data_add_drift[:, 0], linestyle='-', color='red', label='Augmented Signal')
plt.plot(emg_data[:, 0], linestyle='-', color='blue', label='Original Signal')

# 添加图例并设置位置为右上角
plt.legend(loc='upper right', fontsize=20, frameon=True)

plt.title('Drift (exp, drift rate=0.2)')
plt.xticks([])  # 隐藏x轴刻度标签
plt.yticks([])  # 隐藏y轴刻度标签
plt.savefig('without_axes_drift(exp, drift rate=0.2).png')  # 保存不带坐标轴的图像

plt.show()
