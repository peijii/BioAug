from scipy.io import loadmat
import matplotlib.pyplot as plt

# 读取MAT文件，假设文件名为'emg_example1.mat'
plt.style.use('seaborn-darkgrid')

data = loadmat('emg_example1.mat')

# 确认数据只有一列，并提取该列数据
emg_data = data['emg'][0]

# 设置全局字体大小
plt.rcParams.update({'font.size': 20})

# 绘制带坐标轴的图像
plt.figure(figsize=(10, 6))
plt.plot(emg_data, linestyle='-', color='b')
plt.title('No augmentation')
plt.savefig('with_axes.png')  # 保存带坐标轴的图像

# 绘制不带坐标轴的图像
plt.figure(figsize=(10, 6))
plt.plot(emg_data, linestyle='-', color='b')
plt.title('No augmentation')
plt.xticks([])  # 隐藏x轴刻度标签
plt.yticks([])  # 隐藏y轴刻度标签
plt.savefig('without_axes.png')  # 保存不带坐标轴的图像

# 显示图像
plt.show()