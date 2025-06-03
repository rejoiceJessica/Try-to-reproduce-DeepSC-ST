import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 后端
import matplotlib.pyplot as plt
import os

# 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
train_mse_file = os.path.join(script_dir, "train_data", "train", "train_mse_snr_-10_awgn.mat")
valid_mse_file = os.path.join(script_dir, "train_data", "valid", "valid_mse_snr_-10_awgn.mat")

# 加载 .mat 文件
train_data = loadmat(train_mse_file)
valid_data = loadmat(valid_mse_file)

train_mse = train_data['train_mse']  # (num_epochs, num_batches)
valid_mse = valid_data['valid_mse']  # (num_epochs, num_valid_batches)

# 计算每个 epoch 的平均 MSE
train_mse_mean = np.mean(train_mse, axis=1)  # (num_epochs,)
valid_mse_mean = np.mean(valid_mse, axis=1)  # (num_epochs,)

# 绘制
epochs = np.arange(1, len(train_mse_mean) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_mse_mean, label='Train MSE', color='blue', linewidth=2)
plt.plot(epochs, valid_mse_mean, label='Valid MSE', color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Loss vs. Epoch for Channel=awgn, SNR=-10 dB')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图表
output_path = os.path.join(script_dir, "train_data", "mse_loss_plot.png")
plt.savefig(output_path)
plt.close()
print(f"Saved plot to {output_path}")