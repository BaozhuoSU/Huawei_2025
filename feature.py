import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Prevent OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ChannelSvdDataset 类（从您提供的代码复制）
class ChannelSvdDataset(Dataset):
    def __init__(self, data_path, label_path=None):
        raw = np.load(data_path, mmap_mode='r')  # float32 (Ns, M, N, 2)
        self.M, self.N = raw.shape[1], raw.shape[2]
        self.X = raw.transpose(0, 3, 1, 2)  # 转成 (Ns, 2, M, N)
        if label_path is not None:
            lab = np.load(label_path, mmap_mode='r')
            if lab.ndim == 4 and lab.shape[-1] == 2:
                H_real = lab[..., 0]
                H_imag = lab[..., 1]
                self.H = (H_real + 1j * H_imag).astype(np.complex64)
            else:
                self.H = lab
        else:
            self.H = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # (2, M, N)
        if self.H is not None:
            h = self.H[idx]  # complex64 (M, N)
            h = torch.from_numpy(h)
            return x, h
        else:
            return x

if __name__ == "__main__":
    # 数据路径
    root = "./CompetitionData1"
    test_sets = []

    # 加载测试数据
    for idx in range(1, 2):
        td = os.path.join(root, f"Round1TrainData{idx}.npy")
        if os.path.exists(td):
            test_sets.append(ChannelSvdDataset(td, label_path=None))
        else:
            print(f"Warning: {td} not found!")
    assert test_sets, "No test data found!"

    # 计算 Frobenius 范数的 dB 值
    h_db_values = []
    h_db_sum = None
    total_samples = 0
    singular_values_all = []
    U_all = []
    Vt_all = []

    for i, dataset in enumerate(test_sets):
        print(f"Processing testData{i + 1}.npy...")
        for idx in range(len(dataset)):
            x = dataset[idx]  # (2, M, N)
            # 将接收信号 X 转为复数形式
            x_complex = torch.complex(x[0], x[1])  # (M, N)
            # 计算每个元素的模 |H| = sqrt(real^2 + imag^2)
            h_magnitude = torch.abs(x_complex)  # (64, 64)
            # 转换为 dB: 10 * log10(|H|^2)
            h_db = 10 * torch.log10(h_magnitude ** 2 + 1e-12)  # (64, 64)，加 1e-8 避免 log(0)
            # 展平并收集所有值
            h_db_values.extend(h_db.flatten().tolist())

            if h_db_sum is None:
                h_db_sum = h_db.clone()  # 初始化
            else:
                h_db_sum += h_db
            total_samples += 1

            # SVD计算
            h_magnitude_np = h_magnitude.numpy()  # 转换为numpy数组
            U, s, Vt = np.linalg.svd(h_magnitude_np, full_matrices=False)  # SVD分解
            singular_values_all.append(s)  # 存储奇异值
            U_all.append(U)  # 存储左奇异向量
            Vt_all.append(Vt)  # 存储右奇异向量的转置

    # 转换为 numpy 数组以便统计
    h_db_values = np.array(h_db_values)
    singular_values_all = np.array(singular_values_all)
    U_all = np.array(U_all)
    Vt_all = np.array(Vt_all)

    # 计算平均值
    mean_h_db = h_db_sum / total_samples  # (64, 64)
    mean_U = np.mean(U_all, axis=0)  # 平均左奇异向量
    mean_S = np.mean(singular_values_all, axis=0)  # 平均奇异值
    mean_Vt = np.mean(Vt_all, axis=0)  # 平均右奇异向量
    mean_S_diag = np.zeros((64, 64))
    np.fill_diagonal(mean_S_diag, mean_S)
    # # 统计分析
    # mean_db = np.mean(h_db_values)
    # std_db = np.std(h_db_values)
    # min_db = np.min(h_db_values)
    # max_db = np.max(h_db_values)
    # median_db = np.median(h_db_values)
    #
    # print("\n统计结果 (|H|_F in dB):")
    # print(f"均值: {mean_db:.2f} dB")
    # print(f"标准差: {std_db:.2f} dB")
    # print(f"最小值: {min_db:.2f} dB")
    # print(f"最大值: {max_db:.2f} dB")
    # print(f"中位数: {median_db:.2f} dB")

    # 可视化：直方图
    # plt.figure(figsize=(8, 6))
    # plt.hist(h_db_values, bins=50, density=False, alpha=0.7, color='blue')
    # plt.title('Distribution of |H| in dB')
    # plt.xlabel('|H|_F (dB)')
    # plt.ylabel('Density')
    # plt.grid(True)
    # plt.show()

    # mean_h_db = h_db_sum / total_samples  # (64, 64)
    # mean_h_db_np = mean_h_db.numpy()  # 转为 numpy 数组以便可视化
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(mean_h_db_np, cmap='viridis', cbar_kws={'label': '|H| (dB)'})
    # plt.title('Average |H| in dB Across All Samples (64x64)')
    # plt.xlabel('Column Index')
    # plt.ylabel('Row Index')
    # plt.show()

    # 2. 平均 U 热力图
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_U, cmap='viridis', cbar_kws={'label': 'U Value'})
    plt.title('Average U (Left Singular Vectors)')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    # 3. 平均 S 热力图
    plt.subplot(2, 2, 2)
    sns.heatmap(mean_S_diag, cmap='viridis', cbar_kws={'label': 'Singular Value'})
    plt.title('Average S (Singular Values)')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Sample Index')

    # 4. 平均 Vt 热力图
    plt.subplot(2, 2, 3)
    sns.heatmap(mean_Vt, cmap='viridis', cbar_kws={'label': 'Vt Value'})
    plt.title('Average Vt (Right Singular Vectors)')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    plt.tight_layout()
    plt.show()