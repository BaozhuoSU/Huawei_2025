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

def visualize_channel_samples():
    root = "./CompetitionData2"
    test_sets = []

    NUM_EXAMPLES_TO_SHOW = 5

    for idx in range(1, 5):
        td = os.path.join(root, f"Round2TrainData{idx}.npy")
        if os.path.exists(td):
            test_sets.append(ChannelSvdDataset(td, label_path=None))
        else:
            print(f"Warning: {td} not found!")
    assert test_sets, "No test data found!"

    for i, dataset in enumerate(test_sets):
        print(f"Processing testData{i + 1}.npy...")
        # --- Modified: Loop only for the specified number of examples ---
        for idx in range(min(NUM_EXAMPLES_TO_SHOW, len(dataset))):
            print(f"  Displaying example #{idx + 1}")

            x, h = dataset[idx]  # Get (M, N) complex Tensor
            H = x.numpy()  # Convert to numpy array
            H_sparse = np.fft.ifft(np.fft.fft(H, axis=0, norm='ortho'), axis=1, norm='ortho')

            sns.set_style("whitegrid")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'Channel Sample #{idx + 1} Visualization', fontsize=16)

            im1 = ax1.imshow(np.abs(H), aspect='auto', cmap='viridis', interpolation='nearest')
            ax1.set_title("Original Channel Magnitude (Antenna-Frequency Domain)")
            ax1.set_xlabel("Frequency (Subcarrier) Index")
            ax1.set_ylabel("Antenna Index")
            fig.colorbar(im1, ax=ax1, label='Magnitude')

            # --- Plot 2: Sparse Representation ---
            # Using np.fft.fftshift to move the zero-frequency component to the center for better visualization
            im2 = ax2.imshow(np.abs(np.fft.fftshift(H_sparse)), aspect='auto', cmap='viridis', interpolation='nearest')
            ax2.set_title("Sparse Representation (Angular-Delay Domain)")
            ax2.set_xlabel("Delay Index (0 at center)")
            ax2.set_ylabel("Angular Index (0 at center)")
            fig.colorbar(im2, ax=ax2, label='Magnitude')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


def analyze_fourier_bases():
    DATA_PATH = "./CompetitionData2/Round2TrainData1.npy"
    LABEL_PATH = "./CompetitionData2/Round2TrainLabel1.npy"
    SAMPLE_IDX = 100  # 我们分析第100个样本
    M = 128
    N = 128
    r = 64

    # --- 加载数据 ---
    # H_label是我们需要分析的目标
    lab = np.load(LABEL_PATH, mmap_mode='r')
    H_label_complex = (lab[SAMPLE_IDX, ..., 0] + 1j * lab[SAMPLE_IDX, ..., 1]).astype(np.complex64)

    # --- 1. 计算真实的U矩阵 ---
    # 这是我们希望模型能预测出的目标
    U_true, S_true, Vh_true = np.linalg.svd(H_label_complex, full_matrices=False)
    U_true_r = U_true[:, :r]

    # --- 2. 构造傅里叶基矩阵 F_M ---
    F_M = np.fft.fft(np.eye(M), norm="ortho")

    # --- 3. 定量分析：用傅里叶基重构U_true_r的误差是多少？---
    # 这是所有基于傅里叶基的方案所能达到的理论性能上限
    # 我们将U_true_r投影到由所有M个傅里叶基张成的空间中（这应该无损）
    # 然后看只用最好的r个傅里叶基能恢复多少
    # (为了简单，我们直接用U_true_r去乘以F_M的共轭转置，找到最重要的r个基)
    coeffs = F_M.conj().T @ U_true_r  # (M, M) @ (M, r) -> (M, r)
    # 找到每一列U最重要的r个傅里叶基分量
    energy_per_base = np.sum(np.abs(coeffs) ** 2, axis=1)  # (M,)
    top_r_indices = np.argsort(energy_per_base)[-r:]  # 找到能量最强的r个基的索引

    # 只用这r个基来重构U
    F_M_top_r = F_M[:, top_r_indices]  # (M, r)
    # C = F_M_top_r.T @ U_true_r, U_recon = F_M_top_r @ C
    U_recon = F_M_top_r @ (F_M_top_r.conj().T @ U_true_r)

    # 计算归一化误差
    recon_error = np.linalg.norm(U_true_r - U_recon) / np.linalg.norm(U_true_r)
    print(f"【核心指标】使用最优的 {r} 个傅里叶基重构真实U矩阵的归一化误差: {recon_error:.4f}")

    # --- 4. 可视化对比 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 对比 U_true 的第一列 和 最相关的傅里叶基
    axes[0].plot(np.abs(U_true_r[:, 0]), label='|True U Column 0|')
    axes[0].plot(np.abs(F_M[:, np.argmax(np.abs(coeffs[:, 0]))]), label='|Best Fit Fourier Base|', linestyle='--')
    axes[0].set_title('Compare True U_col_0 with Best Fourier Base')
    axes[0].legend()

    # 可视化真实U矩阵的能量谱（在傅里叶域）
    axes[1].stem(energy_per_base)
    axes[1].set_title('Energy Spectrum of True U on Fourier Bases')
    axes[1].set_xlabel('Fourier Base Index')
    axes[1].set_ylabel('Energy')

    # 可视化信道矩阵H的频域稀疏性
    H_freq = np.fft.fft2(H_label_complex, norm="ortho")
    im = axes[2].imshow(np.abs(H_freq))
    axes[2].set_title('Sparsity of H in Frequency Domain')
    fig.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_channel_samples()



