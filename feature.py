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

    # --- 新增：设置显示样本数量 ---
    NUM_EXAMPLES_TO_SHOW = 3

    # 加载测试数据
    for idx in range(1, 2):
        td = os.path.join(root, f"Round1TrainData{idx}.npy")
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

            # Get a sample from the dataset. We only need the label 'h' for visualization.
            _, h = dataset[idx]  # Get (M, N) complex Tensor
            H = h.numpy()  # Convert to numpy array

            # 2. Perform the 2D transform suggested by the hint to get the sparse representation
            # Antenna domain -> Angular domain (FFT on axis=0)
            # Frequency domain -> Delay domain (IFFT on axis=1)
            # norm='ortho' option preserves energy
            H_sparse = np.fft.ifft(np.fft.fft(H, axis=0, norm='ortho'), axis=1, norm='ortho')

            # 3. Create and display the plots
            sns.set_style("whitegrid")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'Channel Sample #{idx + 1} Visualization', fontsize=16)

            # --- Plot 1: Original Channel Magnitude ---
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


