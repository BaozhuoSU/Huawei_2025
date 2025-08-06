# train_svd_net.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F


# =============================================================
# 定义：ChannelSvdDataset
# ———— 使用 mmap_mode 快速加载 .npy，不再 astype
# =============================================================
class ChannelSvdDataset(Dataset):
    def __init__(self, data_path, label_path=None):
        raw = np.load(data_path, mmap_mode='r')  # float32 (Ns, M, N, 2)
        self.M, self.N = raw.shape[1], raw.shape[2]
        # 转成 (Ns, 2, M, N)
        self.X = raw.transpose(0, 3, 1, 2)
        # 如果给出了 label_path，再加载 H
        if label_path is not None:
            lab = np.load(label_path, mmap_mode='r')
            # 若 shape==(Ns,M,N,2) 则合成 complex
            if lab.ndim == 4 and lab.shape[-1] == 2:
                H_real = lab[..., 0]
                H_imag = lab[..., 1]
                self.H = (H_real + 1j * H_imag).astype(np.complex64)
            else:
                # 已经是 (Ns, M, N) complex 存储
                self.H = lab
        else:
            self.H = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # (2, M, N)
        x_numpy = self.X[idx]  # (2, M, N)
        x_complex = torch.complex(torch.from_numpy(x_numpy[0]), torch.from_numpy(x_numpy[1]))
        fro_norm = torch.linalg.norm(x_complex, ord='fro') + 1e-8
        x_normalized = x_complex / fro_norm
        x_normalized = torch.stack([x_normalized.real, x_normalized.imag], dim=0).float()

        if self.H is not None:
            h = self.H[idx]  # complex64 (M, N)
            h = torch.from_numpy(h)
            # h_fro_norm = torch.linalg.norm(h, ord='fro') + 1e-8
            # h_normalized = h / h_fro_norm
            return x_normalized, h, fro_norm
        else:
            return x_normalized, fro_norm


class SimpleChannelDataset(Dataset):
    def __init__(self, data_path, label_path=None):
        self.X = np.load(data_path, mmap_mode='r').transpose(0, 3, 1, 2)
        self.M, self.N = self.X.shape[2], self.X.shape[3]

        if label_path is not None:
            lab = np.load(label_path, mmap_mode='r')
            if lab.ndim == 4 and lab.shape[-1] == 2:
                self.H = (lab[..., 0] + 1j * lab[..., 1]).astype(np.complex64)
            else:
                self.H = lab
        else:
            self.H = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # (2, M, N)
        if self.H is not None:
            h = torch.from_numpy(self.H[idx])  # (M, N) complex64
            return x, h
        else:
            return x


class AugmentationWrapper(Dataset):
    """
    一个通用的数据增强包装器。
    它接收一个现有的数据集对象，并在__getitem__时动态应用增强。
    """

    def __init__(self,
                 base_dataset: Dataset,
                 p_shift: float = 0.5,
                 p_scale: float = 0.5,
                 scale_range_db: tuple = (-10, 10),
                 p_noise: float = 0.5,
                 noise_std: float = 0.01,):

        self.base_dataset = base_dataset
        self.p_shift = p_shift
        self.p_scale = p_scale
        self.scale_range_db = scale_range_db
        self.p_noise = p_noise
        self.noise_std = noise_std

        self.M, self.N = 64, 64

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        # 1. 从原始数据集中获取未经增强的数据
        # base_dataset 返回的是 (x, h) 元组
        x, h = self.base_dataset[idx]

        # --- 随机循环移位 ---
        if torch.rand(1).item() < self.p_shift:
            shift_h = torch.randint(0, self.M, (1,)).item()
            shift_w = torch.randint(0, self.N, (1,)).item()
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))
            if h is not None:
                h = torch.roll(h, shifts=(shift_h, shift_w), dims=(0, 1))

        # --- 随机功率缩放 ---
        if torch.rand(1).item() < self.p_scale:
            x_complex = torch.complex(x[0], x[1])
            log_min = self.scale_range_db[0] / 10.0
            log_max = self.scale_range_db[1] / 10.0
            power_scale = 10 ** (torch.rand(1).item() * (log_max - log_min) + log_min)
            amplitude_scale = power_scale ** 0.5
            x_complex = x_complex * amplitude_scale
            x = torch.stack([x_complex.real, x_complex.imag]).float()
            if h is not None:
                h = h * amplitude_scale

        # --- 随机高斯噪声注入 ---
        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x, h


# =============================================================
# 定义：SvdNet 模型
# ———— 轻量级 Conv→GAP→FC → 输出 U, V
# =============================================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=4, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        channel_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.relu(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.path = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, stride=stride),
            CBAM(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 残差连接
        return nn.functional.relu(self.shortcut(x) + self.path(x))


class GlobalDecoder(nn.Module):
    def __init__(self, in_channels, M, N, r, hidden_dim=512):
        super().__init__()
        self.r = r
        self.M = M
        self.N = N

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        output_dim = 2 * M * r + 2 * N * r  # 8192
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: 输入特征图, shape (B, C, H, W), e.g., (B, 32, 8, 8)
        B = x.shape[0]

        global_vec = self.flatten(self.pool(x))  # (B, C)
        output_vec = self.mlp(global_vec)  # (B, 8192)

        p = 0
        U_r = output_vec[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        U_i = output_vec[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        V_r = output_vec[:, p:p + self.N * self.r].reshape(B, self.N, self.r)
        p += self.N * self.r
        V_i = output_vec[:, p:p + self.N * self.r].reshape(B, self.N, self.r)

        U = torch.complex(U_r, U_i)
        V = torch.complex(V_r, V_i)

        return U, V


class AxisPoolDecoder(nn.Module):
    def __init__(self, M, N, r, feature_C=128, feature_H=8, feature_W=8, hidden_dim=512):
        super().__init__()
        self.r, self.M, self.N = r, M, N

        # 全局池化分支
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 轴向池化分支
        self.h_pool = nn.AdaptiveAvgPool2d((feature_H, 1))
        self.w_pool = nn.AdaptiveAvgPool2d((1, feature_W))

        # 输入维度 = 全局特征(C) + 轴向特征(H*C + W*C)
        input_dim = feature_C + (feature_H * feature_C + feature_W * feature_C)  # 128 + 2048 = 2176
        output_dim = 2 * M * r + 2 * N * r

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: 输入特征图, shape (B, 128, 8, 8)
        B = x.shape[0]

        # 提取三种特征
        global_vec = self.global_pool(x).view(B, -1)  # (B, 128)
        h_vec = self.h_pool(x).view(B, -1)  # (B, 1024)
        w_vec = self.w_pool(x).view(B, -1)  # (B, 1024)

        # 拼接所有特征
        hybrid_vec = torch.cat([global_vec, h_vec, w_vec], dim=1)  # (B, 2176)

        output_vec = self.mlp(hybrid_vec)

        p = 0
        U_r = output_vec[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        U_i = output_vec[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        V_r = output_vec[:, p:p + self.N * self.r].reshape(B, self.N, self.r)
        p += self.N * self.r
        V_i = output_vec[:, p:p + self.N * self.r].reshape(B, self.N, self.r)

        U = torch.complex(U_r, U_i)
        V = torch.complex(V_r, V_i)

        return U, V

class SvdNet(nn.Module):
    def __init__(self, M=64, N=64, r=32):
        super().__init__()
        self.M, self.N, self.r = M, N, r

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, stride=2, bias=False),  # 64x64 -> 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16, 32, stride=2)  # 32x32 -> 16x16
        )
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, stride=2, bias=False),  # 64x64 -> 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16, 32, stride=2)  # 32x32 -> 16x16
        )
        self.shared_encoder = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # 16x16 -> 8x8
            ResidualBlock(128, 128, stride=1),
            CBAM(128)
        )

        fc_input_size = 128
        # self.decoder = GlobalDecoder(fc_input_size, M, N, r, hidden_dim=400)
        self.decoder = AxisPoolDecoder(M, N, r, feature_C=fc_input_size, feature_H=8, feature_W=8, hidden_dim=256)

    def forward(self, x):
        # fea = self.enc(x)
        x_spatial = x[:, 0:4, :, :]
        x_freq = x[:, 4:8, :, :]
        fea_spatial = self.spatial_branch(x_spatial)
        fea_freq = self.frequency_branch(x_freq)
        fea_concat = torch.cat([fea_spatial, fea_freq], dim=1)  # (B, 32+32, 16, 16)
        fea = self.shared_encoder(fea_concat)

        U, V = self.decoder(fea)

        U = F.normalize(U, p=2, dim=1)
        V = F.normalize(V, p=2, dim=1)
        return U, V


# =============================================================
# 定义：自监督 AE Loss
# L_AE = ||H - UΣV^H||_F/||H||_F  +  ||U^H U - I||_F  +  ||V^H V - I||_F
# =============================================================
def analytic_sigma(U, V, H):
    # 计算 Σ = diag( U^H H V )
    # U:(B,M,r), V:(B,N,r), H:(B,M,N)
    raw_S = (U.conj().permute(0, 2, 1) @ H @ V) \
        .diagonal(dim1=1, dim2=2).real  # (B, r)

    S = torch.clamp(raw_S, min=0.0)
    return S  # (B, r)


def ae_loss(U, S, V, H, lam=1.0):
    B = H.size(0)
    # 构造 Σ 张量
    Sigma = torch.zeros(B, S.size(1), S.size(1),
                        dtype=torch.complex64, device=H.device)
    Sigma[:, torch.arange(S.size(1)), torch.arange(S.size(1))] = S.type(torch.complex64)
    # 重构
    H_hat = U @ Sigma @ V.conj().permute(0, 2, 1)
    # 重构误差
    num = torch.linalg.norm(H - H_hat, ord='fro', dim=(1, 2))
    denom = torch.linalg.norm(H, ord='fro', dim=(1, 2)).clamp_min(1e-8)
    recon = num / denom
    # 正交惩罚
    I_r = torch.eye(S.size(1), dtype=torch.complex64, device=H.device)
    UU = U.conj().permute(0, 2, 1) @ U
    VV = V.conj().permute(0, 2, 1) @ V
    err_u = torch.linalg.norm(UU - I_r, ord='fro', dim=(1, 2))
    err_v = torch.linalg.norm(VV - I_r, ord='fro', dim=(1, 2))
    return (recon + lam * (err_u + err_v)).mean()


def load_model(weight_path: str, M: int, N: int, r: int) -> nn.Module:
    model = SvdNet(M, N, r)
    state_dict = torch.load(weight_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict)
    return model


# =============================================================
# 主函数：加载 data2/data3/data4，训练 & 验证
# =============================================================
def main(weight_path=None):
    # 1) 准备数据集
    # ROOTS = [Path.cwd() / f"data{i}" for i in (1, 2, 3)]
    train_sets = []
    for idx in range(1, 4):
        td = os.path.join(DATASET_DIR, f"Round1TrainData{idx}.npy")
        tl = os.path.join(DATASET_DIR, f"Round1TrainLabel{idx}.npy")
        train_sets.append(SimpleChannelDataset(td, tl))
    assert train_sets, "No training data found!"

    full_ds = ConcatDataset(train_sets)
    N = len(full_ds)
    n_train = int(0.9 * N)
    n_val = N - n_train

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # train_ds = AugmentationWrapper(
    #     base_dataset=train_ds,
    #     p_shift=0.3,
    #     p_scale=0.3,
    #     p_noise=0.3,
    # )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 2) 模型 & 优化器 & LR Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    M, Nmat, r = train_sets[0].M, train_sets[0].N, 32

    if weight_path:
        print(f"Loading model weights from {weight_path}")
        model = load_model(weight_path, M, Nmat, r).to(device)
    else:
        print("Initializing new model")
        model = SvdNet(M, Nmat, r).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                        T_max=NUM_EPOCHS,
    #                                                        eta_min=1e-6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(
            f"Batch sizes: Train={BATCH_SIZE}\n Learning rate: {LEARNING_RATE}\n Epochs: {NUM_EPOCHS}\n")

    # 3) 训练循环
    best_loss, patience = float('inf'), 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running, count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", ncols=80)
        for x_raw, H in pbar:
            # x_norm, H, fro_norm = x_norm.to(device), H.to(device), fro_norm.to(device)

            x_raw, H = x_raw.to(device), H.to(device)
            x_complex = torch.complex(x_raw[:, 0, :, :], x_raw[:, 1, :, :])
            fro_norm = torch.linalg.norm(x_complex, ord='fro', dim=(1, 2)) + 1e-8
            x_normalized_complex = x_complex / fro_norm.view(-1, 1, 1)
            x_fft_complex = torch.fft.fft2(x_normalized_complex, norm='ortho')

            x_mag = torch.abs(x_normalized_complex)
            x_log_power = torch.log(x_mag ** 2 + 1e-8)
            x_fft_mag = torch.abs(x_fft_complex)
            x_fft_log_power = torch.log(x_fft_mag ** 2 + 1e-8)

            x_out = torch.stack([
                x_normalized_complex.real,
                x_normalized_complex.imag,
                x_mag,
                x_log_power,
                x_fft_complex.real,
                x_fft_complex.imag,
                x_fft_mag,
                x_fft_log_power
            ], dim=1).float()
            x_normalized = torch.stack([
                x_normalized_complex.real,
                x_normalized_complex.imag
            ], dim=1).float()

            optimizer.zero_grad()
            U, V = model(x_out)
            H_normalized = H / fro_norm.view(-1, 1, 1)
            S_norm = analytic_sigma(U, V, H_normalized)
            S = S_norm * fro_norm.unsqueeze(1)  # 恢复原始范数
            loss = ae_loss(U, S, V, H)
            # loss = supervised_ae_loss(U_ortho, V_ortho, S, H, U_raw, V_raw, lam=0.1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            b = x_normalized.size(0)
            running += loss.item() * b
            count += b
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / count

        # 验证
        model.eval()
        v_sum, v_cnt = 0.0, 0
        with torch.no_grad():
            # for x_norm, H, fro_norm, x_fft in val_loader:
            #     x_norm, H, fro_norm, x_fft = x_norm.to(device), H.to(device), fro_norm.to(device), x_fft.to(device)
            #     U, V = model(x_fft)
            #     S_norm = analytic_sigma(U, V, torch.view_as_complex(x_norm.permute(0, 2, 3, 1).contiguous()))
            #     S = S_norm * fro_norm.unsqueeze(1)
            #     l = ae_loss(U, S, V, H)
            #     b = x_norm.size(0)
            #     v_sum += l.item() * b
            #     v_cnt += b
            for x_raw, H in val_loader:
                x_raw, H = x_raw.to(device), H.to(device)

                x_complex = torch.complex(x_raw[:, 0, :, :], x_raw[:, 1, :, :])
                fro_norm = torch.linalg.norm(x_complex, ord='fro', dim=(1, 2)) + 1e-8
                x_normalized_complex = x_complex / fro_norm.view(-1, 1, 1)

                x_fft_complex = torch.fft.fft2(x_normalized_complex, norm='ortho')

                x_mag = torch.abs(x_normalized_complex)
                x_log_power = torch.log(x_mag ** 2 + 1e-8)
                x_fft_mag = torch.abs(x_fft_complex)
                x_fft_log_power = torch.log(x_fft_mag ** 2 + 1e-8)

                x_model_input = torch.stack([
                    x_normalized_complex.real,
                    x_normalized_complex.imag,
                    x_mag,
                    x_log_power,
                    x_fft_complex.real,
                    x_fft_complex.imag,
                    x_fft_mag,
                    x_fft_log_power
                ], dim=1).float()

                U, V = model(x_model_input)
                H_normalized = H / fro_norm.view(-1, 1, 1)
                S_norm = analytic_sigma(U, V, H_normalized)
                S = S_norm * fro_norm.unsqueeze(1)
                l = ae_loss(U, S, V, H)
                b = x_raw.size(0)
                v_sum += l.item() * b
                v_cnt += b

        val_loss = v_sum / v_cnt

        # scheduler.step(val_loss)
        print(f"Epoch {epoch} done — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # 保存最优
        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), os.path.join(LOG_DIR, 'svd_best_multi.pth'))
            print("  → saved new best weights")

        with open(log_file, 'a') as f:
            f.write(
                f"Epoch {epoch}: Train Loss : {train_loss:.6f} Val Loss: {val_loss:.6f}\n")

    print(f"Training complete, best val_loss = {best_loss:.4f}")


NUM_EPOCHS = 500
# LEARNING_RATE = 3e-4
LEARNING_RATE = 3e-3
BATCH_SIZE = 64
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_DIR = f"model/{timestamp}"
DATASET_DIR = "./CompetitionData1"

if __name__ == "__main__":
    # model_path = "./svd_best_multi.pth"
    model_path = None
    main(model_path)
