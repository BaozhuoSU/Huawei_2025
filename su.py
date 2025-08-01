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
        # x = torch.from_numpy(self.X[idx]).float()  # (2, M, N)

        x_numpy = self.X[idx] # (2, M, N)
        h_complex = torch.complex(torch.from_numpy(x_numpy[0]), torch.from_numpy(x_numpy[1]))
        fro_norm = torch.linalg.norm(h_complex, ord='fro') + 1e-8
        h_normalized = h_complex / fro_norm
        x = torch.stack([h_normalized.real, h_normalized.imag], dim=0).float()

        if self.H is not None:
            h = self.H[idx]  # complex64 (M, N)
            return x, torch.from_numpy(h)
        else:
            return x


# =============================================================
# 定义：SvdNet 模型
# ———— 轻量级 Conv→GAP→FC → 输出 U, V
# =============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        out = self.elu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return self.elu(out)

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
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.relu(out)


class SvdNet(nn.Module):
    def __init__(self, M=64, N=64, r=32):
        super().__init__()
        self.M, self.N, self.r = M, N, r
        # encoder backbone
        # self.enc = nn.Sequential(
        #     nn.Conv2d(2, 8, 3, 1, 1),
        #     nn.ELU(),
        #     # ResidualBlock(8),
        #     CBAM(8),
        #
        #     nn.Conv2d(8,16, 3, 1, 1),
        #     nn.ELU(),
        #     # ResidualBlock(16),
        #     CBAM(16),
        #
        #     nn.MaxPool2d(2),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten()
        # )
        #
        # # predictor FC: [U_real, U_imag, V_real, V_imag]
        # self.fc = nn.Sequential(
        #     nn.Linear(16, 32),
        #     nn.ELU(),
        #     nn.Linear(32, 2*M*r + 2*N*r)
        # )
        # # 初始化 Σ bias>0 （前 r 个偏置位置）
        # with torch.no_grad():
        #     self.fc[-1].bias[:r].fill_(0.5)

        self.enc = nn.Sequential(
            # 初始层: 64x64 -> 32x32
            nn.Conv2d(2, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            DepthwiseSeparableConv(16, 32, stride=2),
            CBAM(32),
            DepthwiseSeparableConv(32, 64, stride=2),
            CBAM(64),
            DepthwiseSeparableConv(64, 128, stride=1),
            CBAM(128),

            # 全局平均池化
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 轻量级的 Predictor FC (保持不变)
        fc_input_size = 128
        output_size = 2 * M * r + 2 * N * r
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        # x: (B,2,M,N)
        B = x.size(0)
        y = self.fc(self.enc(x))  # (B, 2*M*r + 2*N*r)
        p = 0
        U_r = y[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        U_i = y[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        V_r = y[:, p:p + self.N * self.r].reshape(B, self.N, self.r)
        p += self.N * self.r
        V_i = y[:, p:p + self.N * self.r].reshape(B, self.N, self.r)

        U = torch.complex(U_r, U_i)  # (B, M, r)
        V = torch.complex(V_r, V_i)  # (B, N, r)
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
    root = "./CompetitionData1"
    train_sets = []
    for idx in range(1, 4):
        td = os.path.join(root, f"Round1TrainData{idx}.npy")
        tl = os.path.join(root, f"Round1TrainLabel{idx}.npy")
        train_sets.append(ChannelSvdDataset(td, tl))
    assert train_sets, "No training data found!"

    full_ds = ConcatDataset(train_sets)
    N = len(full_ds)
    n_train = int(0.9 * N)
    n_val = N - n_train

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # 2) 模型 & 优化器 & LR Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    M, Nmat, r = train_sets[0].M, train_sets[0].N, 32

    if weight_path:
        print(f"Loading model weights from {weight_path}")
        model = load_model(weight_path, M, Nmat, r).to(device)
    else:
        print("Initializing new model")
        model = SvdNet(M, Nmat, r).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=NUM_EPOCHS,
                                                           eta_min=1e-6)

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
        for x, H in pbar:
            x, H = x.to(device), H.to(device)
            optimizer.zero_grad()
            U, V = model(x)
            S = analytic_sigma(U, V, H)
            loss = ae_loss(U, S, V, H)
            loss.backward()
            optimizer.step()

            b = x.size(0)
            running += loss.item() * b
            count += b
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / count

        # 验证
        model.eval()
        v_sum, v_cnt = 0.0, 0
        with torch.no_grad():
            for x, H in val_loader:
                x, H = x.to(device), H.to(device)
                U, V = model(x)
                S = analytic_sigma(U, V, H)
                l = ae_loss(U, S, V, H)
                b = x.size(0)
                v_sum += l.item() * b
                v_cnt += b

        val_loss = v_sum / v_cnt

        # scheduler.step(val_loss)
        print(f"Epoch {epoch} done — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # 保存最优
        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), os.path.join(LOG_DIR,'svd_best_multi.pth'))
            print("  → saved new best weights")

        with open(log_file, 'a') as f:
            f.write(
                f"Epoch {epoch+1}: Train Loss : {train_loss:.6f} Val Loss: {val_loss:.6f}\n")

    print(f"Training complete, best val_loss = {best_loss:.4f}")


NUM_EPOCHS = 300
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_DIR = f"model/{timestamp}"

if __name__ == "__main__":
    model_path = "./svd_best_multi.pth"
    main(model_path)
