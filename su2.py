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
        input_dim = feature_C + (feature_H * feature_C + feature_W * feature_C)  # 256 + 2048 + 2048 = 4352
        # output_dim = 2 * M * r + 2 * N * r
        output_dim = 2 * M * r  #(16384)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: 输入特征图, shape (B, feature_C, 8, 8)
        B = x.shape[0]

        # 提取三种特征
        global_vec = self.global_pool(x).view(B, -1)  # (B, 256)
        h_vec = self.h_pool(x).view(B, -1)  # (B, 2048)
        w_vec = self.w_pool(x).view(B, -1)  # (B, 2048)

        # 拼接所有特征
        hybrid_vec = torch.cat([global_vec, h_vec, w_vec], dim=1)  # (B, 4352)

        output_vec = self.mlp(hybrid_vec)

        p = 0
        U_r = output_vec[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        U_i = output_vec[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r

        U = torch.complex(U_r, U_i)
        U = F.normalize(U, p=2, dim=1)
        U = bjorck_orthogonalize(U, iterations=2)

        V = U.clone()

        return U, V

class GlobalDecoder(nn.Module):
    def __init__(self, in_channels, M, N, r, hidden_dim=512):
        super().__init__()
        self.r = r
        self.M = M
        self.N = N

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        output_dim = 2 * M * r
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        B = x.shape[0]

        global_vec = self.flatten(self.pool(x))
        output_vec = self.mlp(global_vec)

        p = 0
        U_r = output_vec[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        U_i = output_vec[:, p:p + self.M * self.r].reshape(B, self.M, self.r)


        U = torch.complex(U_r, U_i)
        U = F.normalize(U, p=2, dim=1)
        U = bjorck_orthogonalize(U, iterations=2)

        V = U.clone()

        return U, V


def bjorck_orthogonalize(matrix, iterations=2):
    """
    Applies the Björck iterative orthogonalization process.
    This function is differentiable and helps enforce orthogonality.
    """
    with torch.no_grad():  # The identity matrix should not require gradients
        I = torch.eye(matrix.size(2), device=matrix.device, dtype=matrix.dtype)

    for _ in range(iterations):
        # The core update rule
        matrix_T_matrix = matrix.conj().permute(0, 2, 1) @ matrix
        correction = 0.5 * (I - matrix_T_matrix)
        matrix = matrix @ (I + correction)
    return matrix

class SvdNet(nn.Module):
    def __init__(self, M=64, N=64, r=32):
        super().__init__()
        self.M, self.N, self.r = M, N, r

        self.spatial_branch = nn.Sequential(
            # nn.Conv2d(2, 16, kernel_size=3, padding=1, stride=2, bias=False),  # 128x128 -> 64x64
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            ResidualBlock(2, 16, stride=2), # 128x128 -> 64x64
            ResidualBlock(16, 32, stride=2)  # 64x64 -> 32x32
        )
        self.frequency_branch = nn.Sequential(
            # nn.Conv2d(2, 16, kernel_size=3, padding=1, stride=2, bias=False),  # 128x128 -> 64x64
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            ResidualBlock(2, 16, stride=2),  # 128x128 -> 64x64
            ResidualBlock(16, 32, stride=2)  # 64x64 -> 32x32
        )
        self.shared_encoder = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # 32x32 -> 16x16
            ResidualBlock(128, 256, stride=2),  # 16x16 -> 8x8
        )

        fc_input_size = 256
        self.decoder = GlobalDecoder(fc_input_size, M, N, r, hidden_dim=256)
        # self.decoder = AxisPoolDecoder(M, N, r, feature_C=fc_input_size, feature_H=8, feature_W=8, hidden_dim=256)

    def forward(self, x):
        # fea = self.enc(x)
        x_spatial = x[:, 0:2, :, :]
        x_freq = x[:, 2:4, :, :]
        fea_spatial = self.spatial_branch(x_spatial)
        fea_freq = self.frequency_branch(x_freq)
        fea_concat = torch.cat([fea_spatial, fea_freq], dim=1)  # (B, 32+32, 16, 16)
        fea = self.shared_encoder(fea_concat)

        U, V = self.decoder(fea)

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

def calculate_losses(U, S, V, H):
    # --- 重构损失分量 ---
    Sigma = torch.diag_embed(S.to(torch.complex64))
    H_hat = U @ Sigma @ V.conj().permute(0, 2, 1)
    num = torch.linalg.norm(H - H_hat, ord='fro', dim=(1, 2))
    denom = torch.linalg.norm(H, ord='fro', dim=(1, 2)).clamp_min(1e-8)
    recon_loss = (num / denom).mean()  # 对批次取平均

    # --- 正交损失分量 ---
    I_r = torch.eye(U.size(2), dtype=torch.complex64, device=H.device)
    err_u = torch.linalg.norm(U.conj().permute(0, 2, 1) @ U - I_r, ord='fro', dim=(1, 2))
    err_v = torch.linalg.norm(V.conj().permute(0, 2, 1) @ V - I_r, ord='fro', dim=(1, 2))
    ortho_loss = (err_u + err_v).mean()  # 对批次取平均

    return {
        'recon': recon_loss,
        'ortho': ortho_loss
    }

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
    for idx in range(1, 5):
        td = os.path.join(DATASET_DIR, f"Round2TrainData{idx}.npy")
        tl = os.path.join(DATASET_DIR, f"Round2TrainLabel{idx}.npy")
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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 2) 模型 & 优化器 & LR Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    M, Nmat, r = train_sets[0].M, train_sets[0].N, 64

    if weight_path:
        print(f"Loading model weights from {weight_path}")
        model = load_model(weight_path, M, Nmat, r).to(device)
    else:
        print("Initializing new model")
        model = SvdNet(M, Nmat, r).to(device)

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
        count = 0
        running_total, running_recon, running_ortho = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", ncols=80)
        if epoch % 2 != 0:
            # 奇数 epoch: 执行重构阶段 (Fit Phase)
            pbar.set_description(f"Epoch {epoch}/{NUM_EPOCHS} [Fit Phase]")
            current_phase = 'fit'
        else:
            # 偶数 epoch: 执行正交阶段 (Ortho Phase)
            pbar.set_description(f"Epoch {epoch}/{NUM_EPOCHS} [Ortho Phase]")
            current_phase = 'ortho'
        for x_raw, H in pbar:
            x_raw, H = x_raw.to(device), H.to(device)
            x_complex = torch.complex(x_raw[:, 0, :, :], x_raw[:, 1, :, :])
            fro_norm = torch.linalg.norm(x_complex, ord='fro', dim=(1, 2)) + 1e-8
            x_normalized_complex = x_complex / fro_norm.view(-1, 1, 1)
            x_fft_complex = torch.fft.fft2(x_normalized_complex, norm='ortho')

            x_out = torch.stack([
                x_normalized_complex.real,
                x_normalized_complex.imag,
                x_fft_complex.real,
                x_fft_complex.imag,
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

            loss_components = calculate_losses(U, S, V, H)
            l_rec = loss_components['recon']
            l_orth = loss_components['ortho']
            if current_phase == 'fit':
                loss = l_rec + 1e-3 * l_orth
            else:
                loss = l_orth

            # loss = ae_loss(U, S, V, H)
            loss.backward()
            optimizer.step()
            scheduler.step()

            b = x_normalized.size(0)
            running_total += loss.item() * b
            running_recon += l_rec.item() * b
            running_ortho += l_orth.item() * b
            count += b
            pbar.set_postfix(L_total=f"{loss.item():.4f}", L_rec=f"{l_rec.item():.4f}", L_orth=f"{l_orth.item():.4f}")

        train_loss_total = running_total / count
        train_loss_recon = running_recon / count
        train_loss_ortho = running_ortho / count

        # 验证
        model.eval()
        v_sum, v_cnt = 0.0, 0
        with torch.no_grad():
            for x_raw, H in val_loader:
                x_raw, H = x_raw.to(device), H.to(device)

                x_complex = torch.complex(x_raw[:, 0, :, :], x_raw[:, 1, :, :])
                fro_norm = torch.linalg.norm(x_complex, ord='fro', dim=(1, 2)) + 1e-8
                x_normalized_complex = x_complex / fro_norm.view(-1, 1, 1)

                x_fft_complex = torch.fft.fft2(x_normalized_complex, norm='ortho')

                x_model_input = torch.stack([
                    x_normalized_complex.real,
                    x_normalized_complex.imag,
                    x_fft_complex.real,
                    x_fft_complex.imag,
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
        # print(f"Epoch {epoch} done — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        print(f"Epoch {epoch} done — "
              f"train_loss: {train_loss_total:.4f}, "
              f"train_loss_recon: {train_loss_recon:.4f}, "
              f"train_loss_ortho: {train_loss_ortho:.4f}, "
              f"val_loss: {val_loss:.4f}")

        # 保存最优
        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), os.path.join(LOG_DIR, 'svd_best_multi.pth'))
            print("  → saved new best weights")

        with open(log_file, 'a') as f:
            f.write(
                # f"Epoch {epoch}: Train Loss : {train_loss:.6f} Val Loss: {val_loss:.6f}\n")
                f"Epoch {epoch}: "
                f"Train Loss Total: {train_loss_total:.6f}, "
                f"Train Loss Recon: {train_loss_recon:.6f}, "
                f"Train Loss Ortho: {train_loss_ortho:.6f}, "
                f"Val Loss: {val_loss:.6f}\n")

    print(f"Training complete, best val_loss = {best_loss:.4f}")


NUM_EPOCHS = 500
LEARNING_RATE = 3e-4
# LEARNING_RATE = 3e-3
BATCH_SIZE = 64
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_DIR = f"mode2/{timestamp}"
DATASET_DIR = "./CompetitionData2"

if __name__ == "__main__":
    # model_path = "./svd_best_multi.pth"
    model_path = None
    main(model_path)
