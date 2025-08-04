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


# =============================================================
# 定义：SvdNet 模型
# ———— 轻量级 Conv→GAP→FC → 输出 U, V
# =============================================================
def qr_orthonormalize(x_raw):
    """
    使用 QR 分解来使矩阵列正交。
    x_raw: 神经网络输出的原始矩阵, shape (B, M, r)
    """
    # mode='reduced' 会返回一个 (M, r) 的 Q 矩阵
    Q, _ = torch.linalg.qr(x_raw, mode='reduced')
    return Q


def calculate_rms_delay_spread(H_complex, sample_time=1.0):
    """
    计算一批MIMO信道矩阵的RMS延迟扩展。

    参数:
    H_complex (torch.Tensor): 复数信道矩阵, shape (B, M, N)。假设N是频域/子载波维度。
    sample_time (float): 采样时间或时延单元的间隔，用于给延迟扩展一个物理尺度。默认为1.0。

    返回:
    torch.Tensor: RMS延迟扩展, shape (B,)
    """
    # 1. 沿频域维度(最后一维)做IFFT，得到时域信道冲激响应(CIR)
    cir = torch.fft.ifft(H_complex, dim=-1)

    # 2. 计算功率延迟分布 (PDP)，即CIR模长的平方，并在天线维度上取平均
    pdp = torch.abs(cir) ** 2
    pdp = torch.mean(pdp, dim=1)  # Shape: (B, N)

    # 确保PDP非负且总和不为零
    pdp = torch.clamp(pdp, min=0)
    pdp_sum = torch.sum(pdp, dim=1, keepdim=True) + 1e-8

    # 3. 计算平均延迟 (一阶矩)
    # 创建时间/延迟轴
    delay_taps = torch.arange(H_complex.shape[-1], device=H_complex.device, dtype=torch.float32) * sample_time
    mean_delay = torch.sum(pdp * delay_taps, dim=1, keepdim=True) / pdp_sum

    # 4. 计算平均延迟的平方 (二阶矩)
    mean_square_delay = torch.sum(pdp * (delay_taps ** 2), dim=1, keepdim=True) / pdp_sum

    # 5. 计算RMS延迟扩展 (标准差)
    rms_delay_spread = torch.sqrt(torch.clamp(mean_square_delay - mean_delay ** 2, min=0)).squeeze(-1)

    return rms_delay_spread


class FourierFeatureBlock(nn.Module):
    def __init__(self, in_channels, h, w):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(in_channels, h, w, dtype=torch.cfloat))

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # This logic for creating a complex tensor is correct and stays the same
        if x.shape[1] == 2:
            x_complex = torch.complex(x[:, 0, :, :], x[:, 1, :, :]).unsqueeze(1)
        else:
            b, c, h, w = x.shape
            x_complex = torch.complex(x[:, 0:c:2, :, :], x[:, 1:c:2, :, :])

        x_fft = torch.fft.fft2(x_complex, norm='ortho')
        x_filtered = torch.einsum("bchw,chw->bchw", x_fft, self.weights)
        x_ifft = torch.fft.ifft2(x_filtered, s=(x_complex.shape[-2], x_complex.shape[-1]), norm='ortho')

        x_out_real = x_ifft.real
        x_out_imag = x_ifft.imag
        x_out = torch.cat([x_out_real, x_out_imag], dim=1)

        # The residual connection is correct and stays the same
        return x + self.alpha * x_out


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


class GatedConv1d(nn.Module):
    """
    实现门控线性单元 (GLU) 的一维卷积模块。
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv_data = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        # 计算数据通路和门控通路
        data = self.conv_data(x)
        gate = torch.sigmoid(self.conv_gate(x))
        # 逐元素相乘
        return data * gate


class ConvDecoderHead(nn.Module):
    def __init__(self, in_channels, M, N, r, dropout_rate=0.1):
        super().__init__()
        self.M, self.N, self.r = M, N, r

        # 使用GLU增强的卷积模块
        self.gated_conv_block = nn.Sequential(
            GatedConv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels, 4 * r, kernel_size=1)
        )

    def forward(self, x):
        # 这部分逻辑和 UnifiedConvDecoderHead 完全相同
        B, C, H, W = x.shape
        sequence_len = H * W
        x_seq = x.view(B, C, sequence_len)

        # 使用增强的卷积块
        processed_seq = self.gated_conv_block(x_seq)

        y = processed_seq.permute(0, 2, 1)
        y_u_raw, y_v_raw = torch.split(y, [2 * self.r, 2 * self.r], dim=-1)

        u = y_u_raw.view(B, self.M, self.r, 2)
        U_complex = torch.complex(u[..., 0], u[..., 1])
        v = y_v_raw.view(B, self.N, self.r, 2)
        V_complex = torch.complex(v[..., 0], v[..., 1])

        return U_complex, V_complex


class SingleMatrixHead(nn.Module):
    def __init__(self, in_channels, matrix_dim, r, dropout_rate=0.1):
        super().__init__()
        self.matrix_dim, self.r = matrix_dim, r
        self.sequence_len = matrix_dim
        self.gated_conv_block = nn.Sequential(
            GatedConv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels, 2 * r, kernel_size=1)
        )

    def forward(self, x):
        B = x.shape[0]
        x_seq = x.view(B, -1, self.sequence_len)
        processed_seq = self.gated_conv_block(x_seq)
        y = processed_seq.permute(0, 2, 1)
        y = y.view(B, self.matrix_dim, self.r, 2)
        y_complex = torch.complex(y[..., 0], y[..., 1])
        return y_complex


class SvdNet(nn.Module):
    def __init__(self, M=64, N=64, r=32):
        super().__init__()
        self.M, self.N, self.r = M, N, r

        self.enc = nn.Sequential(
            FourierFeatureBlock(1, h=64, w=64),  # 2 channels = 1 complex pairs
            # (2, 64, 64) -> (16, 32, 32)
            nn.Conv2d(2, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            FourierFeatureBlock(8, h=32, w=32),  # 16 channels = 8 complex pairs
            CBAM(16),
            # (16, 32, 32) -> (32, 16, 16)
            DepthwiseSeparableConv(16, 32, stride=2),
            FourierFeatureBlock(16, h=16, w=16),  # 32 channels = 16 complex pairs
            CBAM(32),
            # (32, 16, 16) -> (64, 8, 8)
            DepthwiseSeparableConv(32, 64, stride=2),
            FourierFeatureBlock(32, h=8, w=8),  # 64 channels = 32 complex pairs
            CBAM(64),
            # (64, 8, 8) -> (128, 8, 8)
            DepthwiseSeparableConv(64, 128, stride=1),
            CBAM(128),

        )

        fc_input_size = 128

        self.conditioning_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * fc_input_size)  # 输出 2*128=256
        )
        # 获取最后一层线性层
        final_layer = self.conditioning_net[-1]
        # 将其权重初始化为0
        torch.nn.init.zeros_(final_layer.weight)
        initial_bias = torch.zeros(2 * fc_input_size)
        initial_bias[:fc_input_size] = 1.0  # 设置gamma的偏置为1
        final_layer.bias.data = initial_bias

        # self.multi_head = ConvDecoderHead(fc_input_size, M, N, r)
        self.v_head = SingleMatrixHead(fc_input_size, N, r)
        self.u_head = SingleMatrixHead(fc_input_size, M, r)

        # self.cross_mod_net = nn.Sequential(
        #     nn.Linear(r * 2, 64),  # 输入V的摘要，r=32 -> 输入维度64
        #     nn.ReLU(),
        #     nn.Linear(64, 2 * fc_input_size)  # 输出gamma和beta，维度2*128=256
        # )

    def forward(self, x):
        H_complex = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        delay_spread = calculate_rms_delay_spread(H_complex)  # Shape: (B,)

        fea = self.enc(x)

        modulation_params = self.conditioning_net(delay_spread.unsqueeze(1))  # (B, 256)
        #    b. 将参数切分为 gamma (用于缩放) 和 beta (用于平移)
        gamma, beta = torch.chunk(modulation_params, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        fea = fea * gamma + beta

        V_raw = self.v_head(fea)
        # v_summary_complex = V_raw.mean(dim=1)
        # v_summary_real = torch.cat([v_summary_complex.real, v_summary_complex.imag], dim=1)  # Shape: (B, r*2)
        # cross_mod_params = self.cross_mod_net(v_summary_real)
        # cross_gamma, cross_beta = torch.chunk(cross_mod_params, 2, dim=1)
        # fea_for_u = fea * cross_gamma.unsqueeze(-1).unsqueeze(-1) + cross_beta.unsqueeze(-1).unsqueeze(-1)
        U_raw = self.u_head(fea)

        U_ortho = qr_orthonormalize(U_raw)
        V_ortho = qr_orthonormalize(V_raw)
        return U_ortho, V_ortho, U_raw, V_raw


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


def supervised_ae_loss(U_ortho, V_ortho, S, H, U_raw, V_raw, lam=0.1):
    B = H.size(0)

    Sigma = torch.zeros(B, S.size(1), S.size(1),
                        dtype=torch.complex64, device=H.device)
    Sigma[:, torch.arange(S.size(1)), torch.arange(S.size(1))] = S.type(torch.complex64)
    H_hat = U_ortho @ Sigma @ V_ortho.conj().permute(0, 2, 1)

    num = torch.linalg.norm(H - H_hat, ord='fro', dim=(1, 2))
    denom = torch.linalg.norm(H, ord='fro', dim=(1, 2)).clamp_min(1e-8)
    recon_loss = num / denom

    r = U_raw.shape[-1]
    I_r = torch.eye(r, dtype=torch.complex64, device=H.device).expand(B, r, r)

    err_u = torch.linalg.norm(U_raw.conj().permute(0, 2, 1) @ U_raw - I_r, ord='fro', dim=(1, 2))

    err_v = torch.linalg.norm(V_raw.conj().permute(0, 2, 1) @ V_raw - I_r, ord='fro', dim=(1, 2))

    # Combine the losses. The lambda value weights the importance of orthogonality.
    total_loss = recon_loss.mean() + lam * (err_u.mean() + err_v.mean())

    return total_loss


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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=NUM_EPOCHS,
                                                           eta_min=1e-6)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=LEARNING_RATE * 0.01)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.01)

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
        for x_norm, H, fro_norm in pbar:
            x_norm, H, fro_norm = x_norm.to(device), H.to(device), fro_norm.to(device)
            optimizer.zero_grad()
            U_ortho, V_ortho, U_raw, V_raw = model(x_norm)
            S_norm = analytic_sigma(U_ortho, V_ortho, torch.view_as_complex(x_norm.permute(0, 2, 3, 1).contiguous()))
            S = S_norm * fro_norm.unsqueeze(1)  # 恢复原始范数
            loss = ae_loss(U_ortho, S, V_ortho, H)
            # loss = supervised_ae_loss(U_ortho, V_ortho, S, H, U_raw, V_raw, lam=0.1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            b = x_norm.size(0)
            running += loss.item() * b
            count += b
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / count

        # 验证
        model.eval()
        v_sum, v_cnt = 0.0, 0
        with torch.no_grad():
            for x_norm, H, fro_norm in val_loader:
                x_norm, H, fro_norm = x_norm.to(device), H.to(device), fro_norm.to(device)
                U, V, _, _ = model(x_norm)
                S_norm = analytic_sigma(U, V, torch.view_as_complex(x_norm.permute(0, 2, 3, 1).contiguous()))
                S = S_norm * fro_norm.unsqueeze(1)
                l = ae_loss(U, S, V, H)
                b = x_norm.size(0)
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
