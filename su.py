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
        # x_numpy = self.X[idx] # (2, M, N)
        # x_complex = torch.complex(torch.from_numpy(x_numpy[0]), torch.from_numpy(x_numpy[1]))
        # fro_norm = torch.linalg.norm(x_complex, ord='fro') + 1e-8
        # x_normalized = x_complex / fro_norm
        # x = torch.stack([x_normalized.real, x_normalized.imag], dim=0).float()

        if self.H is not None:
            h = self.H[idx]  # complex64 (M, N)
            h = torch.from_numpy(h)
            # h_fro_norm = torch.linalg.norm(h, ord='fro') + 1e-8
            # h_normalized = h / h_fro_norm
            return x, h
        else:
            return x


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
            # 最后的1x1卷积用于映射到最终维度
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

# class AttentionCouplingModule(nn.Module):
#     """
#     使用交叉注意力来耦合 U 和 V 的表征。
#     它接收 U 和 V 的初始表征，并通过让它们相互关注来对其进行迭代优化。
#     """
#     def __init__(self, embed_dim, num_heads, num_layers=2, dropout_rate=0.1):
#         super().__init__()
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList([])
#         for _ in range(num_layers):
#             self.layers.append(nn.ModuleList([
#                 # U attends to V
#                 nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True),
#                 # V attends to U
#                 nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True),
#                 nn.LayerNorm(embed_dim),
#                 nn.LayerNorm(embed_dim),
#                 nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim)),
#                 nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim)),
#                 nn.Dropout(dropout_rate),
#                 nn.Dropout(dropout_rate)
#             ]))
#     def forward(self, u, v):
#         """
#         u: (B, M, r) - M是序列长度, r是特征维度(embed_dim)
#         v: (B, N, r) - N是序列长度, r是特征维度(embed_dim)
#         """
#         # 为了处理复数，我们将实部和虚部叠加在特征维度上
#         u_real, u_imag = u.real, u.imag
#         v_real, v_imag = v.real, v.imag
#
#         # 将复数转换为 (B, Seq, 2*r) 的实数张量以输入注意力模块
#         u_rep = torch.cat([u_real, u_imag], dim=-1)
#         v_rep = torch.cat([v_real, v_imag], dim=-1)
#
#         for u_attend_v, v_attend_u, norm_u, norm_v, ffn_u, ffn_v, drop_u, drop_v in self.layers:
#             # U 从 V 中获取信息
#             u_updated, _ = u_attend_v(query=u_rep, key=v_rep, value=v_rep)
#             u_rep = norm_u(u_rep + drop_u(u_updated))  # Add & Norm
#             u_rep = norm_u(u_rep + drop_u(ffn_u(u_rep)))  # FFN
#
#             # V 从 U 中获取信息
#             v_updated, _ = v_attend_u(query=v_rep, key=u_rep, value=u_rep)
#             v_rep = norm_v(v_rep + drop_v(v_updated))  # Add & Norm
#             v_rep = norm_v(v_rep + drop_v(ffn_v(v_rep)))  # FFN
#
#         # 将 (B, Seq, 2*r) 的表征转换回 (B, Seq, r) 的复数张量
#         r = u.shape[-1]
#         u_final_real, u_final_imag = u_rep.split(r, dim=-1)
#         v_final_real, v_final_imag = v_rep.split(r, dim=-1)
#
#         u_final = torch.complex(u_final_real, u_final_imag)
#         v_final = torch.complex(v_final_real, v_final_imag)
#
#         return u_final, v_final


class SvdNet(nn.Module):
    def __init__(self, M=64, N=64, r=32):
        super().__init__()
        self.M, self.N, self.r = M, N, r

        self.enc = nn.Sequential(
            FourierFeatureBlock(1, h=64, w=64), # 2 channels = 1 complex pairs
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
        num_attn_heads = 4
        num_coupling_layers = 2

        self.decoder_head = ConvDecoderHead(fc_input_size, M, N, r)
        # self.coupling_module = AttentionCouplingModule(embed_dim=2 * r, num_heads=num_attn_heads,
        #                                                num_layers=num_coupling_layers)
        # self.coupling_module = LightweightCouplingModule(embed_dim=2 * r)

    def forward(self, x):
        fea = self.enc(x)

        # U = self.u_head(fea)
        # V = self.v_head(fea)
        U, V = self.decoder_head(fea)
        # U, V = self.coupling_module(U, V)

        U = qr_orthonormalize(U)
        V = qr_orthonormalize(V)
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
            S = analytic_sigma(U, V, torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()))
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
                S = analytic_sigma(U, V, torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()))
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
            torch.save(model.state_dict(), os.path.join(LOG_DIR, 'svd_best_multi.pth'))
            print("  → saved new best weights")

        with open(log_file, 'a') as f:
            f.write(
                f"Epoch {epoch + 1}: Train Loss : {train_loss:.6f} Val Loss: {val_loss:.6f}\n")

    print(f"Training complete, best val_loss = {best_loss:.4f}")


NUM_EPOCHS = 500
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_DIR = f"model/{timestamp}"

if __name__ == "__main__":
    # model_path = "./0.3230.pth"
    model_path = None
    main(model_path)
