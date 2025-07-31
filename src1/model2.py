# svd_approximator.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm


# ----------------- 2. Residual Block -----------------
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
            nn.ELU(),
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


# ----------------- 3. Model w/ Σ BatchNorm + 排序 + Clamp -----------------
class Model2(nn.Module):
    def __init__(self, M=64, N=64, r=32):
        super().__init__()
        self.M, self.N, self.r = M, N, r

        self.enc = nn.Sequential(
            nn.Conv2d(2, 8, 3, 1, 1),
            nn.ELU(),
            # ResidualBlock(4),
            CBAM(8),

            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ELU(),
            # ResidualBlock(16),
            CBAM(16),

            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        # predictor FC: [U_real, U_imag, V_real, V_imag]
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, 2 * M * r + 2 * N * r)
        )
        # 初始化 Σ bias>0 （前 r 个偏置位置）
        with torch.no_grad():
            self.fc[-1].bias[:r].fill_(0.5)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # x: (B,2,M,N)
        B = x.size(0)
        y = self.fc(self.enc(x))  # (B, 2*M*r + 2*N*r)
        p = 0
        # U 实部
        U_r = y[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        # U 虚部
        U_i = y[:, p:p + self.M * self.r].reshape(B, self.M, self.r)
        p += self.M * self.r
        # V 实部
        V_r = y[:, p:p + self.N * self.r].reshape(B, self.N, self.r)
        p += self.N * self.r
        # V 虚部
        V_i = y[:, p:p + self.N * self.r].reshape(B, self.N, self.r)
        # 组装复数 U, V
        U = torch.complex(U_r, U_i)  # (B, M, r)
        V = torch.complex(V_r, V_i)  # (B, N, r)
        return U, V


def analytic_sigma(U, V, H):
    # 计算 Σ = diag( U^H H V )
    # U:(B,M,r), V:(B,N,r), H:(B,M,N)
    raw_S = (U.conj().permute(0, 2, 1) @ H @ V) \
        .diagonal(dim1=1, dim2=2).real  # (B, r)

    S = torch.clamp(raw_S, min=0.0)
    return S  # (B, r)


if __name__ == "__main__":
    pass
