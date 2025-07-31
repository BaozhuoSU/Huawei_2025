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


# ----------------- 3. Model w/ Σ BatchNorm + 排序 + Clamp -----------------
class Model2(nn.Module):
    def __init__(self, M=64, N=64, r=32):
        super().__init__()
        self.M, self.N, self.r = M, N, r

        # backbone: conv -> ELU -> residual -> conv -> ELU -> pool -> flatten
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ELU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        feat_dim = (M // 2) * (N // 2) * 64
        out_dim = r + 2 * M * r + 2 * N * r

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ELU(),
            nn.Linear(512, out_dim)
        )
        # 对 Σ 做 BatchNorm（不含可学习参数）
        self.bn_sigma = nn.BatchNorm1d(r, affine=False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B = x.size(0)
        y = self.fc(self.backbone(x))  # (B, out_dim)
        pos = 0

        # 1) raw Σ -> softplus -> clamp -> batchnorm -> sort
        raw_S = y[:, pos:pos + self.r]
        pos += self.r
        S = F.softplus(raw_S)
        S = torch.clamp(S, min=1e-2)  # 下界截断
        S = self.bn_sigma(S)  # 零均值单位方差
        S, _ = torch.sort(S, descending=True, dim=1)

        # 2) U_real / U_imag -> U_complex
        U_r = y[:, pos:pos + self.M * self.r].reshape(B, self.M, self.r)
        pos += self.M * self.r
        U_i = y[:, pos:pos + self.M * self.r].reshape(B, self.M, self.r)
        pos += self.M * self.r
        U = U_r + 1j * U_i

        # 3) V_real / V_imag -> V_complex
        V_r = y[:, pos:pos + self.N * self.r].reshape(B, self.N, self.r);
        pos += self.N * self.r
        V_i = y[:, pos:pos + self.N * self.r].reshape(B, self.N, self.r)
        V = V_r + 1j * V_i

        return U, S, V


if __name__ == "__main__":
    pass
