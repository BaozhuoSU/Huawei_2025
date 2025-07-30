import torch
import torch.nn as nn
from typing import Tuple


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 添加BatchNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 添加BatchNorm

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),  # 添加BatchNorm
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class EfficientSVDNet(nn.Module):
    def __init__(self, M: int, N: int, r: int):
        super(EfficientSVDNet, self).__init__()
        self.M, self.N, self.r = M, N, r

        self.conv = nn.Sequential(
            # nn.Conv2d(2, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            #
            # nn.Conv2d(64, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            #
            # nn.Conv2d(128, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),

            # nn.Conv2d(2, 32, 3, 1, 1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 64, 3, 1, 1),
            # nn.ReLU(True),

            ResidualBlock(2, 32, stride=2),
            ResidualBlock(32, 64, stride=1),

            CBAM(64, reduction_ratio=4, spatial_kernel_size=3),
        )

        self.u_proj = nn.Sequential(
            nn.Conv2d(64, r * 2, 1),
            nn.BatchNorm2d(r * 2),
            nn.AdaptiveAvgPool2d((M, 1)),
            nn.Flatten(2)
        )

        self.v_proj = nn.Sequential(
            nn.Conv2d(64, r * 2, 1),
            nn.BatchNorm2d(r * 2),
            nn.AdaptiveAvgPool2d((1, N)),
            nn.Flatten(2)
        )

        self.s_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, r, 1),
            nn.BatchNorm2d(r),
            nn.Flatten(),
            nn.Softplus()
        )

    def orthogonalize(self, matrix):
        U, _, V = torch.linalg.svd(matrix, full_matrices=False)
        return U @ V

    def forward(self, H_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_real = H_in.permute(0, 3, 1, 2)
        feat = self.conv(H_real)

        u_out = self.u_proj(feat).squeeze(-1)
        U = torch.view_as_complex(u_out.view(-1, self.M, self.r, 2).contiguous())
        U = self.orthogonalize(U)

        v_out = self.v_proj(feat).squeeze(-2)
        V = torch.view_as_complex(v_out.view(-1, self.N, self.r, 2).contiguous())
        V = self.orthogonalize(V)

        S = self.s_proj(feat)
        S = torch.clamp(S, min=1e-6, max=1e3)
        S, idx = torch.sort(S, dim=-1, descending=True)

        U = U.gather(2, idx.unsqueeze(1).expand(-1, self.M, -1))
        V = V.gather(2, idx.unsqueeze(1).expand(-1, self.N, -1))

        return U, S, V