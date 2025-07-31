# train_svd_net.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pathlib import Path
from tqdm import tqdm

# =============================================================
# 定义：ChannelSvdDataset
# ———— 使用 mmap_mode 快速加载 .npy，不再 astype
# =============================================================
class ChannelSvdDataset(Dataset):
    def __init__(self, data_path, label_path=None):
        raw = np.load(data_path, mmap_mode='r')      # float32 (Ns, M, N, 2)
        self.M, self.N = raw.shape[1], raw.shape[2]
        # 转成 (Ns, 2, M, N)
        self.X = raw.transpose(0,3,1,2)
        # 如果给出了 label_path，再加载 H
        if label_path is not None:
            lab = np.load(label_path, mmap_mode='r')
            # 若 shape==(Ns,M,N,2) 则合成 complex
            if lab.ndim==4 and lab.shape[-1]==2:
                H_real = lab[...,0]
                H_imag = lab[...,1]
                self.H = (H_real + 1j*H_imag).astype(np.complex64)
            else:
                # 已经是 (Ns, M, N) complex 存储
                self.H = lab
        else:
            self.H = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()    # (2, M, N)
        if self.H is not None:
            h = self.H[idx]                          # complex64 (M, N)
            return x, torch.from_numpy(h)
        else:
            return (x,)

# =============================================================
# 定义：SvdNet 模型
# ———— 轻量级 Conv→GAP→FC → 输出 U, V
# =============================================================
class SvdNet(nn.Module):
    def __init__(self, M=64, N=64, r=32):
        super().__init__()
        self.M, self.N, self.r = M, N, r
        # encoder backbone
        self.enc = nn.Sequential(
            nn.Conv2d(2, 12, 3, 1, 1), nn.ELU(),
            nn.Conv2d(12,24, 3, 1, 1), nn.ELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        # predictor FC: [U_real, U_imag, V_real, V_imag]
        self.fc = nn.Sequential(
            nn.Linear(24, 48), nn.ELU(),
            nn.Linear(48, 2*M*r + 2*N*r)
        )
        # 初始化 Σ bias>0 （前 r 个偏置位置）
        with torch.no_grad():
            self.fc[-1].bias[:r].fill_(0.5)

    def forward(self, x):
        # x: (B,2,M,N)
        B = x.size(0)
        y = self.fc(self.enc(x))                     # (B, 2*M*r + 2*N*r)
        p = 0
        # U 实部
        U_r = y[:, p:p+self.M*self.r].reshape(B, self.M, self.r)
        p += self.M*self.r
        # U 虚部
        U_i = y[:, p:p+self.M*self.r].reshape(B, self.M, self.r)
        p += self.M*self.r
        # V 实部
        V_r = y[:, p:p+self.N*self.r].reshape(B, self.N, self.r)
        p += self.N*self.r
        # V 虚部
        V_i = y[:, p:p+self.N*self.r].reshape(B, self.N, self.r)
        # 组装复数 U, V
        U = torch.complex(U_r, U_i)                  # (B, M, r)
        V = torch.complex(V_r, V_i)                  # (B, N, r)
        return U, V

# =============================================================
# 定义：自监督 AE Loss
# L_AE = ||H - UΣV^H||_F/||H||_F  +  ||U^H U - I||_F  +  ||V^H V - I||_F
# =============================================================
def analytic_sigma(U, V, H):
    # 计算 Σ = diag( U^H H V )
    # U:(B,M,r), V:(B,N,r), H:(B,M,N)
    return (U.conj().permute(0,2,1) @ H @ V) \
            .diagonal(dim1=1, dim2=2).real  # (B, r)

def ae_loss(U, S, V, H, lam=1.0):
    B = H.size(0)
    # 构造 Σ 张量
    Sigma = torch.zeros(B, S.size(1), S.size(1),
                        dtype=torch.complex64, device=H.device)
    Sigma[:, torch.arange(S.size(1)), torch.arange(S.size(1))] = S.type(torch.complex64)
    # 重构
    H_hat = U @ Sigma @ V.conj().permute(0,2,1)
    # 重构误差
    num   = torch.linalg.norm(H - H_hat, ord='fro', dim=(1,2))
    denom = torch.linalg.norm(H,       ord='fro', dim=(1,2)).clamp_min(1e-8)
    recon = num / denom
    # 正交惩罚
    I_r = torch.eye(S.size(1), dtype=torch.complex64, device=H.device)
    UU = U.conj().permute(0,2,1) @ U
    VV = V.conj().permute(0,2,1) @ V
    err_u = torch.linalg.norm(UU - I_r, ord='fro', dim=(1,2))
    err_v = torch.linalg.norm(VV - I_r, ord='fro', dim=(1,2))
    return (recon + lam*(err_u + err_v)).mean()

# =============================================================
# 主函数：加载 data2/data3/data4，训练 & 验证
# =============================================================
def main():
    # 1) 准备数据集
    ROOTS = [Path.cwd()/f"data{i}" for i in (1,2,3)]
    train_sets = []
    for root in ROOTS:
        td = root / f"Round1TrainData{root.name[-1]}.npy"
        tl = root / f"Round1TrainLabel{root.name[-1]}.npy"
        if td.exists() and tl.exists():
            train_sets.append(ChannelSvdDataset(str(td), str(tl)))
    assert train_sets, "No training data found!"

    full_ds = ConcatDataset(train_sets)
    N = len(full_ds)
    n_train = int(0.9 * N)
    n_val   = N - n_train

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    # 2) 模型 & 优化器 & LR Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    M, Nmat, r = train_sets[0].M, train_sets[0].N, 32
    model = SvdNet(M, Nmat, r).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=30,
                                                           eta_min=1e-6)

    # 3) 训练循环
    best_loss, patience = float('inf'), 0
    for epoch in range(1, 31):
        model.train()
        running, count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/30", ncols=80)
        for x, H in pbar:
            x, H = x.to(device), H.to(device)
            optimizer.zero_grad()
            U, V = model(x)
            S     = analytic_sigma(U, V, H)
            loss  = ae_loss(U, S, V, H)
            loss.backward()
            optimizer.step()

            b = x.size(0)
            running += loss.item() * b
            count   += b
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / count

        # 验证
        model.eval()
        v_sum, v_cnt = 0.0, 0
        with torch.no_grad():
            for x, H in val_loader:
                x, H = x.to(device), H.to(device)
                U, V = model(x)
                S     = analytic_sigma(U, V, H)
                l     = ae_loss(U, S, V, H)
                b     = x.size(0)
                v_sum += l.item() * b
                v_cnt += b
        val_loss = v_sum / v_cnt

        scheduler.step(val_loss)
        print(f"Epoch {epoch} done — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # 保存最优
        if val_loss < best_loss - 1e-4:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), 'svd_best_multi.pth')
            print("  → saved new best weights")
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping.")
                break

    print(f"Training complete, best val_loss = {best_loss:.4f}")

if __name__ == "__main__":
    main()