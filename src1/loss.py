import torch

class LAELoss(torch.nn.Module):
    def __init__(self, recon_weight=1.0, ortho_weight=1.0):
        super(LAELoss, self).__init__()
        self.recon_weight = recon_weight
        self.ortho_weight = ortho_weight

    def forward(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, H_label: torch.Tensor) -> torch.Tensor:
        # batch_size, r = S.shape
        #
        # Sigma = torch.diag_embed(S.float()).cfloat()
        # H_hat = U @ Sigma @ V.mH
        #
        # h_norm = torch.linalg.matrix_norm(H_label, ord='fro', dim=(-2,-1), keepdim=True)
        # h_norm = torch.clamp(h_norm, min=1e-8)
        # recon_error = torch.linalg.matrix_norm(H_label - H_hat, ord='fro', dim=(-2,-1)) / h_norm.squeeze()
        #
        # I_r = torch.eye(r, device=U.device, dtype=U.dtype)
        # u_ortho = torch.linalg.matrix_norm(U.mH @ U - I_r, ord='fro', dim=(-2,-1))
        # v_ortho = torch.linalg.matrix_norm(V.mH @ V - I_r, ord='fro', dim=(-2,-1))
        B = H_label.size(0)
        # 构造 Σ 张量
        Sigma = torch.zeros(B, S.size(1), S.size(1),
                            dtype=torch.complex64, device=H_label.device)
        Sigma[:, torch.arange(S.size(1)), torch.arange(S.size(1))] = S.type(torch.complex64)
        # 重构
        H_hat = U @ Sigma @ V.conj().permute(0, 2, 1)
        # 重构误差
        num = torch.linalg.norm(H_label - H_hat, ord='fro', dim=(1, 2))
        denom = torch.linalg.norm(H_label, ord='fro', dim=(1, 2)).clamp_min(1e-8)
        recon = num / denom
        # 正交惩罚
        I_r = torch.eye(S.size(1), dtype=torch.complex64, device=H_label.device)
        UU = U.conj().permute(0, 2, 1) @ U
        VV = V.conj().permute(0, 2, 1) @ V
        err_u = torch.linalg.norm(UU - I_r, ord='fro', dim=(1, 2))
        err_v = torch.linalg.norm(VV - I_r, ord='fro', dim=(1, 2))
        return (self.recon_weight * recon + self.ortho_weight * (err_u + err_v)).mean()
        
        # return (self.recon_weight * recon_error + self.ortho_weight * (u_ortho + v_ortho)).mean()