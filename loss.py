import torch

class LAELoss(torch.nn.Module):
    def __init__(self, recon_weight=1.0, ortho_weight=1.0):
        super(LAELoss, self).__init__()
        self.recon_weight = recon_weight
        self.ortho_weight = ortho_weight

    def forward(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, H_label: torch.Tensor) -> torch.Tensor:
        batch_size, r = S.shape
        
        Sigma = torch.diag_embed(S.float()).cfloat()
        H_hat = U @ Sigma @ V.mH
        
        h_norm = torch.linalg.matrix_norm(H_label, ord='fro', dim=(-2,-1), keepdim=True)
        h_norm = torch.clamp(h_norm, min=1e-8)
        recon_error = torch.linalg.matrix_norm(H_label - H_hat, ord='fro', dim=(-2,-1)) / h_norm.squeeze()
        
        I_r = torch.eye(r, device=U.device, dtype=U.dtype)
        u_ortho = torch.linalg.matrix_norm(U.mH @ U - I_r, ord='fro', dim=(-2,-1))
        v_ortho = torch.linalg.matrix_norm(V.mH @ V - I_r, ord='fro', dim=(-2,-1))
        
        return (self.recon_weight * recon_error + self.ortho_weight * (u_ortho + v_ortho)).mean()