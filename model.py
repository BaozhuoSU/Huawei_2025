import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from typing import Tuple, List
from dataset_loader import create_train_val_datasets
from torch.utils.data import DataLoader

class OrthonormalizeLayer(nn.Module):
    """
    A layer to enforce orthonormality on input matrices using Gram-Schmidt process.
    """

    def __init__(self):
        super(OrthonormalizeLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Gram-Schmidt orthogonalization to enforce orthonormality.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, M, r) or (batch_size, N, r), complex-valued.

        Returns:
            torch.Tensor: Orthonormalized tensor of the same shape.
        """
        batch_size, m, r = x.shape
        q = torch.zeros_like(x, dtype=torch.complex64)

        for b in range(batch_size):
            for i in range(r):
                v = x[b, :, i]
                for j in range(i):
                    q_j = q[b, :, j]
                    v = v - (torch.dot(q_j.conj(), v) / torch.dot(q_j.conj(), q_j)) * q_j
                norm = torch.sqrt(torch.dot(v.conj(), v).real)
                q[b, :, i] = v / norm if norm > 1e-6 else v
        return q


class SVDNet(nn.Module):
    """
    Neural network for predicting SVD components of MIMO channel matrices.

    Args:
        M (int): Number of rows in the channel matrix.
        N (int): Number of columns in the channel matrix.
        r (int): Number of singular values to predict.
    """

    def __init__(self, M: int, N: int, r: int):
        super(SVDNet, self).__init__()
        self.M = M
        self.N = N
        self.r = r

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # Input: (batch, 2, M, N)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce to (batch, 256, 4, 4)
        )

        self.fc_common = nn.Linear(256 * 4 * 4, 2048)
        self.fc_U = nn.Linear(2048, M * r * 2)
        self.fc_V = nn.Linear(2048, N * r * 2)
        self.fc_sigma = nn.Linear(2048, r)

        self.orthonorm_U = OrthonormalizeLayer()
        self.orthonorm_V = OrthonormalizeLayer()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict SVD components.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, M, N, 2), real-valued,
                             where the last dimension represents real and imaginary parts.

        Returns:
            Tuple containing:
            - U (torch.Tensor): Left singular vectors, shape (batch_size, M, r), complex-valued, orthonormal.
            - Sigma (torch.Tensor): Singular values, shape (batch_size, r), real-valued, positive, sorted.
            - V (torch.Tensor): Right singular vectors, shape (batch_size, N, r), complex-valued, orthonormal.
        """
        x_real = x.permute(0, 3, 1, 2)  # Shape: (batch_size, 2, M, N)

        features = self.feature_extractor(x_real)
        features = features.view(features.size(0), -1)
        features = F.relu(self.fc_common(features))

        u = self.fc_U(features)
        u = u.view(-1, self.M, self.r, 2).contiguous()  # Ensure contiguous for view_as_complex
        u = torch.view_as_complex(u)  # Shape: (batch_size, M, r)
        u = self.orthonorm_U(u)

        v = self.fc_V(features)
        v = v.view(-1, self.N, self.r, 2).contiguous()  # Ensure contiguous for view_as_complex
        v = torch.view_as_complex(v)  # Shape: (batch_size, N, r)
        v = self.orthonorm_V(v)

        sigma = self.fc_sigma(features)
        sigma = F.softplus(sigma)  # Ensure positive
        sigma, _ = torch.sort(sigma, dim=1, descending=True)  # Sort descending

        return u, sigma, v


def label_loss(labels: torch.Tensor, U: torch.Tensor, Sigma: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Computes the Frobenius norm between reconstructed and ground-truth channel matrices.

    Args:
        labels (torch.Tensor): Ground-truth channel matrices, shape (batch_size, M, N, 2), real-valued.
        U (torch.Tensor): Predicted left singular vectors, shape (batch_size, M, r), complex-valued.
        Sigma (torch.Tensor): Predicted singular values, shape (batch_size, r), real-valued.
        V (torch.Tensor): Predicted right singular vectors, shape (batch_size, N, r), complex-valued.

    Returns:
        torch.Tensor: Mean Frobenius norm of the reconstruction error.
    """
    H_gt = torch.complex(labels[..., 0], labels[..., 1])  # Shape: (batch_size, M, N)
    Sigma_diag = torch.diag_embed(Sigma)  # Shape: (batch_size, r, r), real-valued
    # Perform matrix multiplication, ensuring Sigma_diag is compatible with complex U
    U_Sigma = torch.bmm(U, Sigma_diag.to(dtype=U.dtype))  # Shape: (batch_size, M, r)
    H_pred = torch.bmm(U_Sigma, V.transpose(-1, -2).conj())  # Shape: (batch_size, M, N)
    return torch.norm(H_gt - H_pred, p='fro', dim=(1, 2)).mean()


def orthonormality_loss(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Penalizes deviation from orthonormality for U and V.

    Args:
        U (torch.Tensor): Left singular vectors, shape (batch_size, M, r), complex-valued.
        V (torch.Tensor): Right singular vectors, shape (batch_size, N, r), complex-valued.

    Returns:
        torch.Tensor: Mean Frobenius norm of the orthonormality error.
    """
    eye = torch.eye(U.shape[-1], device=U.device, dtype=torch.float32)
    U_loss = torch.norm(U.transpose(-1, -2).conj() @ U - eye[None, :, :], p='fro', dim=(1, 2)).mean()
    V_loss = torch.norm(V.transpose(-1, -2).conj() @ V - eye[None, :, :], p='fro', dim=(1, 2)).mean()
    return U_loss + V_loss


def get_avg_macs(model: nn.Module, input_data: torch.Tensor) -> float:
    """
    Estimates the average Mega MACs per sample for a model using PyTorch Profiler.

    Args:
        model (torch.nn.Module): The neural network model to profile.
        input_data (torch.Tensor): Input tensor for the model (must include batch dimension).

    Returns:
        float: Average Mega MACs per sample in the batch.

    Raises:
        RuntimeError: If input batch size is 0.
    """
    if input_data.dim() == 0 or input_data.size(0) == 0:
        raise RuntimeError("Input data must have a non-zero batch dimension")

    batch_size = input_data.size(0)
    model = model.eval().cpu()
    input_data = input_data.cpu()

    with torch.no_grad():
        with profile(
                activities=[ProfilerActivity.CPU],
                with_flops=True,
                record_shapes=False
        ) as prof:
            model(input_data)
    total_flops = sum(event.flops for event in prof.events() if event.flops is not None)
    avg_macs = total_flops / batch_size / 2 / 1e6  # Convert FLOPs to Mega MACs
    return avg_macs


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        lr: float = 0.001,
        ortho_weight: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[float]:
    """
    Trains the SVDNet model and evaluates on the validation set.

    Args:
        model (nn.Module): The SVDNet model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        ortho_weight (float): Weight for the orthonormality loss.
        device (str): Device to train on ('cpu' or 'cuda').

    Returns:
        List[float]: List of validation losses per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, labels, _ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            u, sigma, v = model(data)
            recon_loss = label_loss(labels, u, sigma, v)
            ortho_loss = orthonormality_loss(u, v)
            loss = recon_loss + ortho_weight * ortho_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data, labels = data.to(device), labels.to(device)
                u, sigma, v = model(data)
                recon_loss = label_loss(labels, u, sigma, v)
                loss = recon_loss
                val_loss += loss.item() * data.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}")

    return val_losses


if __name__ == "__main__":
    # Example usage with real data
    data_dir = "CompetitionData1"
    output_dir = "submission"
    try:
        # Create training and validation DataLoaders
        train_loader, val_loader = create_train_val_datasets(
            data_dir=data_dir,
            round_idx=[1, 2, 3],
            val_split=0.2,
            batch_size=32,
            shuffle=True
        )

        # Initialize model with config from dataset
        config = train_loader.dataset.dataset.datasets[0].config
        model = SVDNet(M=config['M'], N=config['N'], r=config['r'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        # Get a batch of real data for testing
        data, labels, _ = next(iter(train_loader))
        data = data.to(device)
        labels = labels.to(device)

        # Verify tensor types and shapes
        print(f"Data dtype: {data.dtype}")  # Expected: torch.float32
        print(f"Labels dtype: {labels.dtype}")  # Expected: torch.float32
        print(f"Data shape: {data.shape}")  # Expected: (32, 64, 64, 2)
        print(f"Labels shape: {labels.shape}")  # Expected: (32, 64, 64, 2)

        # TODO
        # # Forward pass
        # u, sigma, v = model(data)
        # print(f"U shape: {u.shape}")  # Expected: (32, 64, 32)
        # print(f"Sigma shape: {sigma.shape}")  # Expected: (32, 32)
        # print(f"V shape: {v.shape}")  # Expected: (32, 64, 32)
        #
        # # Compute losses
        # recon_loss = label_loss(labels, u, sigma, v)
        # ortho_loss = orthonormality_loss(u, v)
        # print(f"Reconstruction loss: {recon_loss.item():.4f}")
        # print(f"Orthonormality loss: {ortho_loss.item():.4f}")
        #
        # # Estimate MACs
        # macs = get_avg_macs(model, data)
        # print(f"Average Mega MACs per sample: {macs:.2f}")
        #
        # # Train the model
        # val_losses = train_model(model, train_loader, val_loader, num_epochs=10)
        # print(f"Validation losses: {val_losses}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")