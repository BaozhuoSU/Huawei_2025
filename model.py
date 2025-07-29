import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import Tensor
from torch.profiler import profile, ProfilerActivity
from typing import Tuple, List
from dataset_loader import create_train_val_datasets
from torch.utils.data import DataLoader
from datetime import datetime

class SVDNet(nn.Module):
    """
    Neural network for predicting SVD components of MIMO channel matrices.

    Args:
        M (int): Number of rows in the channel matrix.
        N (int): Number of columns in the channel matrix.
        r (int): Number of singular values to predict.
        Q (int): 2 for complex-valued data.
    """

    def __init__(self, M: int = 64, N: int = 64, r: int = 32, Q: int = 2):
        super(SVDNet, self).__init__()
        self.M = M
        self.N = N
        self.r = r
        self.Q = Q

        # Convolutional layers
        self.pad = nn.ZeroPad2d(7)
        self.conv1 = nn.Conv2d(Q, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.2)

        conv_out_h = (M + 14 - 3 + 1) // 1
        conv_out_w = (N + 14 - 3 + 1) // 1
        conv_out_h = (conv_out_h - 3 + 1) // 1
        conv_out_w = (conv_out_w - 3 + 1) // 1
        pool_out_h = conv_out_h // 2
        pool_out_w = conv_out_w // 2
        flat_dim = 64 * pool_out_h * pool_out_w

        # Fully connected layers
        self.fc1 = nn.Linear(flat_dim, 512)
        self.fc2_s = nn.Linear(512, r)
        self.fc2_u = nn.Linear(512, M * r * Q)
        self.fc2_v = nn.Linear(512, N * r * Q)
        self.dropout_fc = nn.Dropout(0.1)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.permute(0, 3, 1, 2)
        x = self.pad(x)
        x = F.elu(self.conv1(x))  # (batch_size, 32, M+12, N+12)
        x = F.elu(self.conv2(x))  # (batch_size, 64, M+10, N+10)
        x = self.pool(x)  # (batch_size, 64, M+5, N+5)
        x = self.dropout_conv(x)
        # x = x.view(x.size(0), -1)  # Flatten
        x = x.reshape(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = self.dropout_fc(x)

        # Predict S, U, V
        S = self.fc2_s(x)  # (batch_size, r)
        U = self.fc2_u(x).reshape(-1, self.M, self.r, self.Q)
        V = self.fc2_v(x).reshape(-1, self.N, self.r, self.Q)
        return U, S, V


def loss_AE(labels: torch.Tensor, U: torch.Tensor, Sigma: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Computes the Approximation Error (AE) for SVD reconstruction and orthonormality.

    Args:
        labels (torch.Tensor): Ground-truth channel matrices, shape (batch_size, M, N, 2), real-valued.
        U (torch.Tensor): Predicted left singular vectors, shape (batch_size, M, r), complex-valued.
        Sigma (torch.Tensor): Predicted singular values, shape (batch_size, r), real-valued.
        V (torch.Tensor): Predicted right singular vectors, shape (batch_size, N, r), complex-valued.

    Returns:
        torch.Tensor: Mean Approximation Error (AE) across the batch, as defined by:
            L_AE = ||H_label - UΣV^H||_F / ||H_label||_F + ||U^H U - I||_F + ||V^H V - I||_F
    """
    # Convert labels to complex-valued ground-truth channel matrix H_gt
    H_gt = torch.complex(labels[..., 0], labels[..., 1])  # Shape: (batch_size, M, N)

    U_complex = torch.complex(U[..., 0], U[..., 1])  # Shape: (batch_size, M, r)
    V_complex = torch.complex(V[..., 0], V[..., 1])  # Shape: (batch_size, N, r)
    # Construct diagonal matrix from Sigma
    Sigma_diag = torch.diag_embed(Sigma)  # Shape: (batch_size, r, r), real-valued

    # Compute reconstructed channel matrix H_pred = U Σ V^H
    U_Sigma = torch.bmm(U_complex, Sigma_diag.to(dtype=U_complex.dtype))  # Shape: (batch_size, M, r)
    H_pred = torch.bmm(U_Sigma, V_complex.transpose(-1, -2).conj())  # Shape: (batch_size, M, N)

    # Compute Frobenius norm of reconstruction error
    recon_error = torch.norm(H_gt - H_pred, p='fro', dim=(1, 2))  # Shape: (batch_size,)
    H_norm = torch.norm(H_gt, p='fro', dim=(1, 2))  # Shape: (batch_size,)
    H_norm = torch.clamp(H_norm, min=1e-6)  # Avoid division by zero
    recon_loss = (recon_error / H_norm).mean()  # Normalized reconstruction error

    # Compute orthonormality losses for U and V
    eye = torch.eye(U_complex.shape[-1], device=U_complex.device, dtype=torch.complex64)  # Identity matrix
    U_loss = torch.norm(U_complex.transpose(-1, -2).conj() @ U_complex - eye[None, :, :], p='fro', dim=(1, 2)).mean()
    V_loss = torch.norm(V_complex.transpose(-1, -2).conj() @ V_complex - eye[None, :, :], p='fro', dim=(1, 2)).mean()

    # Total Approximation Error (AE)
    total_loss = recon_loss + U_loss + V_loss
    return total_loss, recon_loss, U_loss, V_loss

def get_avg_flops(model: nn.Module, input_data: Tensor) -> float:
    """
    Estimates the average FLOPs per sample for a model using PyTorch Profiler.

    Args:
        model (torch.nn.Module): The neural network model to profile.
        input_data (torch.Tensor): Input tensor for the model (must include batch dimension).

    Returns:
        float: Average Mega FLOPs per sample in the batch.

    Raises:
        RuntimeError: If no CUDA device is available or input batch size is 0.
    """

    # Ensure batch dimension exists
    if input_data.dim() == 0 or input_data.size(0) == 0:
        raise RuntimeError("Input data must have a non-zero batch dimension")

    batch_size = input_data.size(0)

    # Evaluation mode, improved inference and freeze norm layers
    model = model.eval().cpu()
    input_data = input_data.cpu()

    with torch.no_grad():
        with profile(
                activities=[ProfilerActivity.CPU],
                with_flops=True,
                record_shapes=False
        ) as prof:
            model(input_data)
    # Calculate total FLOPs
    total_flops = sum(event.flops for event in prof.events())
    avg_flops = total_flops / batch_size

    return avg_flops * 1e-6 / 2

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        lr: float = 0.001,
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
        device (str): Device to train on ('cpu' or 'cuda').

    Returns:
        List[float]: List of validation losses per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_path = None
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f"model/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'training_log.txt')

    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Dataset sizes: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}\n")
        f.write(f"Batch sizes: Train={train_loader.batch_size}, Val={val_loader.batch_size}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(
            "Epoch,Train Loss,Train Recon Loss,Train U Loss,Train V Loss,Val Loss,Val Recon Loss,Val U Loss,Val V Loss\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_U_loss = 0.0
        train_V_loss = 0.0
        for data, labels, _ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            u, sigma, v = model(data)
            loss, recon_loss, U_loss, V_loss = loss_AE(labels, u, sigma, v)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            train_recon_loss += recon_loss.item() * data.size(0)
            train_U_loss += U_loss.item() * data.size(0)
            train_V_loss += V_loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)
        train_U_loss /= len(train_loader.dataset)
        train_V_loss /= len(train_loader.dataset)

        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, train Recon Loss: {train_recon_loss:.4f}, train U Loss: {train_U_loss:.4f}, train V Loss: {train_V_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_U_loss = 0.0
        val_V_loss = 0.0
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data, labels = data.to(device), labels.to(device)
                u, sigma, v = model(data)
                loss, recon_loss, U_loss, V_loss = loss_AE(labels, u, sigma, v)
                val_loss += loss.item() * data.size(0)

                val_recon_loss += recon_loss.item() * data.size(0)
                val_U_loss += U_loss.item() * data.size(0)
                val_V_loss += V_loss.item() * data.size(0)
            val_loss /= len(val_loader.dataset)
            val_recon_loss /= len(val_loader.dataset)
            val_U_loss /= len(val_loader.dataset)
            val_V_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, val Recon Loss: {val_recon_loss:.4f}, val U Loss: {val_U_loss:.4f}, val V Loss: {val_V_loss:.4f}")

        model_path = os.path.join(log_dir, f"model_epoch_{epoch + 1}.pth")
        if val_loss < best_val_loss:
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            torch.save(model.state_dict(), model_path)
            best_val_loss = val_loss
            best_model_path = model_path
            print(f"New best model saved at {model_path} with Val Loss: {val_loss:.4f}")
        else:
            if os.path.exists(model_path):
                os.remove(model_path)

        with open(log_file, 'a') as f:
            f.write(
                f"{epoch + 1},{train_loss:.4f},{train_recon_loss:.4f},{train_U_loss:.4f},{train_V_loss:.4f},{val_loss:.4f},{val_recon_loss:.4f},{val_U_loss:.4f},{val_V_loss:.4f}\n")

    return train_losses, val_losses


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
        print(f"Using device: {device}")
        model = model.to(device)

        # Get a batch of real data for testing
        data, labels, _ = next(iter(train_loader))
        data = data.to(device)
        labels = labels.to(device)

        print(f"Data shape: {data.shape}")  # Expected: (32, 64, 64, 2)
        print(f"Labels shape: {labels.shape}")  # Expected: (32, 64, 64, 2)

        # Forward pass
        u, sigma, v = model(data)
        print(f"U shape: {u.shape}")  # Expected: (32, 64, 32, 2)
        print(f"Sigma shape: {sigma.shape}")  # Expected: (32, 32)
        print(f"V shape: {v.shape}")  # Expected: (32, 64, 32, 2)

        # Compute losses
        loss, recon_loss, U_loss, V_loss = loss_AE(labels, u, sigma, v)
        print(f"AE loss: {loss.item():.4f}, Reconstruction loss: {recon_loss.item():.4f}, U loss: {U_loss.item():.4f}, V loss: {V_loss.item():.4f}")

        # Train the model
        train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend(['Train Loss', 'Validation Loss'])
        plt.savefig('loss_plot.png')

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")