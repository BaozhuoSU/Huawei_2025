import json
import os
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader

from dataset import parse_cfg, ChannelDataset
from augment_dataset import AugmentChannelDataset
from loss import LAELoss
from model import Model1
from model2 import Model2


def get_avg_flops(model:nn.Module, input_data:Tensor)->float:
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

def load_model(weight_path: str, M: int, N: int, r: int) -> nn.Module:
    model = Model1(M, N, r)
    state_dict = torch.load(weight_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict)
    return model

def process_and_export(model: nn.Module, input_tensor: Tensor, output_path: str, file_index: int, device: str = 'cuda') -> None:
    model.eval()

    input_tensor = input_tensor.to(device, non_blocking=True)

    with torch.no_grad():
        U, S, V = model(input_tensor)

    U = U.cpu().numpy()
    V = V.cpu().numpy()
    S = S.cpu().numpy()

    U = np.stack([U.real, U.imag], axis=-1)
    V = np.stack([V.real, V.imag], axis=-1)
    print(f"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")

    C = get_avg_flops(model, input_tensor)
    print(f"Average FLOPs for file {file_index}: {C:.4f} M")

    np.savez(
        os.path.join(output_path, f"{file_index}.npz"),
        U=U,
        S=S,
        V=V,
        C=float(C)
    )


def validate_model(model: nn.Module, data_loader: DataLoader, device: str, loss_fn: nn.Module) -> float:
    model.eval()
    model.to(device)
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)

            U_pred, S_pred, V_pred = model(data)
            H_label = labels

            loss = loss_fn(U_pred, S_pred, V_pred, H_label)

            total_loss += loss.item() * data.size(0)
            num_samples += data.size(0)

    mean_loss = total_loss / num_samples

    print(f"Validation Loss: {mean_loss:.4f}")
    return mean_loss

if __name__ == "__main__":
    DATA_DIR = './CompetitionData1'
    MODEL_PATH = './model/model_epoch_1.pth'
    # MODEL_PATH = './svd_approximator.pth'
    OUTPUT_DIR = './outputs'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    M, N, r = 64, 64, 32
    loss_fn = LAELoss(recon_weight=1, ortho_weight=0.1).to(DEVICE)

    model = load_model(MODEL_PATH, M, N, r)

    for idx in range(1, 4):
        data_path = os.path.join(DATA_DIR, f'Round1TrainData{idx}.npy')
        label_path = os.path.join(DATA_DIR, f'Round1TrainLabel{idx}.npy')

        dataset = AugmentChannelDataset(
            data_path=data_path,
            label_path=label_path,
            cfg_path=None,
            is_augment=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        mean_loss = validate_model(model, dataloader, DEVICE, loss_fn)

        npy_path = os.path.join(DATA_DIR, f'Round1TestData{idx}.npy')
        input_array = np.load(npy_path)
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        process_and_export(model, input_tensor, OUTPUT_DIR, idx)