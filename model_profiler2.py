import json
import os
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader

from su2 import SvdNet, ae_loss, analytic_sigma, SimpleChannelDataset


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


def load_model(weight_path: str, M: int, N: int, r: int) -> nn.Module:
    model = SvdNet(M, N, r)
    state_dict = torch.load(weight_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {DEVICE}")
    model.load_state_dict(state_dict)
    return model


def process_and_export(model: nn.Module, data_loader: DataLoader, output_path: str, file_index: int,
                       device: str = 'cuda') -> None:
    model.eval()

    U_list, S_list, V_list = [], [], []
    with torch.no_grad():
        # for x_norm, fro_norm in data_loader:
        for x_raw in data_loader:
            # # print("Debug fro_norm:", fro_norm)
            # x_norm, fro_norm = x_norm.to(device), fro_norm.to(device)
            # U_pred, V_pred = model(x_norm)
            # x_norm = x_norm.permute(0, 2, 3, 1).contiguous()
            # x_norm_complex = torch.view_as_complex(x_norm)
            # S_norm = analytic_sigma(U_pred, V_pred, x_norm_complex).to(device)
            # S_pred = S_norm * fro_norm.unsqueeze(1)
            x_raw = x_raw.to(device)
            x_complex = torch.complex(x_raw[:, 0, :, :], x_raw[:, 1, :, :])
            fro_norm = torch.linalg.norm(x_complex, ord='fro', dim=(1, 2)) + 1e-8
            x_normalized_complex = x_complex / fro_norm.view(-1, 1, 1)
            x_fft_complex = torch.fft.fft2(x_normalized_complex, norm='ortho')

            # x_mag = torch.abs(x_normalized_complex)
            # x_log_power = torch.log(x_mag ** 2 + 1e-8)
            # x_fft_mag = torch.abs(x_fft_complex)
            # x_fft_log_power = torch.log(x_fft_mag ** 2 + 1e-8)

            x_model_input = torch.stack([
                x_normalized_complex.real,
                x_normalized_complex.imag,
                # x_mag,
                # x_log_power,
                x_fft_complex.real,
                x_fft_complex.imag,
                # x_fft_mag,
                # x_fft_log_power
            ], dim=1).float()
            x_normalized_for_sigma = torch.view_as_complex(
                torch.stack([x_normalized_complex.real, x_normalized_complex.imag], dim=1).permute(0, 2, 3, 1).contiguous()
            )
            U_pred, V_pred = model(x_model_input)
            S_norm = analytic_sigma(U_pred, V_pred, x_normalized_for_sigma)
            S_pred = S_norm * fro_norm.unsqueeze(1)
            U_list.append(U_pred.cpu().numpy())
            S_list.append(S_pred.cpu().numpy())
            V_list.append(V_pred.cpu().numpy())

    U_arr = np.concatenate(U_list, axis=0)  # (Ns, M, r)
    V_arr = np.concatenate(V_list, axis=0)  # (Ns, N, r)

    S_out = np.concatenate(S_list, axis=0)  # (Ns, r)
    U_out = np.stack([U_arr.real, U_arr.imag], axis=-1)  # (Ns, M, r, 2)
    V_out = np.stack([V_arr.real, V_arr.imag], axis=-1)  # (Ns, N, r, 2)
    print(f"U shape: {U_out.shape}, S shape: {S_out.shape}, V shape: {V_out.shape}")

    dummy = torch.randn(1, 4, M, N).to(device)
    C = get_avg_flops(model, dummy)
    print(f"Average FLOPs for file {file_index}: {C:.4f} M")

    np.savez(
        os.path.join(output_path, f"{file_index}.npz"),
        U=U_out,
        S=S_out,
        V=V_out,
        C=float(C)
    )


def validate_model(model: nn.Module, data_loader: DataLoader, device: str) -> float:
    model.eval()
    model.to(device)
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        # for x_norm, H, fro_norm in data_loader:
        for x_raw, H_label in data_loader:
            # x_norm, H, fro_norm = x_norm.to(device), H.to(device), fro_norm.to(device)
            # U_pred, V_pred = model(x_norm)
            # H_label = H
            # x_norm = x_norm.permute(0, 2, 3, 1).contiguous()
            # x_norm_complex = torch.view_as_complex(x_norm)
            # S_norm = analytic_sigma(U_pred, V_pred, x_norm_complex).to(device)
            # S_pred = S_norm * fro_norm.unsqueeze(1)
            # loss = ae_loss(U_pred, S_pred, V_pred, H_label)
            # b = x_norm.size(0)
            # total_loss += loss.item() * b
            # num_samples += b
            x_raw, H_label = x_raw.to(device), H_label.to(device)
            x_complex = torch.complex(x_raw[:, 0, :, :], x_raw[:, 1, :, :])
            fro_norm = torch.linalg.norm(x_complex, ord='fro', dim=(1, 2)) + 1e-8
            x_normalized_complex = x_complex / fro_norm.view(-1, 1, 1)
            x_fft_complex = torch.fft.fft2(x_normalized_complex, norm='ortho')

            # x_mag = torch.abs(x_normalized_complex)
            # x_log_power = torch.log(x_mag ** 2 + 1e-8)
            # x_fft_mag = torch.abs(x_fft_complex)
            # x_fft_log_power = torch.log(x_fft_mag ** 2 + 1e-8)

            x_model_input = torch.stack([
                x_normalized_complex.real,
                x_normalized_complex.imag,
                # x_mag,
                # x_log_power,
                x_fft_complex.real,
                x_fft_complex.imag,
                # x_fft_mag,
                # x_fft_log_power
            ], dim=1).float()
            x_normalized_for_sigma = torch.view_as_complex(
                torch.stack([x_normalized_complex.real, x_normalized_complex.imag], dim=1).permute(0, 2, 3, 1).contiguous()
            )
            U_pred, V_pred = model(x_model_input)
            S_norm = analytic_sigma(U_pred, V_pred, x_normalized_for_sigma)
            S_pred = S_norm * fro_norm.unsqueeze(1)
            loss = ae_loss(U_pred, S_pred, V_pred, H_label)
            b = x_raw.size(0)
            total_loss += loss.item() * b
            num_samples += b

    mean_loss = total_loss / num_samples

    print(f"Validation Loss: {mean_loss:.4f}")
    return mean_loss


if __name__ == "__main__":
    DATA_DIR = './CompetitionData2'
    # MODEL_PATH = './model/model_epoch_422.pth'
    MODEL_PATH = './svd_best_multi.pth'
    OUTPUT_DIR = './outputs'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    M, N, r = 128, 128, 64

    model = load_model(MODEL_PATH, M, N, r)

    for idx in range(1, 5):
        data_path = os.path.join(DATA_DIR, f'Round2TrainData{idx}.npy')
        label_path = os.path.join(DATA_DIR, f'Round2TrainLabel{idx}.npy')

        train_dataset = SimpleChannelDataset(
            data_path=data_path,
            label_path=label_path
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        mean_loss = validate_model(model, train_dataloader, DEVICE)

        test_path = os.path.join(DATA_DIR, f'Round2TestData{idx}.npy')
        test_dataset = SimpleChannelDataset(
            data_path=test_path,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )


        process_and_export(model, test_dataloader, OUTPUT_DIR, idx)
