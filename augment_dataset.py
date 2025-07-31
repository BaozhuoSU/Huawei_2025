import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


def parse_cfg(cfg_path: str) -> Tuple[int, int, int, int, int]:
    with open(cfg_path, 'r') as f:
        params = [int(line.strip()) for line in f if line.strip()]
    return tuple(params)


class AugmentChannelDataset(Dataset):
    def __init__(self,
                 data_path: str = None,
                 label_path: str = None,
                 cfg_path: str = None,
                 is_augment: bool = True,
                 augment_noise_std: float = 0.005,
                 augment_noise_prob: float = 0.5,
                 augment_phase_range: float = 0.05,
                 augment_phase_prob: float = 0.5,
                 augment_scale_range: Tuple[float, float] = (0.9, 1.1),
                 augment_scale_prob: float = 0.5,
                 augment_dropout_prob: float = 0.05,
                 augment_dropout_apply_prob: float = 0.5):
        """
        Args:
            data_path: Path to data .npy file
            label_path: Path to labels .npy file
            cfg_path: Path to config file with nsample, M, N, Q, r
            augment_noise_std: Standard deviation for Gaussian noise
            augment_noise_prob: Probability of applying noise augmentation
            augment_phase_range: Range for random phase rotation (in radians)
            augment_phase_prob: Probability of applying phase rotation
            augment_scale_range: Range for random amplitude scaling
            augment_scale_prob: Probability of applying amplitude scaling
            augment_dropout_prob: Probability of dropping antennas
            augment_dropout_apply_prob: Probability of applying antenna dropout
        """

        if data_path is not None:
            self.data = np.load(data_path)
            self.labels = np.load(label_path) if label_path else None
        else:
            raise ValueError("Must provide data_path.")

        if self.data.shape[-1] != 2:
            raise ValueError("The last dimension of the data must be 2 (real and imaginary parts).")

        self.data_complex = torch.view_as_complex(torch.from_numpy(self.data).float())

        if self.labels is not None:
            self.labels_complex = torch.view_as_complex(torch.from_numpy(self.labels).float())
        else:
            self.labels_complex = None

        if cfg_path:
            self.nsample, self.M, self.N, self.Q, self.r = parse_cfg(cfg_path)
        else:
            self.nsample, self.M, self.N = self.data_complex.shape
            self.r = -1

        self.is_augment = is_augment
        self.augment_noise_std = augment_noise_std
        self.augment_noise_prob = augment_noise_prob
        self.augment_phase_range = augment_phase_range
        self.augment_phase_prob = augment_phase_prob
        self.augment_scale_range = augment_scale_range
        self.augment_scale_prob = augment_scale_prob
        self.augment_dropout_prob = augment_dropout_prob
        self.augment_dropout_apply_prob = augment_dropout_apply_prob

    def __len__(self) -> int:
        return self.data_complex.shape[0]

    def _augment(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to a single complex-valued sample with probabilities.
        Args:
            data: Complex tensor of shape (M, N)
        Returns:
            Augmented complex tensor of shape (M, N)
        """

        data_complex = data

        # 1. noise injection
        if self.augment_noise_std > 0 and torch.rand(1).item() < self.augment_noise_prob:
            noise = torch.randn_like(data_complex) * self.augment_noise_std
            data_complex = data_complex + noise

        # 2. phase augmentation
        if self.augment_phase_range > 0 and torch.rand(1).item() < self.augment_phase_prob:
            phase = torch.rand(1, device=data_complex.device) * 2 * self.augment_phase_range - self.augment_phase_range
            rotation = torch.exp(1j * phase)
            data_complex = data_complex * rotation

        # 3. amplitude scaling
        if self.augment_scale_range is not None and torch.rand(1).item() < self.augment_scale_prob:
            scale = torch.rand(1, device=data_complex.device) * (
                        self.augment_scale_range[1] - self.augment_scale_range[0]) + self.augment_scale_range[0]
            data_complex = data_complex * scale

        # 4. Antenna subset selection (randomly dropping rows or columns)
        if self.augment_dropout_prob > 0 and torch.rand(1).item() < self.augment_dropout_apply_prob:
            row_mask = torch.rand(self.M, device=data_complex.device) > self.augment_dropout_prob
            col_mask = torch.rand(self.N, device=data_complex.device) > self.augment_dropout_prob
            augmented_data = data_complex * row_mask.view(-1, 1) * col_mask.view(1, -1)
            # Check matrix rank
            rank = torch.linalg.matrix_rank(augmented_data)
            if rank >= min(self.M, self.N) // 2:  # Ensure sufficient rank
                data_complex = augmented_data

        return data_complex

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        data_sample = self.data_complex[idx]
        label_sample = self.labels_complex[idx]

        if self.is_augment:
            data_sample = self._augment(data_sample)

        data_sample = torch.view_as_real(data_sample).float()

        return data_sample, label_sample
