import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

def parse_cfg(cfg_path: str) -> Tuple[int, int, int, int, int]:
    with open(cfg_path, 'r') as f:
        params = [int(line.strip()) for line in f if line.strip()]
    return tuple(params)

class ChannelDataset(Dataset):
    def __init__(self, data_path: str = None, label_path: str = None, cfg_path: str = None, 
                 preloaded_data: np.ndarray = None, preloaded_labels: np.ndarray = None):
        
        if preloaded_data is not None:
            self.data = preloaded_data
            self.labels = preloaded_labels
        elif data_path is not None:
            self.data = np.load(data_path)
            self.labels = np.load(label_path) if label_path else None
        else:
            raise ValueError("Must provide either data_path or preloaded_data.")

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

    def __len__(self) -> int:
        return self.data_complex.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # data_sample = self.data_complex[idx]
        data_sample = self.data[idx]
        if self.labels_complex is not None:
            label_sample = self.labels_complex[idx]
            # label_sample = self.labels[idx]
            return data_sample, label_sample
        else:
            return data_sample, data_sample