import numpy as np
import torch
import os
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset


class MIMODataset(Dataset):
    """
    A PyTorch Dataset for loading MIMO channel data, labels, test data, and configurations.

    Args:
        data_dir (str): Directory containing the CompetitionData1 files.
        mode (str): Either 'train' or 'test' to specify the data to load.
        round_idx (int): Index of the round to load (0, 1, or 2 for Round1TrainData1-3).

    Attributes:
        train_data (List[np.ndarray]): List of training data arrays.
        train_labels (List[np.ndarray]): List of training label arrays.
        test_data (List[np.ndarray]): List of test data arrays.
        configs (List[Dict]): List of configuration dictionaries.
        mode (str): Dataset mode ('train' or 'test').
        round_idx (int): Selected round index.
        data (np.ndarray): Selected data array for the specified round.
        labels (np.ndarray): Selected label array (if mode='train', else None).
        config (Dict): Configuration dictionary for the round.
    """

    def __init__(self, data_dir: str, mode: str = 'train', round_idx: int = 0):
        if mode not in ['train', 'test']:
            raise ValueError("Mode must be 'train' or 'test'")
        if round_idx not in [0, 1, 2]:
            raise ValueError("round_idx must be 0, 1, or 2")

        self.mode = mode
        self.round_idx = round_idx
        self.train_data, self.train_labels, self.test_data, self.configs = self._load_data(data_dir)

        self.data = self.train_data[round_idx] if mode == 'train' else self.test_data[round_idx]
        self.labels = self.train_labels[round_idx] if mode == 'train' else None
        self.config = self.configs[round_idx]

        expected_shape = (self.config['N_sample'], self.config['M'], self.config['N'], self.config['Q'])
        if self.data.shape != expected_shape:
            raise ValueError(f"Data for round {round_idx + 1} has shape {self.data.shape}, expected {expected_shape}")

    def _load_data(self, data_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Loads MIMO channel data, labels, test data, and configurations from the specified directory.

        Args:
            data_dir (str): Directory containing the CompetitionData1 files.

        Returns:
            Tuple containing:
            - List of training data arrays (Round1TrainData1-3)
            - List of training label arrays (Round1TrainLabel1-3)
            - List of test data arrays (Round1TestData1-3)
            - List of configuration dictionaries (Round1CfgData1-3)

        Raises:
            FileNotFoundError: If any expected file is missing.
            ValueError: If data format is invalid or config file content is incorrect.
        """
        train_data = []
        train_labels = []
        test_data = []
        configs = []

        train_data_files = [f"Round1TrainData{i}.npy" for i in range(1, 4)]
        train_label_files = [f"Round1TrainLabel{i}.npy" for i in range(1, 4)]
        test_data_files = [f"Round1TestData{i}.npy" for i in range(1, 4)]
        config_files = [f"Round1CfgData{i}.txt" for i in range(1, 4)]

        for file in train_data_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Training data file not found: {file_path}")
            data = np.load(file_path)
            train_data.append(data)

        for file in train_label_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Training label file not found: {file_path}")
            labels = np.load(file_path)
            train_labels.append(labels)

        for file in test_data_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Test data file not found: {file_path}")
            data = np.load(file_path)
            test_data.append(data)

        for file in config_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Config file not found: {file_path}")
            with open(file_path, 'r') as f:
                lines = f.readlines()
                try:
                    config = {
                        'N_sample': int(lines[0].strip()),
                        'M': int(lines[1].strip()),
                        'N': int(lines[2].strip()),
                        'Q': int(lines[3].strip()),
                        'r': int(lines[4].strip())
                    }
                except ValueError as e:
                    raise ValueError(f"Invalid format in config file {file}: all lines must be integers")
            configs.append(config)

        return train_data, train_labels, test_data, configs

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple containing:
            - data (torch.Tensor): Complex-valued channel matrix of shape (M, N).
            - labels (torch.Tensor or None): Corresponding labels if mode='train', else None.
            - config (Dict): Configuration dictionary for the round.
        """
        data = torch.from_numpy(self.data[idx]).to(dtype=torch.float32)
        labels = torch.from_numpy(self.labels[idx]).to(dtype=torch.float32) if self.mode == 'train' else None
        return data, labels, self.config


def create_train_val_datasets(
        data_dir: str,
        round_idx: List[int] = [1, 2, 3],
        val_split: float = 0.2,
        batch_size: int = 32,
        shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation DataLoaders from the training dataset.

    Args:
        data_dir (str): Directory containing the CompetitionData1 files.
        round_idx (int): Index of the round to load (0, 1, or 2).
        val_split (float): Proportion of training data to use for validation (between 0 and 1).
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the training data.

    Returns:
        Tuple containing:
        - train_loader (DataLoader): DataLoader for the training data.
        - val_loader (DataLoader): DataLoader for the validation data.

    Raises:
        ValueError: If val_split is invalid or the dataset is empty.
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")

    # Initialize the training dataset
    datasets = []
    for idx in round_idx:
        dataset = MIMODataset(data_dir, mode='train', round_idx=idx-1)
        datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)
    total_size = len(combined_dataset)

    # Calculate sizes for training and validation sets
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    if val_size == 0 or train_size == 0:
        raise ValueError("Train or validation dataset is empty due to split ratio")

    # Randomly split the dataset
    train_subset, val_subset = random_split(combined_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    data_dir = "CompetitionData1"
    try:
        # Create training and validation DataLoaders (using 20% of data for validation)
        train_loader, val_loader = create_train_val_datasets(
            data_dir=data_dir,
            round_idx=[1, 2, 3],
            val_split=0.2,
            batch_size=32,
            shuffle=True
        )

        # Verify dataset sizes
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        print(f"Combined training dataset size: {len(train_dataset)}")
        print(f"Combined validation dataset size: {len(val_dataset)}")

        # # Print data and labels for one batch from each round
        # for round_idx in [1, 2, 3]:
        #     print(f"\nRound {round_idx}:")
        #     dataset = MIMODataset(data_dir, mode='train', round_idx=round_idx-1)  # 0-based
        #     loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        #     data, labels, config = next(iter(loader))
        #     print(f"Config: {config}")
        #     print(f"Data shape: {data.shape}, dtype: {data.dtype}")
        #     print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
        #     print(f"Data sample (first element):\n{data[0]}")
        #     print(f"Labels sample (first element):\n{labels[0]}")


        data, labels, _ = next(iter(train_loader))
        print(data.shape)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")