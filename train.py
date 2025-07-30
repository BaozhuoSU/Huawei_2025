import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
import os
import numpy as np

from augment_dataset import AugmentChannelDataset
from model import EfficientSVDNet
from dataset import ChannelDataset, parse_cfg
from loss import LAELoss
from datetime import datetime

from model_profiler import get_avg_flops

DATA_DIR = "./CompetitionData1"
best_model_path = None
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f"model/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_log.txt')

SCENARIOS = ["1", "2", "3"]
BATCH_SIZE = 512
LEARNING_RATE = 2e-4
EPOCHS = 50
VALIDATION_SPLIT = 0.1
WEIGHT_DECAY = 1e-4

def build_full_dataset(scenarios):
    
    datasets = []
    
    first_cfg_path = os.path.join(DATA_DIR, f"Round1CfgData{scenarios[0]}.txt")
    _, M, N, _, r = parse_cfg(first_cfg_path)
    for s_id in scenarios:
        data_path = os.path.join(DATA_DIR, f"Round1TrainData{s_id}.npy")
        label_path = os.path.join(DATA_DIR, f"Round1TrainLabel{s_id}.npy")
        cfg_path = os.path.join(DATA_DIR, f"Round1CfgData{s_id}.txt")
        datasets.append(AugmentChannelDataset(data_path=data_path, label_path=label_path, cfg_path=cfg_path))

    full_dataset = ConcatDataset(datasets)
    return full_dataset, (M, N, r)

def train_full_dataset(device):

    full_dataset, (M, N, r) = build_full_dataset(SCENARIOS)
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    model = EfficientSVDNet(M=M, N=N, r=r).to(device)
    criterion = LAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    
    best_val_loss = float('inf')

    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(
            f"Batch sizes: Train={BATCH_SIZE}\n Learning rate: {LEARNING_RATE}\n weight decay: {WEIGHT_DECAY}\n Epochs: {EPOCHS}\n Validation split: {VALIDATION_SPLIT}\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for H_in, H_label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            H_in, H_label = H_in.to(device), H_label.to(device)
            
            optimizer.zero_grad()
            U_pred, S_pred, V_pred = model(H_in)
            loss = criterion(U_pred, S_pred, V_pred, H_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for H_in, H_label in val_loader:
                H_in, H_label = H_in.to(device), H_label.to(device)
                U_pred, S_pred, V_pred = model(H_in)
                loss = criterion(U_pred, S_pred, V_pred, H_label)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss : {avg_train_loss:.6f} Val Loss: {avg_val_loss:.6f}")

        model_path = os.path.join(log_dir, f"model_epoch_{epoch + 1}.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)

        with open(log_file, 'a') as f:
            f.write(
                f"Epoch {epoch+1}: Train Loss : {avg_train_loss:.6f} Val Loss: {avg_val_loss:.6f}\n")


    print(f"Training finished! Best validation loss: {best_val_loss:.6f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_full_dataset(device)

if __name__ == "__main__":
    main()
