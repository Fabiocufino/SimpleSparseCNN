import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

import MinkowskiEngine as ME

# === Paths and TensorBoard ===
def get_new_log_dir(base_dir="tb_logs", prefix="v"):
    """Automatically create a new subfolder like tb_logs/v1/"""
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix) and d[len(prefix):].isdigit()]
    version = 1 + max([int(d[len(prefix):]) for d in existing], default=0)
    new_log_dir = os.path.join(base_dir, f"{prefix}{version}")
    os.makedirs(new_log_dir, exist_ok=True)
    return new_log_dir

LOG_DIR = get_new_log_dir()
writer = SummaryWriter(log_dir=LOG_DIR)
print(f"TensorBoard logs will be written to: {LOG_DIR}")

# === Import custom modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.funcs import *
from dataset.dataset import XYProjectionFASERCALDataset
from model.networkAttentionSaul import MinkEncClsConvNeXtV2


# === Configurations ===
args = SimpleNamespace(
    dataset_path="/scratch/salonso/sparse-nns/faser/events_v5.1",
    sets_path="/scratch/salonso/sparse-nns/faser/events_v5.1/sets.pkl",
    batch_size=32,
    num_workers=12,
    augmentations_enabled=False,
    train=True
)

version = 2
num_epochs = 50

# === Data ===
dataset = XYProjectionFASERCALDataset(args)
print(f"- Dataset size: {len(dataset)} events")
train_loader, valid_loader, test_loader = split_dataset(dataset, args, splits=[0.6, 0.1, 0.3])

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MinkEncClsConvNeXtV2(in_channels=1, out_channels=4, args=args).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)

# === Training Loop ===
best_val_loss = float('inf')
train_losses = []
val_losses = []
step_tr = 0
step_val = 0

for epoch in range(1, num_epochs + 1):
    # === Training ===
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
        step_tr += 1
        batch_input, batch_input_global = arrange_sparse_minkowski(batch, device)
        target = arrange_truth(batch)
        y = target["flavour_label"].to(device)

        optimizer.zero_grad()
        out = model(batch_input, batch_input_global)
        loss = loss_fn(out["out_flavour"], y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), step_tr)

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Epoch {epoch} - Validation"):
            step_val += 1
            batch_input, batch_input_global = arrange_sparse_minkowski(batch, device)
            target = arrange_truth(batch)
            y = target["flavour_label"].to(device)

            out = model(batch_input, batch_input_global)
            loss = loss_fn(out["out_flavour"], y)
            val_loss += loss.item()

            writer.add_scalar("Loss/val", loss.item(), step_val)

    avg_val_loss = val_loss / len(valid_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")

    # === Save Best Model ===
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"Validation loss improved, saving model (epoch {epoch})...")
        torch.save(model.state_dict(), "best_model.pth")

writer.flush()
writer.close()

print("Training complete!")
