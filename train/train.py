import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.optim as optim
from tqdm import tqdm
from utils.funcs import *
from types import SimpleNamespace
from dataset.dataset import XYProjectionFASERCALDataset
# from model.networkAttention import MinkEncClsConvNeXtV2  # adjust path if needed
from model.network import MinkEncClsConvNeXtV2  # adjust path if needed



args = SimpleNamespace()

# Initialize necessary fields for args
args.dataset_path = "/scratch/salonso/sparse-nns/faser/events_v5.1"
args.sets_path = "/scratch/salonso/sparse-nns/faser/events_v5.1/sets.pkl"
args.batch_size = 32
args.num_workers = 12
args.augmentations_enabled = False  
args.train = True

# Init dataset
dataset = XYProjectionFASERCALDataset(args)
print("- Dataset size: {} events".format(len(dataset)))

train_loader, valid_loader, test_loader = split_dataset(dataset, args, splits=[0.6, 0.1, 0.3]) 

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MinkEncClsConvNeXtV2(in_channels=1, out_channels=4, args=args).to(device)

# Loss
loss_fn = nn.CrossEntropyLoss()

# Optimizer with fixed learning rate
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,  # Fixed learning rate
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)

num_epochs = 50
best_val_loss = float('inf') 

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch_input, batch_input_global = arrange_sparse_minkowski(batch, device)
        target = arrange_truth(batch)
        y = target["flavour_label"]
        y = y.to(device) 

        optimizer.zero_grad()
        out = model(batch_input, batch_input_global)
        out["out_flavour"] = out["out_flavour"].to(device)


        # Now compute the loss
        loss = loss_fn(out["out_flavour"], y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():  # No need to compute gradients during validation
        for batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch}"):
            batch_input, batch_input_global = arrange_sparse_minkowski(batch, device)
            target = arrange_truth(batch)
            y = target["flavour_label"]
            y = y.to(device)

            out = model(batch_input, batch_input_global)

            # Ensure the model's output is on the correct device
            out["out_flavour"] = out["out_flavour"].to(device)

            # Now compute the loss
            loss = loss_fn(out["out_flavour"], y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_loader)
    print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")

    # Save model if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"Validation loss improved, saving model for epoch {epoch}...")
        torch.save(model.state_dict(), "best_model.pth")  # Save the model checkpoint

print("Training complete!")
