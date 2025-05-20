import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

import MinkowskiEngine as ME

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.funcs import *
from dataset.dataset import XYProjectionFASERCALDataset
from model.networkAttention import MinkEncClsConvNeXtV2


# === Plotting function ===
def save_loss_plot(train_losses, val_losses, folder="plots", version=1):
    filename = os.path.join(folder, f"v{version}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"[Epoch {len(train_losses)}] Saved loss plot to {filename}")


# === Optional: attention visualization utilities ===
def plot_token_strength(tokens, title="Token Norms First Batch Event"):
    plt.figure(figsize=(12, 10))
    aver_chan = torch.max(tokens, dim=2)[0].cpu().numpy()

    ax1 = plt.subplot(2, 1, 1)
    im = ax1.imshow(aver_chan, aspect='auto')
    ax1.set_title(title)
    ax1.set_xlabel("Token Index")
    ax1.set_ylabel("Event Index")
    plt.colorbar(im, ax=ax1)

    x = np.arange(aver_chan.shape[1])
    y = aver_chan[0]
    ax2 = plt.subplot(2, 1, 2)
    ax2.scatter(x, y, c='red')
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("Token Strength (First Event)")
    ax2.set_title("Token Strengths - First Event (Scatter)")

    plt.tight_layout()
    plt.savefig('token_strength.png')
    plt.close()


def plot_and_save_attention(attn_weights, filename="attention.png", layer_idx=1, step=0):
    attn = attn_weights[0]
    attn = attn.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(attn, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.title(f'Attention Weights - Layer {layer_idx}, {attn.shape}, Step {step}')
    plt.savefig(filename)
    plt.close()


# === Training Setup ===
args = SimpleNamespace()
args.dataset_path = "/scratch/salonso/sparse-nns/faser/events_v5.1"
args.sets_path = "/scratch/salonso/sparse-nns/faser/events_v5.1/sets.pkl"
args.batch_size = 32
args.num_workers = 12
args.augmentations_enabled = False
args.train = True

version = 2

dataset = XYProjectionFASERCALDataset(args)
print("- Dataset size: {} events".format(len(dataset)))
train_loader, valid_loader, test_loader = split_dataset(dataset, args, splits=[0.6, 0.1, 0.3])

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
num_epochs = 50
best_val_loss = float('inf')

train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch_input, batch_input_global = arrange_sparse_minkowski(batch, device)
        target = arrange_truth(batch)
        y = target["flavour_label"].to(device)

        optimizer.zero_grad()
        out = model(batch_input, batch_input_global)
        out["out_flavour"] = out["out_flavour"].to(device)
        loss = loss_fn(out["out_flavour"], y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # token = model.offset_attn.layer1.input_tokens
        # plot_token_strength(token, title=f"Token Norms First Batch Event - Epoch {epoch}")


    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch}"):
            batch_input, batch_input_global = arrange_sparse_minkowski(batch, device)
            target = arrange_truth(batch)
            y = target["flavour_label"].to(device)

            out = model(batch_input, batch_input_global)
            out["out_flavour"] = out["out_flavour"].to(device)
            loss = loss_fn(out["out_flavour"], y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"Validation loss improved, saving model for epoch {epoch}...")
        torch.save(model.state_dict(), "best_model.pth")

    # === Save loss plot ===
    save_loss_plot(train_losses, val_losses)

print("Training complete!")
