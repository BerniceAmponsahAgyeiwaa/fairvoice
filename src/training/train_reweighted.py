# src/training/train_reweighted.py
"""
Optimized reweighted training script (multiprocessing-safe).

Expectations:
 - src/datasets/crema_dataset.py defines CremaFeatureDataset and returns
   (logmel: Tensor[1,n_mels,T], emotion: Tensor, group: Tensor, idx: int)
 - src/model/baseline_cnn.py defines BaselineCNN
 - metadata CSV: data/processed/metadata_train.csv (we read into pandas and pass df to dataset)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# -------------------------
# Make src/ importable
# -------------------------
ROOT = Path(__file__).resolve().parents[2]   # project root (two parents up)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

print("Loaded src from:", SRC)

# -------------------------
# Project imports (top-level modules only)
# -------------------------
from datasets.crema_dataset import CremaFeatureDataset, EMOTION_MAP
from model.baseline_cnn import BaselineCNN

# -------------------------
# Config / Hyperparams
# -------------------------
METADATA_FP = ROOT / "data" / "processed" / "metadata_train.csv"
GROUP_COL_CANDIDATES = ["Sex", "sex", "Gender", "gender"]
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 1           # start small; increase as needed
NUM_WORKERS = min(4, (os.cpu_count() or 4))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# -------------------------
# Helpers
# -------------------------
def detect_group_col(df):
    for c in GROUP_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def compute_sample_weights_inverse_freq(df, group_col):
    """
    Compute per-sample weights = (total / count[group]) for each sample's group.
    Returns a numpy float array of length len(df).
    """
    counts = df[group_col].value_counts().to_dict()
    total = len(df)
    weight_map = {g: float(total) / float(c) for g, c in counts.items()}
    weights = df[group_col].map(weight_map).to_numpy(dtype=float)
    return weights

# -------------------------
# Pickle-safe collate (top-level class)
# -------------------------
class CollateFn:
    """
    Pickle-safe collate callable. Stores sample_weights as a numpy array.
    When called with a batch (list of tuples returned by dataset),
    pads logmel tensors in time dimension and returns:
      audios (Tensor[B, 1, n_mels, T]), emotions (Tensor[B]), groups (Tensor[B]),
      weights (Tensor[B]), idxs (Tensor[B], long)
    """
    def __init__(self, sample_weights: np.ndarray):
        # store as numpy array (pickle-friendly)
        self.sample_weights = np.asarray(sample_weights, dtype=float)

    def __call__(self, batch):
        # batch: list of (logmel, emotion, group, idx)
        audios, emos, groups, idxs = zip(*batch)

        # ensure each audio is a torch.Tensor
        # audios are expected as Tensor shape (1, n_mels, T)
        lengths = [int(a.shape[-1]) for a in audios]
        max_len = max(lengths)

        padded = []
        for a in audios:
            # pad along time dimension (last dim)
            pad_len = max_len - a.shape[-1]
            if pad_len > 0:
                a = nn.functional.pad(a, (0, pad_len))
            padded.append(a)

        audios_tensor = torch.stack(padded, dim=0)      # (B, 1, n_mels, T)
        emos_tensor = torch.stack(emos).view(-1)        # (B,)
        groups_tensor = torch.stack(groups).view(-1)    # (B,)
        idxs_array = np.array(idxs, dtype=int)
        weights_arr = self.sample_weights[idxs_array]
        weights_tensor = torch.from_numpy(weights_arr).float()

        idxs_tensor = torch.from_numpy(idxs_array).long()

        return audios_tensor, emos_tensor, groups_tensor, weights_tensor, idxs_tensor

# -------------------------
# Training functions
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="Training", ncols=120)
    for audios, emos, groups, weights, idxs in pbar:
        audios = audios.to(device)
        emos = emos.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()
        logits = model(audios)                      # model should accept (B,1,n_mels,T)
        per_sample_loss = criterion(logits, emos)   # shape (B,)
        loss = (per_sample_loss * weights).mean()   # weighted mean

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "shape": tuple(audios.shape)})

    avg_loss = running_loss / max(1, n_batches)
    return avg_loss

# -------------------------
# Main
# -------------------------
def main():
    # Load metadata
    print("Loading metadata:", METADATA_FP)
    df = pd.read_csv(METADATA_FP)

    group_col = detect_group_col(df)
    if group_col is None:
        raise RuntimeError(f"No group column found. Tried: {GROUP_COL_CANDIDATES}. Columns: {list(df.columns)}")
    print("Using group column:", group_col)

    # Compute per-sample weights (inverse-frequency)
    sample_weights = compute_sample_weights_inverse_freq(df, group_col)
    assert len(sample_weights) == len(df), "sample_weights length mismatch"

    # Build dataset (CremaFeatureDataset takes DataFrame and group_column)
    dataset = CremaFeatureDataset(metadata=df, group_column=group_col)
    collate_fn = CollateFn(sample_weights)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=False
    )

    # Build model / optimizer / loss
    num_classes = int(df["emotion"].nunique())
    model = BaselineCNN(num_classes=num_classes).to(DEVICE)
    print(f"Model -> {DEVICE} | num_classes={num_classes}")
    criterion = nn.CrossEntropyLoss(reduction="none")  # per-sample
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt_dir = ROOT / "models" / "reweighted"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "reweighted_checkpoint.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "sample_weights_meta": {"method": "inverse_freq", "group_col": group_col},
        "metadata_path": str(METADATA_FP)
    }, ckpt_path)
    print("Saved checkpoint to:", ckpt_path)
    print("Done. Time elapsed: {:.1f}s".format(time.time() - t0))


# -------------------------
# Entrypoint (spawn-safe)
# -------------------------
if __name__ == "__main__":
    # Multiprocessing on macOS requires the __main__ guard.
    main()
