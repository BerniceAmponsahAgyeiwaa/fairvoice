#!/usr/bin/env python3
"""
Stable oversampling training script for FairVoice
- fixes import path so `src/` packages are found
- computes inverse-frequency per-sample weights by chosen group column
- creates a WeightedRandomSampler and DataLoader
- supports datasets that return either tuples (feat, label, group, idx)
  or dicts {"audio":..., "emotion":..., "group":..., "idx":...}
- minimal robust PyTorch training loop (EPOCHS default 1)
- saves checkpoint at the end
"""

from pathlib import Path
import sys
import time
import json

# -----------------------------------------------------------------------------
# Make project src importable (assumes script is at src/training/*.py)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
print("Loaded src from:", SRC)

# -----------------------------------------------------------------------------
# Standard imports
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Project imports (your modules under src/)
# -----------------------------------------------------------------------------
from datasets.crema_dataset import CremaFeatureDataset
from model.baseline_cnn import BaselineCNN

# -----------------------------------------------------------------------------
# Paths / config
# -----------------------------------------------------------------------------
DATA_FP = ROOT / "data/processed/metadata_train.csv"
SAVE_DIR = ROOT / "models/oversampled"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 1
NUM_WORKERS = 0  # macOS safe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROUP_COL_CANDIDATES = ["Sex", "sex", "Gender", "gender"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def detect_group_col(df: pd.DataFrame):
    for c in GROUP_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def compute_inverse_freq_weights(df: pd.DataFrame, group_col: str):
    """
    Return a list of per-sample weights (float) computed as inverse frequency of the group.
    """
    counts = df[group_col].value_counts().to_dict()
    inv = {k: 1.0 / counts[k] for k in counts}
    # map each row to its group's inverse frequency
    weights = df[group_col].map(inv).astype(float).tolist()
    return weights

def _stack_audios(audios):
    """
    audios: list of torch tensors. Each tensor may be:
      - (1, n_mels, T)   -> keep as is
      - (n_mels, T)      -> add channel dim -> (1, n_mels, T)
      - (L,) or (1,L)    -> convert to (1,1,L)
    Returns stacked tensor: (B, C, F, T) or (B, C, L)
    """
    processed = []
    for a in audios:
        if a.ndim == 3:
            # (1, n_mels, T) or (C, F, T)
            processed.append(a)
        elif a.ndim == 2:
            # (n_mels, T) -> (1, n_mels, T)
            processed.append(a.unsqueeze(0))
        elif a.ndim == 1:
            # (L,) -> (1, 1, L)
            processed.append(a.unsqueeze(0).unsqueeze(0))
        else:
            # unexpected dims -> try to convert to float tensor and keep
            processed.append(a.reshape(1, *a.shape[-2:]) if a.ndim>=2 else a.unsqueeze(0))
    try:
        stacked = torch.stack(processed, dim=0)
    except Exception as e:
        # Provide helpful debug
        shapes = [tuple(x.shape) for x in processed]
        raise RuntimeError(f"Failed to stack audio tensors; individual shapes: {shapes}. Error: {e}")
    return stacked

def pad_or_truncate(tensor, max_len):
    """
    tensor: (1, 64, T)
    Returns tensor padded or truncated to T=max_len
    """
    _, _, T = tensor.shape
    if T == max_len:
        return tensor
    elif T < max_len:
        # pad right
        pad_amount = max_len - T
        return torch.nn.functional.pad(tensor, (0, pad_amount))
    else:
        # truncate
        return tensor[:, :, :max_len]


def collate_batch(batch):
    # batch = list of tuples (logmel, emotion, group, idx)

    audios = [item[0] for item in batch]    # logmels
    emotions = [item[1] for item in batch]
    groups = [item[2] for item in batch]
    idxs = [item[3] for item in batch]

    # pad mel spectrograms (time dimension)
    lengths = [a.shape[-1] for a in audios]
    max_len = max(lengths)

    padded = []
    for mel in audios:
        pad_amt = max_len - mel.shape[-1]
        if pad_amt > 0:
            mel = torch.nn.functional.pad(mel, (0, pad_amt))  # pad time dimension
        padded.append(mel)

    audios = torch.stack(padded, dim=0)   # (B, 1, 64, T)

    emotions = torch.tensor(emotions, dtype=torch.long)
    groups = torch.tensor(groups, dtype=torch.long)

    return audios, emotions, groups, idxs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Loading metadata:", DATA_FP)
    df = pd.read_csv(DATA_FP)

    group_col = detect_group_col(df)
    if group_col is None:
        raise RuntimeError(f"No group column found in metadata. Tried: {GROUP_COL_CANDIDATES}. Columns: {list(df.columns)}")
    print("Using group column:", group_col)

    # compute weights and sampler
    weights = compute_inverse_freq_weights(df, group_col)
    assert len(weights) == len(df), "weights length mismatch"

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True
    )

    # dataset initialization (expects metadata dataframe)
    dataset = CremaFeatureDataset(metadata=df, group_column=group_col)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        sampler=sampler,
                        num_workers=NUM_WORKERS,
                        collate_fn=collate_batch)

    # model
    num_classes = int(df["emotion"].nunique())
    model = BaselineCNN(num_classes=num_classes).to(DEVICE)
    print(f"Model -> {DEVICE} | num_classes={num_classes}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # train loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        n_seen = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for audios, emotions, groups, idxs in pbar:
            # audios -> (B, C, F, T) expected by BaselineCNN which uses conv2d
            audios = audios.to(DEVICE)
            emotions = emotions.to(DEVICE)

            # If audios are waveforms (B, 1, L) and BaselineCNN expects (B,1,F,T),
            # this will still call model but may be suboptimal. Prefer precomputed logmels.
            try:
                logits = model(audios)
            except Exception as e:
                # attempt a fallback reshape/unsqueeze if channels/time dims mismatch
                # This is a best-effort fallback and will print debug info.
                print("Model forward failed with shape:", tuple(audios.shape), " — attempt fallback reshape.")
                if audios.ndim == 3:
                    # (B, 1, L) -> add fake freq dim: (B,1,1,L)
                    audios = audios.unsqueeze(2)
                elif audios.ndim == 4 and audios.shape[2] == 1:
                    pass  # okay
                logits = model(audios)

            loss = criterion(logits, emotions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss * audios.size(0)
            n_seen += audios.size(0)
            pbar.set_postfix(loss=batch_loss, shape=tuple(audios.shape))

        epoch_loss = running_loss / max(1, n_seen)
        print(f"Epoch {epoch} finished — loss: {epoch_loss:.4f}")

    # save checkpoint
    ckpt = SAVE_DIR / "oversampled_checkpoint.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "params": {"batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS},
        "metadata_path": str(DATA_FP)
    }, ckpt)
    print("Saved checkpoint to:", ckpt)
    print("Done. Time elapsed: {:.1f}s".format(time.time() - t0))

if __name__ == "__main__":
    main()
