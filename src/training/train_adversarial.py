#!/usr/bin/env python3
"""
Adversarial training script for FairVoice (CREMA-D).
- Place in: src/training/train_adversarial.py
- Run from project root: python src/training/train_adversarial.py
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import List

# -----------------------
# make src importable
# -----------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
print("Loaded src from:", SRC)

# -----------------------
# standard imports
# -----------------------
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# -----------------------
# torchaudio (audio loader + transforms)
# -----------------------
import torchaudio

# -----------------------
# Project model import
# -----------------------
# Use your existing BaselineCNN implementation
from model.baseline_cnn import BaselineCNN

# -----------------------
# Config
# -----------------------
METADATA_FP = ROOT / "data" / "processed" / "metadata_train.csv"
# fallback audio dirs commonly present in your repo
RAW_AUDIO_DIR = ROOT / "data" / "raw" / "CREMA-D" / "AudioWAV"
CLEAN_AUDIO_DIR = ROOT / "data" / "processed" / "audio_clean"
OUTPUT_DIR = ROOT / "models" / "adversarial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0           # safe on macOS
LR_MAIN = 1e-4
LR_ADV = 1e-4
EPOCHS = 1                # start small, increase when stable
LAMBDA_ADV = 1.0          # weight of adversarial term when updating encoder
SEED = 42

GROUP_COLUMN_CANDIDATES = ["Sex", "sex", "Gender", "gender"]

# -----------------------
# Utility / reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -----------------------
# Audio loader + log-mel extractor (self-contained)
# -----------------------
# reused transforms to avoid re-creating inside loop repeatedly
TARGET_SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)
db_transform = torchaudio.transforms.AmplitudeToDB()

def load_audio_logmel(candidate_path: str):
    """
    Load audio and convert to log-mel spectrogram tensor of shape (1, n_mels, T).
    Accepts either full path or stem; will attempt sensible fallbacks.
    """
    # Try given path first (string)
    path = str(candidate_path)

    # if path looks like a stem without extension, try add .wav
    tried = []
    if not path.endswith(".wav"):
        tried.append(path)
        path_wav = path + ".wav"
        tried.append(path_wav)
    else:
        tried.append(path)

    # Also try path located under processed clean_dir or raw audio dir
    if not any(Path(p).exists() for p in tried):
        # try as-is under CLEAN_AUDIO_DIR
        base = Path(path).name
        tried.append(str(CLEAN_AUDIO_DIR / base))
        tried.append(str(CLEAN_AUDIO_DIR / (base + ".wav")))
        tried.append(str(RAW_AUDIO_DIR / base))
        tried.append(str(RAW_AUDIO_DIR / (base + ".wav")))

    # find first existing
    final = None
    for p in tried:
        if Path(p).exists():
            final = p
            break

    if final is None:
        raise FileNotFoundError(f"Audio file not found. Tried: {tried}")

    # load
    try:
        wav, sr = torchaudio.load(final)
    except Exception as e:
        # try using soundfile via librosa fallback
        import soundfile as sf
        wav_np, sr = sf.read(final, dtype="float32")
        wav = torch.tensor(wav_np, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

    # unify dims: (channels, samples)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    # make mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    mel = mel_transform(wav)      # (1, n_mels, T)
    logmel = db_transform(mel)    # (1, n_mels, T)
    return logmel

# -----------------------
# Dataset
# -----------------------
class CremaAdversarialDataset(Dataset):
    """
    Expects metadata DataFrame containing at least:
      - a column with a filename/path (we auto-detect common names)
      - emotion (string code like 'HAP','SAD', etc.)
      - group column (Sex etc.)
    Returns: logmel tensor (1,n_mels,T), emotion_idx (int), group_idx (int), idx (int)
    """
    EMOTION_ORDER = ["ANG","DIS","FEA","HAP","NEU","SAD"]  # some mapping present in your project
    EMO_MAP = {k: i for i, k in enumerate(EMOTION_ORDER)}  # e.g., "HAP"->3 in that mapping

    def __init__(self, df: pd.DataFrame, group_col: str = "Sex", filename_candidates: List[str]=None):
        self.df = df.reset_index(drop=True)

        # detect filename column
        if filename_candidates is None:
            filename_candidates = ["audio_path", "clean_path", "file", "filename", "FileName"]
        found = None
        for c in filename_candidates:
            if c in self.df.columns:
                found = c
                break
        if found is None:
            raise RuntimeError(f"No filename column found. Tried: {filename_candidates}. Columns: {list(self.df.columns)}")
        self.filename_col = found

        # detect emotion col
        if "emotion" not in self.df.columns:
            raise RuntimeError("No 'emotion' column found in metadata.")
        # detect group col
        if group_col is None:
            raise RuntimeError("group_col must be provided")
        if group_col not in self.df.columns:
            raise RuntimeError(f"Group column '{group_col}' not present in metadata.")
        self.group_col = group_col

        # canonicalize group mapping (string -> small int)
        unique_groups = sorted(self.df[self.group_col].dropna().unique().tolist())
        # if Sex has 'Male','Female' map accordingly
        if set(unique_groups) <= {"Male","Female","male","female","M","F"}:
            self.group_map = {"Male":0, "male":0, "M":0, "Female":1, "female":1, "F":1}
        else:
            # create deterministic mapping
            self.group_map = {g: i for i, g in enumerate(unique_groups)}

        # build cache of resolved file paths (lazy resolution on demand would also be ok)
        # but we'll leave resolution to load_audio_logmel which has fallback heuristics

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row[self.filename_col]
        emotion_str = row["emotion"]
        group_raw = row[self.group_col]

        # normalize emotion -> integer
        if isinstance(emotion_str, (int, float)) and not np.isnan(emotion_str):
            # if already numeric index
            emo_idx = int(emotion_str)
        else:
            emo_code = str(emotion_str).strip()
            if emo_code not in self.EMO_MAP:
                # as a last resort, try upper-cased first 3 chars
                emo_code = emo_code.upper()[:3]
            if emo_code not in self.EMO_MAP:
                raise RuntimeError(f"Unknown emotion token: {emotion_str}")
            emo_idx = self.EMO_MAP[emo_code]

        # group mapping
        group_idx = self.group_map.get(group_raw, None)
        if group_idx is None:
            # attempt string normalization
            group_idx = self.group_map.get(str(group_raw).strip(), 0)

        # audio path may be a Path object or a stem. Convert to string
        audio_candidate = str(fname)

        # Load log-mel (will try many fallbacks if needed)
        logmel = load_audio_logmel(audio_candidate)  # (1, n_mels, T)

        return logmel, int(emo_idx), int(group_idx), int(idx)

# -----------------------
# Collate (pads time dimension)
# -----------------------
def collate_pad_logmel(batch):
    """
    batch: list of tuples (logmel, emo_idx, group_idx, idx)
      logmel: tensor (1, n_mels, T)
    Returns:
      X: (B, 1, n_mels, T_max)
      y: (B,)
      g: (B,)
      idxs: list
    """
    logmels = [item[0] for item in batch]
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    gs = torch.tensor([item[2] for item in batch], dtype=torch.long)
    idxs = [item[3] for item in batch]

    # find max time
    max_t = max(l.shape[-1] for l in logmels)
    padded = []
    for l in logmels:
        t = l.shape[-1]
        if t < max_t:
            pad = (0, max_t - t)  # (left_pad, right_pad) for last dim
            lpad = F.pad(l, pad)  # F.pad expects (pad_left, pad_right) for last dim
            padded.append(lpad)
        else:
            padded.append(l)
    X = torch.stack(padded, dim=0)  # (B, 1, n_mels, T_max)
    return X, ys, gs, idxs

# -----------------------
# Gradient Reversal (simple)
# -----------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lamb * grad_output, None

def grad_reverse(x, lamb=1.0):
    return GradReverse.apply(x, lamb)

# -----------------------
# Helper: extract CNN features (reuse BaselineCNN layers)
# -----------------------
def extract_features_from_baseline(baseline_model: BaselineCNN, x: torch.Tensor):
    """
    Given baseline_model (instance of BaselineCNN) and input x (B,1,n_mels,T),
    compute the feature vector before fc (B, feature_dim).
    Mirrors the conv/pool stack inside BaselineCNN forward, but stops before fc.
    """
    # conv1
    x = F.relu(baseline_model.bn1(baseline_model.conv1(x)))
    x = baseline_model.pool1(x)
    x = F.relu(baseline_model.bn2(baseline_model.conv2(x)))
    x = baseline_model.pool2(x)
    x = F.relu(baseline_model.bn3(baseline_model.conv3(x)))
    x = baseline_model.pool3(x)          # (B, 64, 1, 1)
    x = x.view(x.size(0), -1)            # (B, 64)
    return x

# -----------------------
# Training: adversarial steps
# -----------------------
def train_adversarial(
    model: BaselineCNN,
    adv_head: nn.Module,
    loader: DataLoader,
    opt_main: torch.optim.Optimizer,
    opt_adv: torch.optim.Optimizer,
    criterion_main,
    criterion_adv,
    device: str = "cpu",
    lambda_adv: float = 1.0,
):
    model.train()
    adv_head.train()

    total_main = 0.0
    total_adv = 0.0
    steps = 0

    pbar = tqdm(loader, desc="Adversarial training")
    for X, y, g, idxs in pbar:
        X = X.to(device)
        y = y.to(device)
        g = g.to(device)

        # -----------------------
        # 1) Update adversary (encoder frozen)
        # -----------------------
        opt_adv.zero_grad()
        with torch.no_grad():
            feats_detached = extract_features_from_baseline(model, X).detach()
        adv_logits = adv_head(feats_detached)
        loss_adv = criterion_adv(adv_logits, g)
        loss_adv.backward()
        opt_adv.step()

        # -----------------------
        # 2) Update encoder + classifier (adversarial via grad reverse)
        # -----------------------
        opt_main.zero_grad()

        feats = extract_features_from_baseline(model, X)  # requires grad
        logits = model.fc(feats)                          # classifier head
        loss_main = criterion_main(logits, y)

        # adversarial loss applied via grad reverse
        rev_feats = grad_reverse(feats, lambda_adv)
        adv_logits_for_encoder = adv_head(rev_feats)
        loss_adv_for_enc = criterion_adv(adv_logits_for_encoder, g)

        total_loss = loss_main + lambda_adv * loss_adv_for_enc
        total_loss.backward()
        opt_main.step()

        total_main += loss_main.item()
        total_adv += loss_adv.item()
        steps += 1

        pbar.set_postfix({"main": f"{loss_main.item():.4f}", "adv": f"{loss_adv.item():.4f}", "shape": str(X.shape)})

    return (total_main / max(1, steps), total_adv / max(1, steps))

# -----------------------
# Main function
# -----------------------
def main():
    set_seed(SEED)

    print("Loading metadata:", METADATA_FP)
    df = pd.read_csv(METADATA_FP)

    # detect group column
    group_col = None
    for c in GROUP_COLUMN_CANDIDATES:
        if c in df.columns:
            group_col = c
            break
    if group_col is None:
        raise RuntimeError(f"No group column found in metadata. Tried: {GROUP_COLUMN_CANDIDATES}")

    print("Unique groups (sample):", df[group_col].unique()[:10])

    # build dataset & loader
    dataset = CremaAdversarialDataset(df, group_col=group_col)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_pad_logmel)

    # build model & adversary
    num_classes = int(df["emotion"].nunique())
    model = BaselineCNN(num_classes=num_classes).to(DEVICE)

    # feature_dim = 64 in BaselineCNN (see baseline_cnn.py)
    feature_dim = 64
    adv_head = nn.Sequential(
        nn.Linear(feature_dim, 64),
        nn.ReLU(),
        nn.Linear(64, len(sorted(dataset.group_map.keys())) if len(dataset.group_map)>0 else max(dataset.group_map.values())+1)
    ).to(DEVICE)

    criterion_main = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()

    opt_main = optim.Adam(list(model.parameters()), lr=LR_MAIN)
    opt_adv = optim.Adam(list(adv_head.parameters()), lr=LR_ADV)

    # training
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        avg_main, avg_adv = train_adversarial(
            model, adv_head, loader,
            opt_main, opt_adv,
            criterion_main, criterion_adv,
            device=DEVICE, lambda_adv=LAMBDA_ADV
        )
        print(f"Epoch {epoch} finished | main_loss={avg_main:.4f} adv_loss={avg_adv:.4f}")

    # save checkpoint
    ckpt = OUTPUT_DIR / "adversarial_checkpoint.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "adv_state_dict": adv_head.state_dict(),
        "group_map": dataset.group_map,
        "metadata_used": str(METADATA_FP)
    }, ckpt)
    print("Saved checkpoint:", ckpt)
    print("Done. Time elapsed: {:.1f}s".format(time.time() - start))

if __name__ == "__main__":
    main()
