#!/usr/bin/env python3
"""
train_adversarial.py
Single-file adversarial debiasing training (DANN-style) for FairVoice.
Drop into: /Users/pc/Desktop/CODING/Others/fairvoice/src/model/train_adversarial.py
Run: python src/model/train_adversarial.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# CONFIG
# -----------------------
ROOT = "/Users/pc/Desktop/CODING/Others/fairvoice"
META_CSV = os.path.join(ROOT, "data", "processed", "metadata.csv")
FEAT_DIR = os.path.join(ROOT, "data", "features")
OUT_DIR = os.path.join(ROOT, "src", "model", "adversarial_results")

EPOCHS = 10
BATCH = 8
LR = 1e-4
SEED = 42
ADV_LAMBDA = 1.0  # multiplies adversary loss (after GRL it's reversed)

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------
# Gradient Reversal Layer (your version)
# -----------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


# -----------------------
# Dataset + dataloader (same pattern as reweighting)
# -----------------------
def extract_emotion_from_filename(f):
    parts = f.split("_")
    return parts[2] if len(parts) >= 3 else None

class FairVoiceDataset(Dataset):
    def __init__(self, root_dir, metadata_filename="metadata.csv"):
        meta_path = os.path.join(root_dir, "data", "processed", metadata_filename)
        feat_dir = os.path.join(root_dir, "data", "features")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Processed metadata not found: {meta_path}")

        df = pd.read_csv(meta_path)

        # required columns check
        if "Race" not in df.columns:
            df["Race"] = "Unknown"
        sex_col = "Sex" if "Sex" in df.columns else ("Gender" if "Gender" in df.columns else None)
        if sex_col is None:
            df["Sex"] = "Unknown"
            sex_col = "Sex"

        df["Race"] = df["Race"].fillna("Unknown")
        df[sex_col] = df[sex_col].fillna("Unknown")

        # emotion label
        df["emotion"] = df["file"].astype(str).apply(extract_emotion_from_filename)
        emotion_list = sorted(df["emotion"].dropna().unique())
        self.emotion_to_id = {e: i for i, e in enumerate(emotion_list)}
        df["label"] = df["emotion"].map(self.emotion_to_id)

        # keep only files with features
        rows = []
        for _, row in df.iterrows():
            pt = os.path.join(feat_dir, f"{row['file']}.pt")
            if os.path.exists(pt):
                rows.append(row)
        if len(rows) == 0:
            raise RuntimeError("No .pt files found in data/features.")

        self.df = pd.DataFrame(rows).reset_index(drop=True)
        self.feat_dir = feat_dir
        self.sex_col = sex_col

        # race mapping
        races = sorted(self.df["Race"].fillna("Unknown").unique())
        self.race_to_id = {r: i for i, r in enumerate(races)}

        self.rows = self.df.to_dict(orient="records")
        print(f"[Dataset] Loaded {len(self.rows)} samples. Emotions: {self.emotion_to_id}. Races: {self.race_to_id}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pt_path = os.path.join(self.feat_dir, f"{row['file']}.pt")
        data = torch.load(pt_path)

        if isinstance(data, torch.Tensor):
            feats = data
        elif isinstance(data, dict):
            feats = next(v for v in data.values() if isinstance(v, torch.Tensor))
        else:
            raise TypeError("Unknown .pt feature format")

        # squeeze if [1, 40, T]
        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats.squeeze(0)

        label = int(row["label"])
        race = row.get("Race", "Unknown")
        meta = {
            "file": row["file"],
            "ActorID": row.get("ActorID", ""),
            "Race": race,
            "Sex": row.get(self.sex_col, "Unknown"),
            "label": label
        }
        return feats, label, race, meta

def pad_batch(batch):
    feats, labels, races, metas = zip(*batch)
    feats = [f.squeeze(0) if f.ndim == 3 and f.shape[0] == 1 else f for f in feats]
    max_len = max(x.shape[-1] for x in feats)
    padded = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in feats]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long), list(races), list(metas)


# -----------------------
# SimpleCNN encoder (returns feature vector)
# -----------------------
class SimpleCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(40, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # outputs [B, 256, 1]
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        feat = self.pool(x).squeeze(-1)  # [B, 256]
        return feat


# -----------------------
# Emotion classifier & Adversary (from your code, adapted)
# -----------------------
class EmotionClassifier(nn.Module):
    def __init__(self, feat_dim=256, num_classes=6):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, h):
        return self.fc(h)

class Adversary(nn.Module):
    def __init__(self, feat_dim=256, num_races=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_races)
        )
    def forward(self, h):
        return self.net(h)


# -----------------------
# TRAINING LOOP
# -----------------------
def train_adversarial(root=ROOT, epochs=EPOCHS, batch_size=BATCH, lr=LR, adv_lambda=ADV_LAMBDA):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ds = FairVoiceDataset(root, metadata_filename="metadata.csv")
    num_classes = len(ds.emotion_to_id)
    num_races = len(ds.race_to_id)

    # Save mappings
    torch.save(ds.emotion_to_id, os.path.join(OUT_DIR, "label_map_adv.pt"))
    with open(os.path.join(OUT_DIR, "race_map_adv.json"), "w") as f:
        json.dump(ds.race_to_id, f, indent=2)

    # split
    total = len(ds)
    val_size = int(0.2 * total)
    train_size = total - val_size
    torch.manual_seed(SEED)
    train_set, val_set = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] device:", device)

    encoder = SimpleCNNEncoder().to(device)
    emotion_clf = EmotionClassifier(feat_dim=256, num_classes=num_classes).to(device)
    adversary = Adversary(feat_dim=256, num_races=num_races).to(device)

    # optimizers
    optimizer_main = optim.Adam(list(encoder.parameters()) + list(emotion_clf.parameters()), lr=lr)
    optimizer_adv = optim.Adam(adversary.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    training_log = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        emotion_clf.train()
        adversary.train()

        total_loss = 0.0
        total_samples = 0
        correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for feats, labels, races, metas in loop:
            feats = feats.to(device)
            labels = labels.to(device)

            # --------------------
            # Forward encoder -> emotion
            # --------------------
            feats_repr = encoder(feats)  # [B, 256]
            logits_em = emotion_clf(feats_repr)
            loss_em = criterion(logits_em, labels)

            # --------------------
            # Adversary: maximize race loss via GRL
            # --------------------
            # map race strings to ids
            race_ids = [ds.race_to_id.get(r, ds.race_to_id.get("Unknown", 0)) for r in races]
            race_ids = torch.tensor(race_ids, dtype=torch.long).to(device)

            # Adversary step: use optimizer_adv to train adversary to predict race from features
            # but when computing encoder gradients we use GRL to reverse
            # First, update adversary: forward on feats_repr detached
            optimizer_adv.zero_grad()
            adv_logits_on_detached = adversary(feats_repr.detach())
            loss_adv_det = criterion(adv_logits_on_detached, race_ids)
            loss_adv_det.backward()
            optimizer_adv.step()

            # Now compute adversary loss that will be reversed into encoder via GRL
            adv_logits = adversary(grad_reverse(feats_repr, adv_lambda))
            loss_adv = criterion(adv_logits, race_ids)

            # --------------------
            # Combined update for encoder + emotion classifier
            # --------------------
            optimizer_main.zero_grad()
            loss_combined = loss_em + loss_adv  # GRL flips encoder grads in backward
            loss_combined.backward()
            optimizer_main.step()

            total_loss += float(loss_combined.item()) * feats.size(0)
            total_samples += feats.size(0)
            preds = logits_em.argmax(dim=1)
            correct += (preds == labels).sum().item()

            loop.set_postfix({"loss": total_loss / total_samples, "acc": correct / total_samples})

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = correct / total_samples if total_samples > 0 else 0.0

        # validation
        encoder.eval()
        emotion_clf.eval()
        all_preds = []
        all_labels = []
        all_meta = []
        with torch.no_grad():
            for feats, labels, races, metas in val_loader:
                feats = feats.to(device)
                logits = emotion_clf(encoder(feats))
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_meta.extend(metas)

        val_acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if len(all_labels) > 0 else 0.0
        training_log.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), os.path.join(OUT_DIR, "best_encoder.pth"))
            torch.save(emotion_clf.state_dict(), os.path.join(OUT_DIR, "best_emotion_clf.pth"))
            torch.save(adversary.state_dict(), os.path.join(OUT_DIR, "best_adversary.pth"))
            print(f"[Info] New best saved (val_acc={best_val_acc:.4f})")

    # Save training log
    pd.DataFrame(training_log).to_csv(os.path.join(OUT_DIR, "training_log_adversarial.csv"), index=False)

    # Final evaluation & save predictions
    full_loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=pad_batch)
    encoder.eval()
    emotion_clf.eval()
    preds_all = []
    trues_all = []
    metas_all = []
    with torch.no_grad():
        for feats, labels, races, metas in full_loader:
            feats = feats.to(device)
            logits = emotion_clf(encoder(feats))
            preds = logits.argmax(dim=1).cpu().tolist()
            preds_all.extend(preds)
            trues_all.extend(labels.tolist())
            metas_all.extend(metas)

    # attach preds
    for m, p, t in zip(metas_all, preds_all, trues_all):
        m["pred_label"] = int(p)
        m["true_label"] = int(t)

    preds_csv = os.path.join(OUT_DIR, "predictions_adversarial.csv")
    pd.DataFrame(metas_all).to_csv(preds_csv, index=False)

    # classification report + confusion matrix
    report = classification_report(trues_all, preds_all, target_names=list(ds.emotion_to_id.keys()), zero_division=0)
    with open(os.path.join(OUT_DIR, "classification_report_adversarial.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(trues_all, preds_all)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(ds.emotion_to_id.keys()),
                yticklabels=list(ds.emotion_to_id.keys()))
    plt.title("Confusion Matrix - Adversarial")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(OUT_DIR, "confusion_matrix_adversarial.png")
    plt.savefig(cm_path)
    plt.close()

    print("\nAdversarial training finished.")
    print("Best val acc:", best_val_acc)
    print("Saved predictions:", preds_csv)
    print("Saved report:", os.path.join(OUT_DIR, "classification_report_adversarial.txt"))
    print("Saved confusion matrix:", cm_path)


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    train_adversarial(root=ROOT, epochs=EPOCHS, batch_size=BATCH, lr=LR, adv_lambda=ADV_LAMBDA)
