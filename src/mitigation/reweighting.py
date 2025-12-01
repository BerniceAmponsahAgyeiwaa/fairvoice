# train_reweighting.py
# Single-file reweighting training (Race + Sex intersectional weights)
# Drop into: /Users/pc/Desktop/CODING/Others/fairvoice/src/model/train_reweighting.py
# Run: python src/model/train_reweighting.py

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

# -----------------------
# CONFIG - adjust if needed
# -----------------------
ROOT = "/Users/pc/Desktop/CODING/Others/fairvoice"
META_CSV = os.path.join(ROOT, "data", "processed", "metadata.csv")
FEAT_DIR = os.path.join(ROOT, "data", "features")
OUT_DIR = os.path.join(ROOT, "src", "model", "reweighting_results")

EPOCHS = 10
BATCH = 8
LR = 1e-4
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Utility: compute race+sex intersectional weights
# -----------------------
def compute_intersectional_weights(metadata_csv, output_json):
    """
    Compute weights per (Race, Sex) group as max_count / group_count,
    normalized so the largest weight = 1.0 (optional normalized).
    Saves JSON mapping and returns dict mapping tuple->weight.
    """
    df = pd.read_csv(metadata_csv)

    # normalize missing values
    df["Race"] = df["Race"].fillna("Unknown")
    # treat Sex column name flexible
    sex_col = "Sex" if "Sex" in df.columns else ("Gender" if "Gender" in df.columns else None)
    if sex_col is None:
        raise ValueError("Metadata has neither 'Sex' nor 'Gender' column.")
    df[sex_col] = df[sex_col].fillna("Unknown")

    # count groups
    group_counts = df.groupby(["Race", sex_col]).size().reset_index(name="count")
    max_count = group_counts["count"].max()

    # weight per group = max_count / count
    group_counts["weight"] = group_counts["count"].apply(lambda c: float(max_count / c))

    # Normalize weights so max becomes 1.0 (optional but keeps scale stable)
    group_counts["weight_norm"] = group_counts["weight"] / group_counts["weight"].max()

    # Build mapping
    weight_map = {}
    for _, r in group_counts.iterrows():
        key = f"{r['Race']}||{r[sex_col]}"
        weight_map[key] = float(r["weight_norm"])

    # Save JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"group_weights": weight_map, "counts": group_counts.to_dict(orient="records")}, f, indent=2)

    print("Computed intersectional weights (Race||Sex) and saved to:", output_json)
    return weight_map


# -----------------------
# Dataset + dataloader (self-contained)
# -----------------------
def extract_emotion_from_filename(f):
    parts = f.split("_")
    return parts[2] if len(parts) >= 3 else None

class ReweightDataset(Dataset):
    """
    Returns: feats (Tensor), label (int), metadata (dict)
    """
    def __init__(self, root_dir, metadata_filename="metadata.csv"):
        meta_path = os.path.join(root_dir, "data", "processed", metadata_filename)
        feat_dir = os.path.join(root_dir, "data", "features")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Processed metadata not found at: {meta_path}")

        df = pd.read_csv(meta_path)

        # normalize columns
        df["file"] = df["file"].astype(str)
        df["Race"] = df.get("Race", "").fillna("Unknown")
        sex_col = "Sex" if "Sex" in df.columns else ("Gender" if "Gender" in df.columns else None)
        if sex_col is None:
            raise ValueError("Metadata must contain 'Sex' or 'Gender' column.")
        df[sex_col] = df[sex_col].fillna("Unknown")

        # emotion and labels
        df["emotion"] = df["file"].apply(extract_emotion_from_filename)
        emotion_list = sorted(df["emotion"].dropna().unique())
        self.emotion_to_id = {e: i for i, e in enumerate(emotion_list)}
        df["label"] = df["emotion"].map(self.emotion_to_id)

        # keep only rows with .pt
        rows = []
        for _, row in df.iterrows():
            pt = os.path.join(feat_dir, f"{row['file']}.pt")
            if os.path.exists(pt):
                rows.append(row)
        if len(rows) == 0:
            raise RuntimeError("No .pt feature files found in data/features for the metadata provided.")

        self.df = pd.DataFrame(rows).reset_index(drop=True)
        self.feat_dir = feat_dir
        self.sex_col = sex_col

        # mapping for race and sex
        self.race_vals = sorted(self.df["Race"].fillna("Unknown").unique())
        self.sex_vals = sorted(self.df[self.sex_col].fillna("Unknown").unique())

        # rows list for indexing
        self.rows = self.df.to_dict(orient="records")
        print(f"Dataset loaded: {len(self.rows)} samples. Emotions: {self.emotion_to_id}")

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

        # squeeze extra dim if present [1, 40, T] -> [40, T]
        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats.squeeze(0)

        label = int(row["label"])
        meta = {
            "file": row["file"],
            "ActorID": row.get("ActorID", ""),
            "Age": row.get("Age", ""),
            "Sex": row.get(self.sex_col, "Unknown"),
            "Race": row.get("Race", "Unknown"),
            "Ethnicity": row.get("Ethnicity", ""),
            "emotion_text": row.get("emotion", ""),
            "label": label
        }
        return feats, label, meta

def pad_batch(batch):
    feats, labels, metas = zip(*batch)
    # ensure feats are [C, T]
    feats = [f.squeeze(0) if f.ndim == 3 and f.shape[0] == 1 else f for f in feats]
    max_len = max(x.shape[-1] for x in feats)
    padded = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in feats]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long), list(metas)


# -----------------------
# SimpleCNN (same shape as other scripts)
# input: [B, C=40, T]
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(40, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# -----------------------
# Training loop with per-sample weighting
# -----------------------
def train_reweighting(root=ROOT, epochs=EPOCHS, batch_size=BATCH, lr=LR):
    # reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # dataset
    ds = ReweightDataset(root, metadata_filename="metadata.csv")

    # compute intersectional weights and save
    group_weights_json = os.path.join(OUT_DIR, "group_weights_race_sex.json")
    group_weights = compute_intersectional_weights(META_CSV, group_weights_json)

    # Build per-sample weight array aligned with ds.rows
    sample_weights = []
    for r in ds.rows:
        key = f"{r['Race']}||{r.get(ds.sex_col, 'Unknown')}"
        w = group_weights.get(key, 1.0)
        sample_weights.append(float(w))
    sample_weights = np.array(sample_weights, dtype=np.float32)

    # Save sample_weights mapping (by file)
    sample_map = {r["file"]: float(w) for r, w in zip(ds.rows, sample_weights)}
    with open(os.path.join(OUT_DIR, "sample_weights_by_file.json"), "w") as f:
        json.dump(sample_map, f, indent=2)

    # train/val split
    total = len(ds)
    val_size = int(0.2 * total)
    train_size = total - val_size
    torch.manual_seed(SEED)
    train_set, val_set = random_split(ds, [train_size, val_size])

    # If using random_split indices, build per-sample weights for train_set
    train_indices = train_set.indices if hasattr(train_set, "indices") else list(range(train_size))
    # random_split in newer PyTorch returns Subset with .indices, older returns datasets; handle both:
    try:
        train_indices = train_set.indices
    except Exception:
        # fallback: compute mapping using lengths (rare)
        train_indices = list(range(train_size))

    # create dataloaders (we'll use standard dataloader and apply weights inside loss)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    model = SimpleCNN(num_classes=len(ds.emotion_to_id)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Use CrossEntropyLoss with reduction='none' and weight per-sample applied manually
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    best_val_acc = 0.0
    training_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for feats, labels, metas in loop:
            feats = feats.to(device)
            labels = labels.to(device)

            logits = model(feats)
            losses = loss_fn(logits, labels)  # shape [B]

            # compute per-sample weights for this batch by looking up file in metas
            batch_weights = []
            for m in metas:
                w = sample_map.get(m["file"], 1.0)
                batch_weights.append(w)
            batch_weights = torch.tensor(batch_weights, dtype=torch.float32).to(device)

            weighted_losses = losses * batch_weights
            loss = weighted_losses.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * feats.size(0)
            total_samples += feats.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

            loop.set_postfix({"loss": epoch_loss / total_samples, "acc": correct / total_samples})

        train_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
        train_acc = correct / total_samples if total_samples > 0 else 0.0

        # validation
        model.eval()
        all_preds = []
        all_labels = []
        all_meta = []
        with torch.no_grad():
            for feats, labels, metas in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                logits = model(feats)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_meta.extend(metas)

        val_acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if len(all_labels) > 0 else 0.0
        training_log.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc)
        })

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(OUT_DIR, "best_model_reweighting.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] New best model saved: {best_path}")

    # Save training log CSV
    import csv
    log_csv = os.path.join(OUT_DIR, "training_log.csv")
    keys = training_log[0].keys() if len(training_log) > 0 else ["epoch", "train_loss", "train_acc", "val_acc"]
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(keys))
        writer.writeheader()
        writer.writerows(training_log)

    # Final evaluation on full dataset (for bias analysis)
    full_loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=pad_batch)
    model.eval()
    preds_all = []
    labels_all = []
    metas_all = []
    with torch.no_grad():
        for feats, labels, metas in full_loader:
            feats = feats.to(device)
            logits = model(feats)
            preds = logits.argmax(dim=1).cpu().tolist()
            preds_all.extend(preds)
            labels_all.extend(labels.tolist())
            metas_all.extend(metas)

    # attach preds and trues to metadata and save CSV
    for m, p, t in zip(metas_all, preds_all, labels_all):
        m["pred_label"] = int(p)
        m["true_label"] = int(t)

    preds_csv = os.path.join(OUT_DIR, "predictions_reweighting.csv")
    pd.DataFrame(metas_all).to_csv(preds_csv, index=False)

    # classification report + confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(labels_all, preds_all, target_names=list(ds.emotion_to_id.keys()), zero_division=0)
    report_path = os.path.join(OUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    cm = confusion_matrix(labels_all, preds_all)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(ds.emotion_to_id.keys()),
                yticklabels=list(ds.emotion_to_id.keys()))
    plt.title("Confusion Matrix - Reweighting")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(OUT_DIR, "confusion_matrix_reweighting.png")
    plt.savefig(cm_path)
    plt.close()

    # Save label map and weights
    torch.save(ds.emotion_to_id, os.path.join(OUT_DIR, "label_map.pt"))
    with open(os.path.join(OUT_DIR, "group_weights_race_sex.json"), "r") as f:
        gw = json.load(f)  # already saved earlier

    print("\nAll outputs saved to:", OUT_DIR)
    print("Best val acc:", best_val_acc)
    print("Predictions CSV:", preds_csv)
    print("Classification report:", report_path)
    print("Confusion matrix:", cm_path)
    print("Sample weights file (by file):", os.path.join(OUT_DIR, "sample_weights_by_file.json"))


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    train_reweighting(root=ROOT, epochs=EPOCHS, batch_size=BATCH, lr=LR)
