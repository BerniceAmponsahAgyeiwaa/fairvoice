import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#######################################################################
#                         DATASET + LOADER
#######################################################################

def extract_emotion_from_filename(f):
    """CREMA-D convention: ID_Sentence_Emotion_Level.wav"""
    parts = f.split("_")
    if len(parts) < 3:
        return None
    return parts[2]


class FairVoiceDataset(Dataset):
    def __init__(self, root_dir, metadata_filename="metadata.csv"):
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, "data/processed", metadata_filename)
        self.feat_dir = os.path.join(root_dir, "data/features")

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Processed metadata not found at: {self.meta_path}")

        df = pd.read_csv(self.meta_path)

        # Extract emotion
        df["emotion"] = df["file"].apply(extract_emotion_from_filename)
        emotion_list = sorted(df["emotion"].unique())

        self.emotion_to_id = {e: i for i, e in enumerate(emotion_list)}
        df["label"] = df["emotion"].map(self.emotion_to_id)

        self.df = df

        # Include only rows that have .pt features
        self.items = []
        for _, row in df.iterrows():
            pt_path = os.path.join(self.feat_dir, f"{row['file']}.pt")
            if os.path.exists(pt_path):
                self.items.append(row)

        print(f"Dataset loaded: {len(self.items)} samples.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        pt_path = os.path.join(self.feat_dir, f"{row['file']}.pt")
        data = torch.load(pt_path)

        # Accept either tensor or dict
        if isinstance(data, torch.Tensor):
            feats = data
        elif isinstance(data, dict):
            feats = next(v for v in data.values() if isinstance(v, torch.Tensor))
        else:
            raise TypeError("Unknown .pt format")

        label = int(row["label"])
        return feats, label


def pad_batch(batch):
    feats, labels = zip(*batch)

    # FIX: remove extra channel dimension (input is [1, 40, T] → [40, T])
    feats = [x.squeeze(0) if x.ndim == 3 else x for x in feats]

    max_len = max(x.shape[-1] for x in feats)

    padded = []
    for x in feats:
        pad_amount = max_len - x.shape[-1]
        padded.append(torch.nn.functional.pad(x, (0, pad_amount)))

    return torch.stack(padded), torch.tensor(labels)


#######################################################################
#                         CNN MODEL
#######################################################################

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
        x = self.fc(x)
        return x


#######################################################################
#                         TRAINING PIPELINE
#######################################################################

def train_baseline(root_dir, epochs=10, batch_size=8, lr=1e-4):

    dataset = FairVoiceDataset(root_dir)
    num_classes = len(dataset.emotion_to_id)

    mapping_path = os.path.join(root_dir, "models/label_mapping_baseline.pt")
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    torch.save(dataset.emotion_to_id, mapping_path)

    # -------------------- Split --------------------
    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    model = SimpleCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # -------------------- Training --------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

    # -------------------- Validation --------------------
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device)
            outputs = model(feats)
            preds = outputs.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # -------------------- Metrics --------------------
    report = classification_report(
        all_labels,
        all_preds,
        target_names=list(dataset.emotion_to_id.keys()),
        zero_division=0
    )

    print("\nValidation Report:\n")
    print(report)

    report_path = os.path.join(root_dir, "models/baseline_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # -------------------- Confusion Matrix --------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(dataset.emotion_to_id.keys()),
                yticklabels=list(dataset.emotion_to_id.keys()))
    plt.title("Confusion Matrix - Baseline Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_path = os.path.join(root_dir, "models/baseline_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # -------------------- Save Model --------------------
    model_path = os.path.join(root_dir, "models/baseline_cnn.pth")
    torch.save(model.state_dict(), model_path)

    print("\n🎉 Training Complete!")
    print("Saved model to:", model_path)
    print("Saved metrics to:", report_path)
    print("Saved confusion matrix to:", cm_path)
    print("Saved label mapping to:", mapping_path)


#######################################################################
#                         MAIN ENTRY
#######################################################################

if __name__ == "__main__":
    ROOT = "/Users/pc/Desktop/CODING/Others/fairvoice"
    train_baseline(ROOT, epochs=10, batch_size=8, lr=1e-4)
