import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F


########################################
# Emotion extraction helper
########################################

def extract_emotion_from_filename(f):
    """
    Example: 1001_IEO_NEU_XX
    Emotion = 3rd segment = NEU
    """
    parts = f.split("_")
    if len(parts) < 3:
        return None
    return parts[2]


########################################
# FairVoice Dataset
########################################

class FairVoiceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, "data/processed/metadata.csv")
        self.feat_dir = os.path.join(root_dir, "data/features")

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Processed metadata not found at: {self.meta_path}")

        df = pd.read_csv(self.meta_path)

        # Extract emotion and encode
        df["emotion"] = df["file"].apply(extract_emotion_from_filename)
        emotion_list = sorted(df["emotion"].unique())
        self.emotion_to_id = {e: i for i, e in enumerate(emotion_list)}
        df["label"] = df["emotion"].map(self.emotion_to_id)

        self.df = df  # store full metadata

        self.items = []
        for _, row in df.iterrows():
            pt_path = os.path.join(self.feat_dir, f"{row['file']}.pt")
            if not os.path.exists(pt_path):
                continue
            self.items.append(row)

        print(f"Dataset loaded: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]

        # load feature
        pt_path = os.path.join(self.feat_dir, f"{row['file']}.pt")
        data = torch.load(pt_path)

        if isinstance(data, torch.Tensor):
            feats = data
        elif isinstance(data, dict):
            feats = next(v for v in data.values() if isinstance(v, torch.Tensor))
        else:
            raise TypeError("Unknown tensor format")

        label = int(row["label"])

        # return two things for training, full metadata for bias
        metadata = {
            "file": row["file"],
            "ActorID": row["ActorID"],
            "Age": row["Age"],
            "Sex": row["Sex"],
            "Race": row["Race"],
            "Ethnicity": row["Ethnicity"],
            "emotion_text": row["emotion"],
            "label": label
        }

        return feats, label, metadata


########################################
# Padding
########################################

def pad_batch(batch):
    feats, labels, metadata = zip(*batch)

    max_len = max(x.shape[-1] for x in feats)

    padded = []
    for x in feats:
        padded.append(torch.nn.functional.pad(x, (0, max_len - x.shape[-1])))

    return torch.stack(padded), torch.tensor(labels), metadata


########################################
# Create dataloader
########################################

def create_dataloader(root_dir, batch_size=8, shuffle=True):
    dataset = FairVoiceDataset(root_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_batch
    )
    return dataset, loader
