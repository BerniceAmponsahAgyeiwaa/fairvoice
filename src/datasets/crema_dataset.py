import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

# -----------------------------------------------------------
# Emotion mapping (CREMA-D → numeric)
# -----------------------------------------------------------
EMOTION_MAP = {
    "HAP": 0,
    "SAD": 1,
    "ANG": 2,
    "FEA": 3,
    "DIS": 4,
    "NEU": 5
}

class CremaFeatureDataset(Dataset):
    """
    Loads raw audio from audio_path column and converts to log-mel.
    Uses emotion string column and maps to integer labels.
    """

    def __init__(self, metadata: pd.DataFrame, group_column: str = "Sex"):
        self.metadata = metadata.reset_index(drop=True)

        # critical: these columns must exist
        if "audio_path" not in self.metadata.columns:
            raise ValueError("metadata must contain 'audio_path' column")

        if "emotion" not in self.metadata.columns:
            raise ValueError("metadata must contain 'emotion' column")

        if group_column not in self.metadata.columns:
            raise ValueError(f"metadata missing group column: {group_column}")

        self.group_column = group_column

        # torchaudio transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # -----------------------------
        # Load audio from audio_path
        # -----------------------------
        audio_path = row["audio_path"]
        wav, sr = torchaudio.load(audio_path)

        # ensure 16 kHz mono
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # -----------------------------
        # Convert to log-mel
        # -----------------------------
        mel = self.mel_transform(wav)  # shape [1, n_mels, time]
        logmel = self.db_transform(mel)

        # -----------------------------
        # Encode emotion
        # -----------------------------
        emo_str = row["emotion"]
        if emo_str not in EMOTION_MAP:
            raise ValueError(f"Unknown emotion code: {emo_str}")

        emotion = torch.tensor(EMOTION_MAP[emo_str], dtype=torch.long)

        # -----------------------------
        # Group label (e.g., Sex)
        # -----------------------------
        group_value = row[self.group_column]
        group = torch.tensor(hash(group_value) % 1000, dtype=torch.long)

        return logmel, emotion, group, idx
