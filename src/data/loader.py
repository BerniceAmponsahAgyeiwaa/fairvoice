from pathlib import Path
import pandas as pd
import torchaudio

root = Path("/Users/pc/Desktop/CODING/Others/fairvoice")

class DataLoaderCREMA:
    def __init__(self):
        self.meta_file = root / "data/processed/crema_metadata.csv"
        self.audio_dir = root / "data/raw/CREMA-D/AudioWAV"
        self.metadata = pd.read_csv(self.meta_file)
        self.metadata["audio_path"] = self.metadata["Filename"].apply(
            lambda x: self.audio_dir / f"{x}.wav"
        )
        self.metadata = self.metadata[self.metadata["audio_path"].apply(lambda p: p.exists())]

    def load_audio(self, file_path):
        return torchaudio.load(str(file_path))

    def get_item(self, idx):
        row = self.metadata.iloc[idx]
        wav, sr = self.load_audio(row["audio_path"])
        return {
            "waveform": wav,
            "sr": sr,
            "emotion": row["Emotion"],
            "actor": row["ActorID"],
            "sex": row["Sex"],
            "race": row["Race"],
            "age": row["Age"],
            "filename": row["Filename"]
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return self.get_item(idx)
