from pathlib import Path
import pandas as pd

root = Path("/Users/pc/Desktop/CODING/Others/fairvoice")

class MetadataExtractor:
    def __init__(self):
        self.metadata_dir = root / "data" / "raw" / "CREMA-D" / "metadata"
        self.audio_dir = root / "data" / "raw" / "CREMA-D" / "AudioWAV"
        self.sentence_csv = self.metadata_dir / "SentenceFilenames.csv"
        self.demo_csv = self.metadata_dir / "VideoDemographics.csv"
        self.output_csv = root / "data" / "processed" / "crema_metadata.csv"

    def validate(self):
        if not self.sentence_csv.exists():
            raise FileNotFoundError(self.sentence_csv)
        if not self.demo_csv.exists():
            raise FileNotFoundError(self.demo_csv)
        if not self.audio_dir.exists():
            raise FileNotFoundError(self.audio_dir)

    def load(self):
        s = pd.read_csv(self.sentence_csv)
        d = pd.read_csv(self.demo_csv)
        return s, d

    def merge(self, s, d):
        s["ActorID"] = s["Filename"].str[:4]
        d["ActorID"] = d["ActorID"].astype(str)
        return s.merge(d, on="ActorID", how="left")

    def add_audio_paths(self, df):
        df["audio_path"] = df["Filename"].apply(
            lambda x: str((self.audio_dir / f"{x}.wav").resolve())
        )
        return df

    def save(self, df):
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)

    def run(self):
        self.validate()
        s, d = self.load()
        merged = self.merge(s, d)
        final = self.add_audio_paths(merged)
        self.save(final)

if __name__ == "__main__":
    MetadataExtractor().run()
