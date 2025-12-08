from pathlib import Path
import torchaudio
import torchaudio.transforms as T
import pandas as pd

root = Path("/Users/pc/Desktop/CODING/Others/fairvoice")

class PreprocessorCREMA:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.meta_file = root / "data/processed/crema_metadata.csv"
        self.audio_in = root / "data/raw/CREMA-D/AudioWAV"
        self.audio_out = root / "data/processed/audio_clean"
        self.audio_out.mkdir(parents=True, exist_ok=True)
        self.metadata = pd.read_csv(self.meta_file)
        self.resampler = T.Resample(orig_freq=48000, new_freq=target_sr)

    def resample_if_needed(self, wav, sr):
        if sr != self.target_sr:
            return self.resampler(wav)
        return wav

    def normalize(self, wav):
        return wav / wav.abs().max()

    def process_one(self, filename):
        in_path = self.audio_in / f"{filename}.wav"
        out_path = self.audio_out / f"{filename}.wav"
        wav, sr = torchaudio.load(str(in_path))
        wav = self.resample_if_needed(wav, sr)
        wav = self.normalize(wav)
        torchaudio.save(str(out_path), wav, self.target_sr)
        return out_path

    def run(self):
        paths = []
        for _, row in self.metadata.iterrows():
            fn = row["Filename"]
            clean_path = self.process_one(fn)
            paths.append(str(clean_path))
        self.metadata["clean_path"] = paths
        self.metadata.to_csv(self.meta_file, index=False)

if __name__ == "__main__":
    PreprocessorCREMA().run()
