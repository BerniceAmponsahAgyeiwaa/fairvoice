from pathlib import Path
import torchaudio
import torch

# Always set manually
root = Path("/Users/pc/Desktop/CODING/Others/fairvoice")


class FeatureExtractor:
    def __init__(self):
        self.processed_dir = root / "data/processed/audio_clean"
        self.feature_dir = root / "data/features"
        self.feature_dir.mkdir(parents=True, exist_ok=True)

        self.target_sr = 16000

        # Stable transforms for your environment
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.target_sr,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )

    def safe_load_audio(self, file_path: Path):
        """Always load with string paths and avoid SOX crashes."""
        try:
            wav, sr = torchaudio.load(str(file_path))
        except Exception:
            print(f"⚠️ torchaudio failed on {file_path.name}, retrying with soundfile backend...")
            import soundfile as sf
            wav_np, sr = sf.read(str(file_path))
            wav = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
        return wav, sr

    def extract_features(self, file_path: Path):
        wav, sr = self.safe_load_audio(file_path)

        # Force mono
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        # MFCC + logmel
        mfcc = self.mfcc_transform(wav)
        mel = self.mel_transform(wav)
        mel = torch.log(mel + 1e-6)

        return {"mfcc": mfcc, "logmel": mel}

    def run(self):
        files = list(self.processed_dir.glob("*.wav"))
        if not files:
            print("❌ No cleaned audio found.")
            return

        for f in files:
            feats = self.extract_features(f)
            out_file = self.feature_dir / f"{f.stem}.pt"
            torch.save(feats, out_file)
            print(f"✔ Saved: {out_file.name}")

        print("\n🎉 Feature extraction complete.\n")


if __name__ == "__main__":
    FeatureExtractor().run()
