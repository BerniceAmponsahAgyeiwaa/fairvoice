import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torchaudio

# Import dataset/dataloader
from src.data.dataloader import FairVoiceDataset, create_dataloader


# ------------------------------
# 1. SIMPLE CNN MODEL
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ------------------------------
# 2. TRAINING FUNCTION
# ------------------------------
def train_model(num_epochs=5, batch_size=16, lr=0.001):

    print("Loading dataset...")
    root = Path("/Users/pc/Desktop/CODING/Others/fairvoice")

    dataset, loader = create_dataloader(root, batch_size=batch_size)
    print(f"Dataset loaded → {len(dataset)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SimpleCNN(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TRAIN LOOP
    for epoch in range(num_epochs):
        print(f"\n----- EPOCH {epoch+1}/{num_epochs} -----")

        running_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(loader)
        accuracy = 100 * correct / total
        print(f"Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Save model
    out_path = root / "models/simple_cnn.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    print(f"\n TRAINING COMPLETE — model saved to: {out_path}")

    return model


# ============================================================
# 3. LOAD MODEL WITHOUT RETRAINING
# ============================================================

def load_model(model_path):
    """Load a trained model safely."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=6).to(device)

    if not Path(model_path).exists():
        print("⚠ Model not found, training will be triggered.")
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f" Loaded model from {model_path}")
    return model


# ============================================================
# 4. EVALUATE MODEL
# ============================================================

def evaluate_model(model, batch_size=16):
    root = Path("/Users/pc/Desktop/CODING/Others/fairvoice")
    dataset, loader = create_dataloader(root, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\n Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy


# ============================================================
# 5. INFERENCE ON A WAV FILE
# ============================================================

EMOTION_MAP = ["neutral", "angry", "fear", "happy", "sad", "disgust"]

def predict_audio(model, wav_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav, sr = torchaudio.load(wav_path)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=256, n_mels=40
    )(wav)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)

    mel = mel.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(mel)
        pred = out.argmax(dim=1).item()

    print(f"\n AUDIO: {wav_path}")
    print(f" Predicted Emotion: {EMOTION_MAP[pred]}")

    return EMOTION_MAP[pred]


# ============================================================
# 6. GREEN BUTTON LOGIC (NO RETRAINING)
# ============================================================
if __name__ == "__main__":

    root = Path("/Users/pc/Desktop/CODING/Others/fairvoice")
    model_path = root / "models/simple_cnn.pth"

    model = load_model(model_path)

    if model is None:
        print("\n No saved model found — training now...\n")
        model = train_model(num_epochs=8, batch_size=16)

    print("\n Evaluating model...")
    evaluate_model(model)
