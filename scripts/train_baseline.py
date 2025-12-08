# scripts/train_baseline.py
import sys
from pathlib import Path

# ensure project root src is loadable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
print("Loaded src from:", SRC)

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.datasets.crema_dataset import CremaFeatureDataset
from src.model.baseline_cnn import BaselineCNN
from src.training.trainer import Trainer, set_seed

def main():
    set_seed(42)

    DATA_ROOT = ROOT / "data"
    FEATURE_DIR = DATA_ROOT / "features"
    PROC_DIR = DATA_ROOT / "processed"

    train_csv = PROC_DIR / "metadata_train.csv"
    val_csv = PROC_DIR / "metadata_val.csv"
    test_csv = PROC_DIR / "metadata_test.csv"

    print("Train CSV:", train_csv)
    print("Val CSV:  ", val_csv)
    print("Test CSV: ", test_csv)
    print("Feature dir:", FEATURE_DIR)

    # Build datasets (uses your existing CremaFeatureDataset signature)
    train_ds = CremaFeatureDataset(metadata_csv=str(train_csv), feature_dir=str(FEATURE_DIR), feature_key="logmel")
    val_ds = CremaFeatureDataset(metadata_csv=str(val_csv), feature_dir=str(FEATURE_DIR), feature_key="logmel", label_map=train_ds.label_map)
    test_ds = CremaFeatureDataset(metadata_csv=str(test_csv), feature_dir=str(FEATURE_DIR), feature_key="logmel", label_map=train_ds.label_map)

    # Build model
    num_classes = len(train_ds.label_map)
    model = BaselineCNN(num_classes=num_classes)

    # Trainer (matches trainer.py signature)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model=model, device=device, out_dir=str(ROOT / "models"), num_classes=num_classes)

    # Train
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds, epochs=5, batch_size=16, lr=1e-3)

    # Evaluate on test set
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")

    # If checkpoint saved, that's the best model. Also save final model state.
    final_fp = ROOT / "models" / "baseline_cnn_final.pth"
    torch.save({"model_state_dict": model.state_dict(), "label_map": train_ds.label_map}, final_fp)
    print(f"Saved final model to: {final_fp}")

if __name__ == "__main__":
    main()
