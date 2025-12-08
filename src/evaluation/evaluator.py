# src/evaluation/evaluator.py
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Ensure src/ is first on path if this is run as script
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def load_checkpoint(model, ckpt_path, map_location=None):
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    label_map = ckpt.get("label_map", None)
    return label_map

def run_inference(model, dataset, device=None, batch_size=32, num_workers=0):
    """
    Runs model on dataset and returns (preds, labels, file_names)
    preds and labels are integer class indices (np.array)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds = []
    labels = []
    files = []

    with torch.no_grad():
        for X, y, fname in loader:
            X = X.to(device)
            logits = model(X)
            p = logits.argmax(dim=1).cpu().numpy()
            preds.extend(p.tolist())
            labels.extend(y.numpy().tolist())
            # fname may be list of filenames or tensors; ensure strings
            files.extend([str(f) for f in fname])

    return np.array(preds), np.array(labels), files
