# src/training/trainer.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Trainer:
    """
    Trainer expected by the training script.
    Signature: Trainer(model, device, out_dir="models", num_classes=6)
    Methods: fit(train_dataset, val_dataset=None, epochs=..., batch_size=..., lr=..., weight_decay=..., num_workers=...)
             evaluate(loader) -> (loss, acc)
    Saves best model to out_dir/baseline_cnn.pth and training history.
    """
    def __init__(self, model, device, out_dir="models", num_classes=6):
        self.model = model.to(device)
        self.device = device
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.num_classes = num_classes

    def fit(self, train_dataset, val_dataset=None,
            epochs=20, batch_size=32, lr=1e-3, weight_decay=1e-5, num_workers=2):
        """
        Train the model. Uses an internal optimizer (Adam).
        Accepts dataset objects (instances of torch.utils.data.Dataset).
        """
        set_seed(42)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Use zero workers for macOS stability; change num_workers if desired on Linux/GPU machines
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs+1):
            self.model.train()
            all_preds = []
            all_labels = []
            running_loss = 0.0

            for X, y, _ in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y.detach().cpu().numpy().tolist())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_labels, all_preds)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            val_loss = None
            val_acc = None
            if val_loader is not None:
                self.model.eval()
                v_preds = []
                v_labels = []
                v_loss_sum = 0.0
                with torch.no_grad():
                    for Xv, yv, _ in val_loader:
                        Xv = Xv.to(self.device)
                        yv = yv.to(self.device)
                        logits = self.model(Xv)
                        lossv = self.criterion(logits, yv)
                        v_loss_sum += lossv.item() * Xv.size(0)
                        v_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                        v_labels.extend(yv.cpu().numpy().tolist())
                val_loss = v_loss_sum / len(val_loader.dataset)
                val_acc = accuracy_score(v_labels, v_preds)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # checkpoint best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    ckpt_path = os.path.join(self.out_dir, "baseline_cnn.pth")
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "label_map": getattr(train_dataset, "label_map", None)
                    }, ckpt_path)

            print(f"Epoch {epoch}/{epochs} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss} val_acc: {val_acc}")

        # save training history
        history_fp = os.path.join(self.out_dir, "training_history.json")
        with open(history_fp, "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"Training complete. Best val acc: {self.best_val_acc:.4f}")
        return history

    def evaluate(self, loader):
        """
        Run evaluation on a DataLoader and return (loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y, _ in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        avg_loss = total_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc
