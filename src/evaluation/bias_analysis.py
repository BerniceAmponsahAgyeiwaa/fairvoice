import torch
from pathlib import Path
import pandas as pd

from src.data.dataloader import create_dataloader
from src.model.simple_cnn import SimpleCNN, load_model
from src.evaluation.bias_metrics import group_accuracy, demographic_parity, equalized_odds


ROOT = Path("/Users/pc/Desktop/CODING/Others/fairvoice")
MODEL_PATH = ROOT / "models/simple_cnn.pth"


def compute_bias():
    print("\n=== LOADING MODEL ===")
    model = load_model(MODEL_PATH)
    model.eval()

    print("\n=== LOADING DATASET ===")
    dataset, loader = create_dataloader(ROOT, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = []

    print("\n=== RUNNING INFERENCE FOR ALL SAMPLES ===")
    with torch.no_grad():
        for feats, labels, metadata in loader:
            feats = feats.to(device)
            outputs = model(feats)
            preds = outputs.argmax(dim=1).cpu().numpy()

            # loop through batch items
            for i, meta in enumerate(metadata):
                records.append({
                    "file": meta["file"],
                    "ActorID": meta["ActorID"],
                    "Age": meta["Age"],
                    "Sex": meta["Sex"],
                    "Race": meta["Race"],
                    "Ethnicity": meta["Ethnicity"],
                    "true_label": meta["label"],
                    "pred_label": int(preds[i]),
                })

    df = pd.DataFrame(records)
    print("\nSaved predictions for", len(df), "items.")

    print("\n=== BIAS RESULTS ===")

    # Group Accuracy
    print("\n--- Accuracy by Race ---")
    print(group_accuracy(df, "Race"))

    print("\n--- Accuracy by Sex ---")
    print(group_accuracy(df, "Sex"))

    print("\n--- Accuracy by Ethnicity ---")
    print(group_accuracy(df, "Ethnicity"))

    # Demographic Parity
    print("\n--- Demographic Parity (Race) ---")
    print(demographic_parity(df, "Race"))

    # Equalized Odds
    print("\n--- Equalized Odds (Race) ---")
    print(equalized_odds(df, "Race"))

    # Save to file
    out_path = ROOT / "evaluation_results/bias_report.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nBias report saved to:", out_path)


if __name__ == "__main__":
    compute_bias()
