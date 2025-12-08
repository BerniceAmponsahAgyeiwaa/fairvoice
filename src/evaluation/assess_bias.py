#!/usr/bin/env python3
"""
scripts/bias_assessment/assess_bias.py

Usage:
    python scripts/bias_assessment/assess_bias.py --model models/baseline_cnn.pth --split test

This script:
- loads the model checkpoint
- loads the test metadata and features
- runs inference
- computes group-wise metrics and fairness metrics
- saves a CSV report and plots in evaluation_results/
"""
import argparse
from pathlib import Path
import sys
import json
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.evaluation.evaluator import run_inference, load_checkpoint
from src.datasets.crema_dataset import CremaFeatureDataset
from src.model.baseline_cnn import BaselineCNN
from src.evaluation.bias_metrics import (
    accuracy_by_group,
    per_class_stat_parity,
    tpr_fpr_by_group,
    confusion_matrix_by_group,
    disparity_gaps
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=str(ROOT / "models" / "baseline_cnn.pth"))
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--feature_dir", type=str, default=str(ROOT / "data" / "features"))
    p.add_argument("--metadata_dir", type=str, default=str(ROOT / "data" / "processed"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "evaluation_results"))
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()
    model_fp = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bias_charts = out_dir / "bias_charts"
    bias_charts.mkdir(parents=True, exist_ok=True)

    # select metadata file
    meta_map = {"train": "metadata_train.csv", "val": "metadata_val.csv", "test": "metadata_test.csv"}
    meta_fp = Path(args.metadata_dir) / meta_map[args.split]
    if not meta_fp.exists():
        raise FileNotFoundError(meta_fp)

    # load dataset
    dataset = CremaFeatureDataset(metadata_csv=str(meta_fp), feature_dir=args.feature_dir, feature_key="logmel")
    num_classes = len(dataset.label_map)
    classes = list(dataset.label_map.keys())

    # build model
    model = BaselineCNN(num_classes=num_classes)
    label_map_from_ckpt = load_checkpoint(model, model_fp, map_location="cpu")
    if label_map_from_ckpt is not None:
        # ensure consistent mapping
        inv = {v:k for k,v in dataset.label_map.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # run inference
    preds, labels, files = run_inference(model, dataset, device=device, batch_size=args.batch_size)

    # build dataframe with metadata
    meta = pd.read_csv(meta_fp)
    # Ensure 'file' matches the .pt stem or filename used; dataset returns feat filename strings
    # files list contains names like "1001_IEO_HAP_LO.pt" or "1001_IEO_HAP_LO.pt"
    stems = [Path(f).stem for f in files]
    df_meta = meta.set_index("file").loc[stems].reset_index()  # will raise if mismatch
    df_meta = df_meta.reset_index(drop=True)
    df_out = pd.DataFrame({"file": stems, "true_idx": labels, "pred_idx": preds})
    # map indices back to label strings
    inv_map = {v:k for k,v in dataset.label_map.items()}
    df_out["true"] = df_out["true_idx"].map(inv_map)
    df_out["pred"] = df_out["pred_idx"].map(inv_map)
    # join demographics
    df_comb = df_out.merge(df_meta, left_on="file", right_on="file", how="left")

    # Save predictions
    preds_fp = out_dir / f"predictions_{args.split}.csv"
    df_comb.to_csv(preds_fp, index=False)
    print("Saved predictions to:", preds_fp)

    # === Group metrics (Sex, Race, Age, Ethnicity) ===
    group_columns = []
    # pick whichever columns exist in metadata
    for c in ["Sex", "sex", "Race", "race", "Age", "Age", "Ethnicity", "ethnicity", "demo"]:
        if c in df_comb.columns and c not in group_columns:
            group_columns.append(c)

    report_rows = []
    # compute accuracy & f1 by group for each column
    for gcol in group_columns:
        df_g = df_comb[[gcol, "true", "pred"]].rename(columns={gcol: "group"})
        df_g = df_g.dropna(subset=["group"])
        if df_g.empty:
            continue
        group_acc = accuracy_by_group(df_g, "group", label_col="true", pred_col="pred")
        gap = disparity_gaps(group_acc, metric="accuracy")
        # Save group_acc CSV
        group_acc_fp = out_dir / "bias_charts" / f"accuracy_by_{gcol}.csv"
        group_acc.to_csv(group_acc_fp, index=False)
        # Plot bar chart
        plt.figure(figsize=(8,4))
        sns.barplot(x="group", y="accuracy", data=group_acc.sort_values("accuracy", ascending=False))
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Accuracy by {gcol}")
        plt.tight_layout()
        plt.savefig(out_dir / "bias_charts" / f"accuracy_by_{gcol}.png")
        plt.close()

        report_rows.append({"group_col": gcol, "disparity_gap_accuracy": gap, "n_groups": len(group_acc)})

        # Per-class SPD for this group
        spd_df = per_class_stat_parity(df_comb.rename(columns={gcol: "group"}), "group", classes=list(dataset.label_map.keys()), label_col="true", pred_col="pred")
        spd_fp = out_dir / "bias_charts" / f"spd_by_{gcol}.csv"
        spd_df.to_csv(spd_fp, index=False)

        # TPR/FPR by group
        tprfpr = tpr_fpr_by_group(df_comb.rename(columns={gcol: "group"}), "group", classes=list(dataset.label_map.keys()), label_col="true", pred_col="pred")
        tprfpr_fp = out_dir / "bias_charts" / f"tprfpr_by_{gcol}.csv"
        tprfpr.to_csv(tprfpr_fp, index=False)

    # Save summary report CSV
    report_df = pd.DataFrame(report_rows)
    report_fp = out_dir / "bias_report.csv"
    report_df.to_csv(report_fp, index=False)
    print("Saved summary report to:", report_fp)

    # Also save overall confusion matrix
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df_comb["true"], df_comb["pred"], labels=list(dataset.label_map.keys()))
        cm_fp = out_dir / "bias_charts" / "confusion_matrix_overall.csv"
        pd.DataFrame(cm, index=list(dataset.label_map.keys()), columns=list(dataset.label_map.keys())).to_csv(cm_fp)
        print("Saved overall confusion matrix to:", cm_fp)
    except Exception as e:
        print("Could not compute confusion matrix:", e)

    print("Bias assessment finished. Charts and CSVs are in:", out_dir)

if __name__ == "__main__":
    main()
