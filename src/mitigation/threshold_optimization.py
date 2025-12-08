#!/usr/bin/env python3
"""
Threshold Optimization for FairVoice
------------------------------------
This script adjusts prediction thresholds to improve fairness across groups.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ---------------------------------------------------------
# 1. Load Predictions + Metadata
# ---------------------------------------------------------

def load_predictions(pred_path, meta_path, group_col):
    """Load predictions & metadata, merge, detect join key, fix suffixes."""
    print(f"\nPredictions file: {pred_path}")
    print(f"Metadata file: {meta_path}")

    preds = pd.read_csv(pred_path)
    meta = pd.read_csv(meta_path)

    print(f"Loaded preds: {len(preds)} rows; columns: {list(preds.columns)}")
    print(f"Loaded meta: {len(meta)} rows; columns: {list(meta.columns)}")

    join_candidates = ["file", "id", "Stimulus_Number"]
    join_key = None

    for key in join_candidates:
        if key in preds.columns and key in meta.columns:
            join_key = key
            break

    if join_key is None:
        raise ValueError(f"Cannot find join key among: {join_candidates}")

    print(f"Using join column: {join_key}")

    # merge with suffixes
    df = preds.merge(meta, on=join_key, how="left", suffixes=("", "_meta"))
    print(f"Merged dataframe shape: {df.shape}")

    # now detect the correct group column
    original_group = group_col

    if group_col not in df.columns:
        # look for suffixed version
        candidates = [group_col + "_meta", group_col + "_x", group_col + "_y"]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break

        if found is None:
            raise KeyError(
                f"Group column '{group_col}' not found after merge. "
                f"Tried: {candidates}. Available columns: {list(df.columns)}"
            )
        else:
            print(f"Group column '{group_col}' not found — using '{found}' instead.")
            group_col = found

    return df, join_key, group_col


# ---------------------------------------------------------
# 2. Label Handling
# ---------------------------------------------------------

EMOTION_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}

def detect_true_labels(df):
    """Detect true label column and convert if needed."""
    candidates = ["true_idx", "label", "emotion", "true"]

    for c in candidates:
        if c in df.columns:
            series = df[c]
            if series.dtype == object:
                print(f"Detected string true-label column: {c}")
                print(f"Mapping string -> int using: {EMOTION_MAP}")
                series = series.map(EMOTION_MAP)
            else:
                print(f"Detected integer true-label column: {c}")
            return series.astype(int)

    raise ValueError("No valid true label column found.")


def detect_pred_labels(df):
    """Use pred_idx (must exist)."""
    if "pred_idx" not in df.columns:
        raise ValueError("pred_idx missing from predictions.")
    return df["pred_idx"].astype(int)


# ---------------------------------------------------------
# 3. Metrics + Threshold Search
# ---------------------------------------------------------

def evaluate_group_fairness(df, group_col, threshold):
    """Compute accuracy per protected group."""
    groups = df[group_col].unique()
    accs = {}
    for g in groups:
        sub = df[df[group_col] == g]
        preds = (sub["pred_prob"] >= threshold).astype(int)
        accs[g] = accuracy_score(sub["true_idx"], preds)
    fairness = min(accs.values())
    return accs, fairness


def find_best_threshold(df, group_col):
    """Global threshold that maximizes minimum group accuracy."""
    best_thresh = None
    best_fairness = -1
    best_accs = None

    thresholds = np.linspace(0.05, 0.95, 50)

    print("\n=== Searching for best global threshold ===")
    for t in tqdm(thresholds):
        accs, fairness = evaluate_group_fairness(df, group_col, t)
        if fairness > best_fairness:
            best_fairness = fairness
            best_thresh = t
            best_accs = accs

    print("\n=== Best Global Threshold ===")
    print(f"Threshold = {best_thresh:.3f}")
    print(f"Fairness (min group acc) = {best_fairness:.4f}")
    print(f"Accuracies by group = {best_accs}")

    return best_thresh, best_accs


# ---------------------------------------------------------
# 4. Save Outputs
# ---------------------------------------------------------

def save_outputs(out_dir, df, threshold, accs):
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "threshold_adjusted_predictions.csv")
    out_txt = os.path.join(out_dir, "threshold_report.txt")

    df.to_csv(out_csv, index=False)

    with open(out_txt, "w") as f:
        f.write("=== Threshold Optimization Report ===\n")
        f.write(f"Chosen threshold: {threshold}\n\n")
        f.write("Group Accuracies:\n")
        for g, a in accs.items():
            f.write(f"{g}: {a:.4f}\n")

    print(f"\nSaved adjusted predictions to: {out_csv}")
    print(f"Saved threshold report to:     {out_txt}")


# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str,
                        default="../../evaluation_results/predictions_test.csv")
    parser.add_argument("--metadata", type=str,
                        default="../../data/processed/metadata_test.csv")
    parser.add_argument("--group_col", type=str, default="Sex")
    parser.add_argument("--out_dir", type=str,
                        default="../../evaluation_results/mitigation_threshold")
    args = parser.parse_args()

    print("\n=== Threshold Optimization (robust) ===")

    df, join_key, group_col = load_predictions(args.predictions,
                                               args.metadata,
                                               args.group_col)

    df["true_idx"] = detect_true_labels(df)
    df["pred_idx"] = detect_pred_labels(df)

    # probability check
    if "pred_prob" not in df.columns:
        print("\nWARNING: 'pred_prob' missing — using dummy prob=1.0 for all rows.")
        df["pred_prob"] = 1.0

    best_threshold, best_accs = find_best_threshold(df, group_col)

    df["pred_adjusted"] = (df["pred_prob"] >= best_threshold).astype(int)

    save_outputs(args.out_dir, df, best_threshold, best_accs)


if __name__ == "__main__":
    main()
