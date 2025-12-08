# group_platt_scaling.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ---------------------------------------------------------
# SAFE LOAD + SMART MERGE
# ---------------------------------------------------------
def safe_load_predictions(pred_path, meta_path, group_col):
    print(f"Predictions: {pred_path}")
    print(f"Metadata: {meta_path}")
    print(f"Group column: {group_col}")

    preds = pd.read_csv(pred_path)
    meta = pd.read_csv(meta_path)

    print("Loaded preds:", len(preds), "rows; columns:", list(preds.columns))
    print("Loaded meta:", len(meta), "rows; columns:", list(meta.columns))

    # -------- Step 1: Find merge key ------
    candidate_keys = ["file", "filename", "Stimulus_Number", "audio_path", "clean_path"]
    merge_key = None

    for k in candidate_keys:
        if k in preds.columns and k in meta.columns:
            merge_key = k
            break

    if merge_key is None:
        raise ValueError("Could not find a common merge key between predictions and metadata.")

    print(f"Using merge key: {merge_key}")

    # -------- Step 2: Rename metadata demographic columns to avoid overwriting -------
    sensitive_cols = ["Sex", "Age", "Race", "Ethnicity", "ActorID", "demo"]

    rename_dict = {c: c + "_meta" for c in sensitive_cols if c in meta.columns}
    meta_renamed = meta.rename(columns=rename_dict)

    # -------- Step 3: Merge safely ----------
    df = preds.merge(meta_renamed, on=merge_key, how="left")
    print("Merged dataframe shape:", df.shape)

    # -------- Step 4: Confirm group column survives ----------
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' missing AFTER merge. "
                         f"Check if metadata overwrote it. Predictions contain it, "
                         f"so metadata values should be suffixed.")

    print(f"Group column '{group_col}' successfully found.")

    # -------- Step 5: Detect true label column ----------
    if "true_idx" in df.columns:
        true_col = "true_idx"
    elif "true" in df.columns:
        true_col = "true"
    else:
        raise ValueError("No true label column found.")

    # -------- Step 6: Check if pred_prob exists ----------
    if "pred_prob" not in df.columns:
        print("WARNING: no 'pred_prob' column found. Using uniform=1.0.")
        df["pred_prob"] = 1.0

    return df, true_col


# ---------------------------------------------------------
# FIT PLATT SCALING PER GROUP
# ---------------------------------------------------------
def platt_scale_group(df, group_col, true_col):
    groups = df[group_col].unique()
    print("\nFound groups:", groups)

    models = {}
    calibrated_probs = []

    for g in groups:
        gdf = df[df[group_col] == g]

        if len(gdf) < 5:
            print(f"Skipping group '{g}' (too few samples).")
            df.loc[df[group_col] == g, "calibrated_prob"] = df.loc[df[group_col] == g, "pred_prob"]
            continue

        model = LogisticRegression(max_iter=1000)

        X = gdf["pred_prob"].values.reshape(-1, 1)
        y = gdf[true_col].values

        # Fit Platt
        model.fit(X, y)
        calibrated = model.predict_proba(X)[:, 1]

        df.loc[df[group_col] == g, "calibrated_prob"] = calibrated
        models[g] = model

    return df, models


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, default="../../evaluation_results/predictions_test.csv")
    parser.add_argument("--meta", type=str, default="../../data/processed/metadata_test.csv")
    parser.add_argument("--group_col", type=str, default="Sex")
    parser.add_argument("--output", type=str, default="../../evaluation_results/mitigation_calibration")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df, true_col = safe_load_predictions(args.preds, args.meta, args.group_col)

    print("\n=== Running Group-wise Platt Scaling ===")
    df, models = platt_scale_group(df, args.group_col, true_col)

    # Save outputs
    out_csv = os.path.join(args.output, "group_calibrated_predictions.csv")
    df.to_csv(out_csv, index=False)

    print(f"\nSaved calibrated predictions to:\n  {out_csv}")


if __name__ == "__main__":
    main()
