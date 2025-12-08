# src/evaluation/bias_metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def accuracy_by_group(df_preds, group_col, label_col="true", pred_col="pred"):
    """
    Returns a DataFrame with accuracy and f1 for each group value.
    df_preds: dataframe with columns [group_col, label_col, pred_col]
    """
    rows = []
    for g, sub in df_preds.groupby(group_col):
        acc = accuracy_score(sub[label_col], sub[pred_col])
        f1 = f1_score(sub[label_col], sub[pred_col], average="macro")
        rows.append({"group": g, "accuracy": acc, "f1_macro": f1, "count": len(sub)})
    return pd.DataFrame(rows)

def disparity_gaps(df_grouped, metric="accuracy"):
    """
    Given df with 'group' and metric column, return gap = max - min
    """
    vals = df_grouped[metric].values
    return float(np.max(vals) - np.min(vals))

def per_class_stat_parity(df_preds, group_col, classes, label_col="true", pred_col="pred"):
    """
    For multiclass, compute per-class statistical parity difference between groups.
    SPD for each class c: max_a P(pred==c | A=a) - min_a P(pred==c | A=a)
    Returns DataFrame with class, max_prob, min_prob, spd
    """
    rows = []
    groups = sorted(df_preds[group_col].unique().tolist())
    for c in classes:
        probs = []
        for g in groups:
            subset = df_preds[df_preds[group_col] == g]
            if len(subset) == 0:
                probs.append(0.0)
            else:
                probs.append((subset[pred_col] == c).mean())
        rows.append({
            "class": c,
            "max_prob": float(np.max(probs)),
            "min_prob": float(np.min(probs)),
            "spd": float(np.max(probs) - np.min(probs))
        })
    return pd.DataFrame(rows)

def tpr_fpr_by_group(df_preds, group_col, classes, label_col="true", pred_col="pred"):
    """
    For each class and group, compute TPR = TP / P, FPR = FP / N
    Returns a DataFrame with columns: class, group, tpr, fpr, support_pos, support_neg
    """
    rows = []
    groups = sorted(df_preds[group_col].unique().tolist())
    for c in classes:
        for g in groups:
            sub = df_preds[df_preds[group_col] == g]
            if len(sub) == 0:
                rows.append({"class": c, "group": g, "tpr": np.nan, "fpr": np.nan, "support_pos": 0, "support_neg": 0})
                continue
            y_true = (sub[label_col] == c).astype(int)
            y_pred = (sub[pred_col] == c).astype(int)
            # True positive rate: TP / P
            P = y_true.sum()
            N = len(y_true) - P
            TP = int(((y_true == 1) & (y_pred == 1)).sum())
            FP = int(((y_true == 0) & (y_pred == 1)).sum())
            tpr = TP / P if P > 0 else np.nan
            fpr = FP / N if N > 0 else np.nan
            rows.append({"class": c, "group": g, "tpr": tpr, "fpr": fpr, "support_pos": int(P), "support_neg": int(N)})
    return pd.DataFrame(rows)

def confusion_matrix_by_group(df_preds, group_col, classes, label_col="true", pred_col="pred"):
    """
    Returns dict: group -> confusion matrix (numpy array) with ordering classes
    """
    cms = {}
    for g, sub in df_preds.groupby(group_col):
        cm = confusion_matrix(sub[label_col], sub[pred_col], labels=classes)
        cms[g] = cm
    return cms
