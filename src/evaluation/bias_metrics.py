import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def group_accuracy(df, group_col):
    """
    Computes accuracy per demographic group.

    df must contain: true_label, pred_label, {group_col}
    """
    groups = df[group_col].unique()
    results = {}

    for g in groups:
        sub = df[df[group_col] == g]
        acc = accuracy_score(sub["true_label"], sub["pred_label"])
        results[g] = acc

    return results


def demographic_parity(df, group_col):
    """
    P(Y_hat = y | group = g)
    Measures differences in prediction distribution across groups.
    """
    groups = df[group_col].unique()
    results = {}

    for g in groups:
        sub = df[df[group_col] == g]
        pred_dist = sub["pred_label"].value_counts(normalize=True).to_dict()
        results[g] = pred_dist

    return results


def equalized_odds(df, group_col):
    """
    P(Y_hat = y | Y = y_true, group = g)
    Measures fairness conditioned on true label.
    """
    groups = df[group_col].unique()
    results = {}

    for g in groups:
        sub = df[df[group_col] == g]

        eq_res = {}
        for label in sorted(df["true_label"].unique()):
            sub_label = sub[sub["true_label"] == label]

            if len(sub_label) == 0:
                continue

            pred_dist = sub_label["pred_label"].value_counts(normalize=True).to_dict()
            eq_res[label] = pred_dist

        results[g] = eq_res

    return results
