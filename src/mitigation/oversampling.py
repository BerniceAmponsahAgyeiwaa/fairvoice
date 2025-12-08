# ============================
# oversampling.py
# Data-level bias mitigation
# ============================

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

def compute_group_weights(metadata_df, group_col):
    """
    Compute weights inversely proportional to group sample frequency.
    """
    if group_col not in metadata_df.columns:
        raise ValueError(f"{group_col} not found in metadata dataframe.")

    counts = metadata_df[group_col].value_counts()
    weights = metadata_df[group_col].map(lambda x: 1.0 / counts[x]).values
    return weights, counts


def make_oversampler(metadata_df, group_col):
    """
    Return a PyTorch WeightedRandomSampler for oversampling minority groups.
    """
    weights, counts = compute_group_weights(metadata_df, group_col)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler, counts
