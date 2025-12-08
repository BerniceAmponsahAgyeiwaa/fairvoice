# ============================
# reweighting.py
# Algorithm-level loss weighting
# ============================

import torch

def compute_class_weights(counts_dict):
    """
    Convert a dictionary of group counts to normalized inverse-frequency weights.
    Example input:
        counts_dict = {'African American': 200, 'Caucasian': 500, 'Asian': 60}
    """
    weights = {k: 1.0 / v for k, v in counts_dict.items()}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return weights


def class_weights_tensor(counts_dict, label_encoder, device="cpu"):
    """
    Turns class weights into tensor aligned with encoded label indices.
    label_encoder: a fitted sklearn LabelEncoder or mapping
    """
    w_dict = compute_class_weights(counts_dict)
    ordered = [w_dict[c] for c in label_encoder.classes_]
    return torch.tensor(ordered, dtype=torch.float).to(device)
