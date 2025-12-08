# ============================================
# adversarial.py
# Representation-level adversarial debiasing
# ============================================

import torch
import torch.nn as nn
from torch.autograd import Function


# -------- Gradient Reversal Layer -------- #
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, λ):
        ctx.lambda_ = λ
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, λ=1.0):
    return GradReverse.apply(x, λ)


# -------- Adversarial Debiasing Model -------- #
class AdversarialSER(nn.Module):
    """
    A general SER architecture with:
    - feature_extractor: CNN/wav2vec2/etc returning feature vectors
    - emotion_head: classifies emotion
    - adv_head: classifies demographic attribute (race/sex/age)
    """
    def __init__(self, feature_extractor, feature_dim, num_emotions, num_groups):
        super().__init__()
        self.feature_extractor = feature_extractor

        # Main emotion classifier
        self.emotion_head = nn.Linear(feature_dim, num_emotions)

        # Adversarial classifier
        self.adv_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_groups)
        )

    def forward(self, x, λ=1.0):
        features = self.feature_extractor(x)  # shape: (B, feature_dim)

        emo_logits = self.emotion_head(features)

        rev = grad_reverse(features, λ)
        adv_logits = self.adv_head(rev)

        return emo_logits, adv_logits


# -------- Loss Combination -------- #
def adversarial_loss(loss_main, loss_adv, adv_weight=0.2):
    """
    Final loss = main_loss - adv_weight * adversarial_loss
    """
    return loss_main - adv_weight * loss_adv
