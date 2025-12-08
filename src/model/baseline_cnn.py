# src/models/baseline_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    Lightweight CNN for log-mel spectrogram inputs.
    Input shape: (batch, 1, n_mels, time)
    Outputs: logits over num_classes
    """
    def __init__(self, n_mels=64, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5,5), padding=(2,2))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))  # global pooling to (1,1)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)           # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.fc(x)             # (B, num_classes)
        return x
