import torch
import torch.nn as nn


class LightweightCNN2D(nn.Module):
    """CNN 2D léger pour classifier les sons cardiaques."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x