import torch
from torch import nn


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018).

    Input: (batch, 1, channels, samples). Output: (batch, n_classes) class scores.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: temporal convolution, then depthwise spatial convolution
            nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            # Block 2: separable convolution (depthwise temporal + pointwise)
            nn.Conv2d(
                F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False
            ),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_features = self.features(torch.zeros(1, 1, n_channels, n_samples)).shape[
                1
            ]
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
