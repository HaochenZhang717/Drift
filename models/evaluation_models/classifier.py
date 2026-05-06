import torch
import torch.nn as nn


class Simple1DCNNClassifier(nn.Module):
    """Simple 1D-CNN classifier for time-series classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        hidden_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        c1 = hidden_channels
        c2 = hidden_channels * 2
        c3 = hidden_channels * 4

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T), (B, T, C), or (B, C, T)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        elif x.ndim == 3:
            # assume (B, T, C) if last dim is channel-like
            if x.shape[1] > x.shape[2]:
                x = x.transpose(1, 2)  # (B, C, T)
        else:
            raise ValueError(f"Expected input with 2 or 3 dims, got {x.ndim}")

        h = self.features(x)
        logits = self.classifier(h)
        return logits
