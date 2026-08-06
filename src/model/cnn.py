"""
src/model/cnn.py
================
Simple CNN baseline model for Cats vs Dogs binary classification.

Architecture:
    3x [Conv2d -> BatchNorm -> ReLU -> MaxPool]
    Flatten -> FC(512) -> ReLU -> Dropout(0.5) -> FC(2)

Input  : (batch, 3, 224, 224)
Output : (batch, 2)  -- raw logits for [cat, dog]

Usage:
    from src.model.cnn import SimpleCNN
    model = SimpleCNN()
    print(model)
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Lightweight 3-block CNN for binary image classification.

    Designed to be fast to train on CPU while still achieving
    reasonable accuracy on the Cats vs Dogs task.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super(SimpleCNN, self).__init__()

        # ── Block 1: 3 -> 32 channels | 224x224 -> 112x112 ────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 224 -> 112
        )

        # ── Block 2: 32 -> 64 channels | 112x112 -> 56x56 ─────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 112 -> 56
        )

        # ── Block 3: 64 -> 128 channels | 56x56 -> 28x28 ──────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 56 -> 28
        )

        # ── Classifier head ────────────────────────────────────────────────
        # After block3: feature map is (128, 28, 28) = 100352 flat features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # ── Weight initialization ──────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for Conv layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SimpleCNN(num_classes=2)
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # Forward pass test with a dummy batch
    dummy_input = torch.randn(4, 3, 224, 224)   # batch of 4 images
    output = model(dummy_input)
    print(f"\nInput  shape : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")    # Expected: (4, 2)
    assert output.shape == (4, 2), "Output shape mismatch!"
    print("\nSanity check PASSED.")
