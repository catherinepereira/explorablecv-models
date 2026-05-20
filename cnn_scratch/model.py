"""
A small CNN for CIFAR-10, built with PyTorch layers with heavy annotations so it's clear what each layer is doing

Architecture (32x32x3 input):

    Conv(3 -> 32, 3x3, pad=1) -> ReLU              [N, 32, 32, 32]
    Conv(32 -> 32, 3x3, pad=1) -> ReLU             [N, 32, 32, 32]
    MaxPool(2x2)                                   [N, 32, 16, 16]

    Conv(32 -> 64, 3x3, pad=1) -> ReLU             [N, 64, 16, 16]
    Conv(64 -> 64, 3x3, pad=1) -> ReLU             [N, 64, 16, 16]
    MaxPool(2x2)                                   [N, 64,  8,  8]

    Conv(64 -> 128, 3x3, pad=1) -> ReLU            [N,128,  8,  8]
    MaxPool(2x2)                                   [N,128,  4,  4]

    Flatten -> Linear(128*4*4 -> 256) -> ReLU -> Dropout
    Linear(256 -> 10)                              logits over 10 classes
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # ---- Block 1 ----
        # Conv2d sweeps a small kernel (3x3) across the image. 
        # For each spatial position it computes a weighted sum over a 3x3 patch and all input channels, 
        # producing one number per output channel. 
        # With 32 output channels we're learning 32 different "what should I look for in a 3x3 patch?" detectors. 
        # padding=1 keeps the H/W dims constant
        self.conv1a = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # ---- Block 2 ----
        # Stacking conv layers expands the *receptive field*: a single 3x3 conv sees a 3x3 window of the input,
        # but two stacked 3x3 convs effectively see a 5x5 window of the layer before them. 
        # Widening from 32 -> 64 channels lets the network learn more distinct mid-level features.
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # ---- Block 3 ----
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # ReLU: f(x) = max(0, x). 
        # Nonlinearity that breaks linearity between conv layers 
        # (without it, stacked convs collapse to a single linear op) and gives sparse activations.
        self.relu = nn.ReLU(inplace=True)

        # MaxPool(2x2): takes the max over each 2x2 spatial window and halves H and W. 
        # This (a) cuts compute for later layers, 
        # (b) gives a bit of translation invariance ("the feature fired *somewhere* in this 2x2 patch"), and 
        # (c) is what lets the receptive field grow exponentially with depth instead of linearly.
        self.pool = nn.MaxPool2d(2, 2)

        # After 3 poolings: 32 -> 16 -> 8 -> 4. So feature map is [N, 128, 4, 4].
        self.flatten = nn.Flatten()

        # Linear classifier head. 
        # The convolutional trunk has produced 128*4*4 = 2048 features per image.
        # The FC layers learn to combine those features into class scores.
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)

        # Block 2
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)

        # Block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # Head
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        # Note: we return raw logits, not softmax probabilities.
        # nn.CrossEntropyLoss applies log-softmax internally for numerical stability, 
        # so the convention is loss-fn-eats-logits.
        return self.fc2(x)

    @torch.no_grad()
    def forward_with_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the forward pass and return every intermediate feature map."""
        acts: dict[str, torch.Tensor] = {"input": x.detach().cpu()}

        x = self.relu(self.conv1a(x)); acts["conv1a"] = x.detach().cpu()
        x = self.relu(self.conv1b(x)); acts["conv1b"] = x.detach().cpu()
        x = self.pool(x);              acts["pool1"]  = x.detach().cpu()

        x = self.relu(self.conv2a(x)); acts["conv2a"] = x.detach().cpu()
        x = self.relu(self.conv2b(x)); acts["conv2b"] = x.detach().cpu()
        x = self.pool(x);              acts["pool2"]  = x.detach().cpu()

        x = self.relu(self.conv3(x));  acts["conv3"]  = x.detach().cpu()
        x = self.pool(x);              acts["pool3"]  = x.detach().cpu()

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        acts["fc1"] = x.detach().cpu()
        logits = self.fc2(x)
        acts["logits"] = logits.detach().cpu()
        return acts
