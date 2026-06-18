"""
We use torch.Tensor and torch.nn.Parameter so autograd handles the backward pass, 
but every forward is written out in terms of slicing, unfolding, and matmul

A conv is "extract sliding windows, flatten each window, multiply by the kernel matrix".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    """2D convolution implemented via im2col + matmul

    Forward:
        1. Unfold the input into sliding windows: (N, C_in, H, W) -> (N, C_in*kH*kW, L)
           where L = H_out * W_out.
        2. Reshape weights to (C_out, C_in*kH*kW).
        3. Matmul: (C_out, K) @ (N, K, L) -> (N, C_out, L), then reshape to (N, C_out, H_out, W_out).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming-ish init for ReLU networks
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c_in, h, w = x.shape
        kh = kw = self.kernel_size
        h_out = (h + 2 * self.padding - kh) // self.stride + 1
        w_out = (w + 2 * self.padding - kw) // self.stride + 1

        # unfold returns (N, C_in*kH*kW, L) where L = H_out * W_out
        cols = torch.nn.functional.unfold(
            x, kernel_size=kh, padding=self.padding, stride=self.stride
        )
        # Weights as a (C_out, C_in*kH*kW) matrix
        w_mat = self.weight.view(self.out_channels, -1)
        # (C_out, K) @ (N, K, L) -> (N, C_out, L)
        out = w_mat @ cols
        out = out + self.bias.view(1, -1, 1)
        return out.view(n, self.out_channels, h_out, w_out)


class MaxPool2d(nn.Module):
    """Max pooling via unfold + max-over-window."""

    def __init__(self, kernel_size: int, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        k = self.kernel_size
        h_out = (h - k) // self.stride + 1
        w_out = (w - k) // self.stride + 1
        # unfold to (N, C*k*k, L), then split channel dim out
        cols = torch.nn.functional.unfold(x, kernel_size=k, stride=self.stride)
        cols = cols.view(n, c, k * k, h_out * w_out)
        out, _ = cols.max(dim=2)
        return out.view(n, c, h_out, w_out)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t() + self.bias


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class CrossEntropyLoss(nn.Module):
    """Softmax + NLL, written out so the math is visible.

    Uses the log-sum-exp trick for numerical stability.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), targets: (N,) of class indices
        m = logits.max(dim=1, keepdim=True).values
        log_sum_exp = m.squeeze(1) + (logits - m).exp().sum(dim=1).log()
        true_logit = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (log_sum_exp - true_logit).mean()
