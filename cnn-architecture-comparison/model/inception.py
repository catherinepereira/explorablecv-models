import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_c, c1, c3r, c3, c5r, c5, pool_c):
        super().__init__()
        self.b1 = nn.Conv2d(in_c, c1, 1)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c, c3r, 1), nn.ReLU(inplace=True), nn.Conv2d(c3r, c3, 3, padding=1)
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(in_c, c5r, 1), nn.ReLU(inplace=True), nn.Conv2d(c5r, c5, 5, padding=2)
        )
        self.bp = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_c, pool_c, 1)
        )

    def forward(self, x):
        return F.relu(torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1))


# Szegedy et al. 2014, "Going Deeper with Convolutions" (GoogLeNet / Inception v1)
# https://arxiv.org/abs/1409.4842
# Reduced-depth variant for CIFAR: 4 Inception blocks instead of 9, no auxiliary classifiers
class InceptionMini(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.inc1 = InceptionBlock(64, 32, 32, 64, 8, 16, 16)
        self.inc2 = InceptionBlock(128, 64, 48, 96, 16, 32, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.inc3 = InceptionBlock(224, 96, 56, 112, 16, 32, 32)
        self.inc4 = InceptionBlock(272, 128, 64, 128, 24, 48, 48)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(352, num_classes)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.inc1(x)
        f2 = self.inc2(f1)
        x = self.pool(f2)
        f3 = self.inc3(x)
        f4 = self.inc4(f3)
        x = self.gap(f4).flatten(1)
        logits = self.fc(x)
        return logits, f1, f2, f4

    @staticmethod
    def export_outputs():
        return ['logits', 'feat_inc1', 'feat_inc2', 'feat_inc4']
