import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_c, growth):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, 4 * growth, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth)
        self.conv2 = nn.Conv2d(4 * growth, growth, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_c, num_layers, growth):
        super().__init__()
        layers = []
        c = in_c
        for _ in range(num_layers):
            layers.append(DenseLayer(c, growth))
            c += growth
        self.block = nn.Sequential(*layers)
        self.out_c = c

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


# Huang et al. 2016, "Densely Connected Convolutional Networks"
# https://arxiv.org/abs/1608.06993
# DenseNet-BC for CIFAR (3 dense blocks of 6/12/24 layers, bottleneck + compression 0.5)
class DenseNetBC(nn.Module):
    def __init__(self, num_classes: int = 10, growth: int = 12):
        super().__init__()
        self.stem = nn.Conv2d(3, 2 * growth, 3, padding=1, bias=False)
        self.db1 = DenseBlock(2 * growth, 6, growth)
        self.t1 = Transition(self.db1.out_c, self.db1.out_c // 2)
        self.db2 = DenseBlock(self.db1.out_c // 2, 12, growth)
        self.t2 = Transition(self.db2.out_c, self.db2.out_c // 2)
        self.db3 = DenseBlock(self.db2.out_c // 2, 24, growth)
        self.bn = nn.BatchNorm2d(self.db3.out_c)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.db3.out_c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.db1(x)
        x = self.t1(f1)
        f2 = self.db2(x)
        x = self.t2(f2)
        f3 = self.db3(x)
        x = self.gap(F.relu(self.bn(f3))).flatten(1)
        logits = self.fc(x)
        return logits, f1, f2, f3

    @staticmethod
    def export_outputs():
        return ['logits', 'feat_db1', 'feat_db2', 'feat_db3']
