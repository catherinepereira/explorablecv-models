import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# Simonyan & Zisserman 2014, "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# https://arxiv.org/abs/1409.1556
# VGG-11 (config A) with BatchNorm, adapted to CIFAR via global average pool head
class VGG11CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = conv_block(3, 64)
        self.block2 = conv_block(64, 128)
        self.block3a = conv_block(128, 256)
        self.block3b = conv_block(256, 256)
        self.block4a = conv_block(256, 512)
        self.block4b = conv_block(512, 512)
        self.block5a = conv_block(512, 512)
        self.block5b = conv_block(512, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        f1 = self.block1(x)
        x = self.pool(f1)
        f2 = self.block2(x)
        x = self.pool(f2)
        x = self.block3a(x)
        f3 = self.block3b(x)
        x = self.pool(f3)
        x = self.block4a(x)
        x = self.block4b(x)
        x = self.pool(x)
        x = self.block5a(x)
        f5 = self.block5b(x)
        x = self.gap(F.relu(f5)).flatten(1)
        logits = self.fc(x)
        return logits, f1, f2, f3, f5

    @staticmethod
    def export_outputs():
        return ['logits', 'feat_block1', 'feat_block2', 'feat_block3', 'feat_block5']
