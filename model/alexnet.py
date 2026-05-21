import torch.nn as nn
import torch.nn.functional as F


# Krizhevsky, Sutskever, Hinton 2012, "ImageNet Classification with Deep Convolutional Neural Networks"
# https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
# CIFAR-scale adaptation: 3x3 kernels and smaller FC layers for 32x32 input
class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        f1 = F.relu(self.conv1(x))
        x = self.pool(f1)
        f2 = F.relu(self.conv2(x))
        x = self.pool(f2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        f5 = F.relu(self.conv5(x))
        x = self.pool(f5)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        logits = self.fc3(x)
        return logits, f1, f2, f5

    @staticmethod
    def export_outputs():
        return ['logits', 'feat_conv1', 'feat_conv2', 'feat_conv5']
