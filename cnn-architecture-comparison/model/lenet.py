import torch.nn as nn
import torch.nn.functional as F


# LeCun et al. 1998, "Gradient-Based Learning Applied to Document Recognition"
# https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        f1 = F.relu(self.conv1(x))
        x = self.pool(f1)
        f2 = F.relu(self.conv2(x))
        x = self.pool(f2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits, f1, f2

    @staticmethod
    def export_outputs():
        return ['logits', 'feat_conv1', 'feat_conv2']
