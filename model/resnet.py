import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return F.relu(out)


def _make_stage(in_c, out_c, blocks, stride):
    layers = [ResBlock(in_c, out_c, stride)]
    for _ in range(blocks - 1):
        layers.append(ResBlock(out_c, out_c))
    return nn.Sequential(*layers)


# He et al. 2015, "Deep Residual Learning for Image Recognition"
# https://arxiv.org/abs/1512.03385
# ResNet-20 is the CIFAR-specific variant defined in section 4.2 of the paper (n=3, 3 stages of 3 blocks)
class ResNet20(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(16, 16, 3, 1)
        self.stage2 = _make_stage(16, 32, 3, 2)
        self.stage3 = _make_stage(32, 64, 3, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        x = self.gap(f3).flatten(1)
        logits = self.fc(x)
        return logits, f1, f2, f3

    @staticmethod
    def export_outputs():
        return ['logits', 'feat_stage1', 'feat_stage2', 'feat_stage3']
