import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import timm

from .custom_cnn import CustomCNN

MODEL_NAMES = ["custom_cnn", "resnet18", "vit_s"]


def build_model(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "vit_s":
        m = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        return m
    raise ValueError(f"unknown model: {name}")


def target_layer_for_cam(model: nn.Module, name: str) -> nn.Module:
    if name == "custom_cnn":
        return model.last_conv_layer()
    if name == "resnet18":
        return model.layer4[-1]
    if name == "vit_s":
        return model.blocks[-1].norm1
    raise ValueError(f"unknown model: {name}")


def penultimate_features(model: nn.Module, name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "custom_cnn":
        f = model.features(x)
        f = model.pool(f).flatten(1)
        return f
    if name == "resnet18":
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x).flatten(1)
        return x
    if name == "vit_s":
        return model.forward_features(x)[:, 0]
    raise ValueError(f"unknown model: {name}")
