from .lenet import LeNet
from .alexnet import AlexNetCIFAR
from .vgg import VGG11CIFAR
from .inception import InceptionMini
from .resnet import ResNet20
from .densenet import DenseNetBC

ARCHITECTURES = {
    'lenet': LeNet,
    'alexnet': AlexNetCIFAR,
    'vgg': VGG11CIFAR,
    'inception': InceptionMini,
    'resnet': ResNet20,
    'densenet': DenseNetBC,
}
