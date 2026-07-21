# Per-channel (R, G, B) mean/std of the CIFAR-10 train set. train.py normalizes
# with these and export_onnx.py bakes them into model_meta.json for the
# frontend, so inference preprocessing stays in sync with training
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
]
