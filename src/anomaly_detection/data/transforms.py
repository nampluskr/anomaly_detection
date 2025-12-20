# src/anomaly_detection/data/transforms.py

import torchvision.transforms as T


def get_train_transforms(image_size=256, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_test_transforms(image_size=256, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_mask_transforms(image_size=256):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])