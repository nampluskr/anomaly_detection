import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) 
SOURCE_DIR = os.path.abspath(os.path.join(ROOT_DIR, "src"))

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

import torch
import torchvision.transforms as T

from anomaly_detection.datasets import MVTecDataset
from anomaly_detection.config import load_config

if __name__ == "__main__":
    config = load_config(os.path.join(ROOT_DIR, "configs", "paths.yaml"))

    train_dataset = MVTecDataset(
        root_dir=config["MVTec_DIR"],
        category="bottle",
        split="train",
        transform=T.Compose([T.Resize((256, 256)), T.ToTensor()]),
        mask_transform=T.Compose([T.Resize((256, 256)), T.ToTensor()]),
    )

    data = train_dataset[10]
    image = data["image"].permute(1, 2, 0).numpy()
    label = data["label"].numpy()
    defect_type = data["defect_type"]
    mask = None if data["mask"] is None else data["mask"].squeeze().numpy()

    print("\n*** Train Dataset ***")
    print(f">> image: {image.shape}")
    print(f">> label: {label}")
    print(f">> label: {defect_type}")
    print(f">> label: {None if mask is None else mask.shape}")


    test_dataset = MVTecDataset(
        root_dir=config["MVTec_DIR"],
        category="bottle",
        split="test",
        transform=T.Compose([T.Resize((256, 256)), T.ToTensor()]),
        mask_transform=T.Compose([T.Resize((256, 256)), T.ToTensor()]),
    )

    data = test_dataset[40]
    image = data["image"].permute(1, 2, 0).numpy()
    label = data["label"].numpy()
    defect_type = data["defect_type"]
    mask = None if data["mask"] is None else data["mask"].squeeze().numpy()

    print("\n*** Test Dataset ***")
    print(f">> image: {image.shape}")
    print(f">> label: {label}")
    print(f">> label: {defect_type}")
    print(f">> label: {None if mask is None else mask.shape}")