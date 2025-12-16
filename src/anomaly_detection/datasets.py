import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class BaseDataset(Dataset):
    def __init__(self, root_dir, category, split, image_size):
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.image_size = image_size

        self.image_paths = []
        self.labels = []

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": path,
        }


# =========================================================
# MVTec
# =========================================================

class MVTecDataset(BaseDataset):
    def __init__(self, root_dir, category, split, image_size):
        self.category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(self.category_dir):
            raise FileNotFoundError(f"MVTec category not found: {category}")

        super().__init__(root_dir, category, split, image_size)

        if split == "train":
            self._load_train()
        else:
            self._load_test()

    def _load_train(self):
        good_dir = os.path.join(self.category_dir, "train", "good")
        if not os.path.isdir(good_dir):
            return

        for fname in sorted(os.listdir(good_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(os.path.join(good_dir, fname))
                self.labels.append(0)

    def _load_test(self):
        test_dir = os.path.join(self.category_dir, "test")
        if not os.path.isdir(test_dir):
            return

        for defect in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect)
            if not os.path.isdir(defect_dir):
                continue

            label = 0 if defect == "good" else 1

            for fname in sorted(os.listdir(defect_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(defect_dir, fname))
                    self.labels.append(label)


# =========================================================
# ViSA
# =========================================================

class ViSADataset(BaseDataset):
    def __init__(self, root_dir, category, split, image_size):
        self.category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(self.category_dir):
            raise FileNotFoundError(f"ViSA category not found: {category}")

        super().__init__(root_dir, category, split, image_size)
        self._load()

    def _load(self):
        split_dir = os.path.join(self.category_dir, self.split)
        if not os.path.isdir(split_dir):
            return

        for fname in sorted(os.listdir(split_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(split_dir, fname)
                self.image_paths.append(path)
                self.labels.append(0 if self.split == "train" else 1)


# =========================================================
# BTAD
# =========================================================

class BTADDataset(BaseDataset):
    def __init__(self, root_dir, category, split, image_size):
        self.category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(self.category_dir):
            raise FileNotFoundError(f"BTAD category not found: {category}")

        super().__init__(root_dir, category, split, image_size)

        if split == "train":
            self._load_train()
        else:
            self._load_test()

    def _load_train(self):
        train_dir = os.path.join(self.category_dir, "train", "ok")
        if not os.path.isdir(train_dir):
            return

        for fname in sorted(os.listdir(train_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(os.path.join(train_dir, fname))
                self.labels.append(0)

    def _load_test(self):
        test_dir = os.path.join(self.category_dir, "test")
        if not os.path.isdir(test_dir):
            return

        for defect in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect)
            if not os.path.isdir(defect_dir):
                continue

            label = 0 if defect == "ok" else 1

            for fname in sorted(os.listdir(defect_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(defect_dir, fname))
                    self.labels.append(label)
