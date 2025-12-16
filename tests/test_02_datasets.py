import os
import pytest
import torch


class BaseTestDataset:
    DatasetClass = None
    root_key = None
    category = None
    image_size = 256
    test_label_set = {0, 1}

    def _create_dataset(self, config, split):
        return self.DatasetClass(
            root_dir=config[self.root_key],
            category=self.category,
            split=split,
            image_size=self.image_size,
        )

    def test_dataset_class_exists(self):
        assert self.DatasetClass is not None

    def test_invalid_category_raises_error(self, config):
        with pytest.raises(FileNotFoundError):
            self.DatasetClass(
                root_dir=config[self.root_key],
                category="__invalid_category__",
                split="train",
                image_size=self.image_size,
            )

    def test_invalid_split_raises_error(self, config):
        with pytest.raises(ValueError):
            self.DatasetClass(
                root_dir=config[self.root_key],
                category=self.category,
                split="__invalid_split__",
                image_size=self.image_size,
            )

    def test_train_dataset_initialization(self, config):
        dataset = self._create_dataset(config, "train")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")

    def test_test_dataset_initialization(self, config):
        dataset = self._create_dataset(config, "test")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")

    def test_train_labels(self, config):
        dataset = self._create_dataset(config, "train")

        if len(dataset) == 0:
            pytest.skip("Empty train dataset")

        labels = {dataset[i]["label"] for i in range(min(10, len(dataset)))}
        assert labels == {0}

    def test_test_labels(self, config):
        dataset = self._create_dataset(config, "test")

        if len(dataset) == 0:
            pytest.skip("Empty test dataset")

        labels = {dataset[i]["label"] for i in range(min(20, len(dataset)))}
        assert labels.issubset(self.test_label_set)

    def test_getitem_output_format(self, config):
        dataset = self._create_dataset(config, "train")

        if len(dataset) == 0:
            pytest.skip("Empty dataset")

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"image", "label", "path"}

        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].ndim == 3
        assert sample["image"].shape[1:] == (self.image_size, self.image_size)

        assert sample["label"] in (0, 1)
        assert isinstance(sample["path"], str)
        assert os.path.exists(sample["path"])


class TestMVTecDataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import MVTecDataset
        cls.DatasetClass = MVTecDataset
        cls.root_key = "MVTec_DIR"
        cls.category = "bottle"
        cls.test_label_set = {0, 1}


class TestViSADataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import ViSADataset
        cls.DatasetClass = ViSADataset
        cls.root_key = "VISA_DIR"
        cls.category = "candle"
        cls.test_label_set = {1}


class TestBTADDataset(BaseTestDataset):
    @classmethod
    def setup_class(cls):
        from anomaly_detection.datasets import BTADDataset
        cls.DatasetClass = BTADDataset
        cls.root_key = "BTAD_DIR"
        cls.category = "01"
        cls.test_label_set = {0, 1}
