# tests/phase_01/test_datasets.py
import os
import pytest
import torch
import torchvision.transforms as T


class BaseTestDataset:
    """Base test class for all dataset implementations.
    
    Subclasses must set:
        - DatasetClass: The dataset class to test
        - root_key: Config key for dataset root directory
        - category: Test category name
        - test_label_set: Expected label set in test split
    """
    DatasetClass = None
    root_key = None
    category = None
    test_label_set = {0, 1}
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    mask_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    img_size = 256

    def _create_dataset(self, config, split):
        """Helper to create dataset instance."""
        return self.DatasetClass(
            root_dir=config[self.root_key],
            category=self.category,
            split=split,
            transform=self.transform,
            mask_transform=self.mask_transform,
        )

    # =========================================================
    # Basic Tests
    # =========================================================

    def test_dataset_class_exists(self):
        """Test that dataset class is properly defined."""
        assert self.DatasetClass is not None, "DatasetClass must be set in subclass"

    def test_dataset_has_categories(self):
        """Test that dataset class has CATEGORIES constant."""
        assert hasattr(self.DatasetClass, 'CATEGORIES'), "Dataset must have CATEGORIES attribute"
        assert isinstance(self.DatasetClass.CATEGORIES, list), "CATEGORIES must be a list"
        assert len(self.DatasetClass.CATEGORIES) > 0, "CATEGORIES must not be empty"

    # =========================================================
    # Initialization Tests
    # =========================================================

    def test_train_dataset_initialization(self, config, split):
        """Test train split initialization."""
        dataset = self._create_dataset(config, split)
        assert hasattr(dataset, "__len__"), "Dataset must implement __len__"
        assert hasattr(dataset, "__getitem__"), "Dataset must implement __getitem__"
        assert dataset.split == split
        assert dataset.category == self.category

    # def test_test_dataset_initialization(self, config):
    #     """Test test split initialization."""
    #     dataset = self._create_dataset(config, "test")
    #     assert hasattr(dataset, "__len__"), "Dataset must implement __len__"
    #     assert hasattr(dataset, "__getitem__"), "Dataset must implement __getitem__"
    #     assert dataset.split == "test"
    #     assert dataset.category == self.category

    def test_invalid_split_raises_error(self, config):
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
            self.DatasetClass(
                root_dir=config[self.root_key],
                category=self.category,
                split="invalid_split",
                transform=self.transform,
                mask_transform=self.mask_transform,
            )

    # =========================================================
    # Dataset Length Tests
    # =========================================================

    def test_train_dataset_not_empty(self, config, split):
        """Test that train dataset is not empty."""
        dataset = self._create_dataset(config, split)
        assert len(dataset) > 0, "Train dataset should not be empty"

    # def test_test_dataset_not_empty(self, config):
    #     """Test that test dataset is not empty."""
    #     dataset = self._create_dataset(config, "test")
    #     assert len(dataset) > 0, "Test dataset should not be empty"

    def test_dataset_len_matches_samples(self, config):
        """Test that __len__ matches number of samples."""
        dataset = self._create_dataset(config, "train")
        assert len(dataset) == len(dataset.samples), "__len__ should equal len(samples)"

    # =========================================================
    # Sample Structure Tests
    # =========================================================

    def test_train_sample_structure(self, config):
        """Test train sample dictionary structure."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        sample = dataset.samples[0]
        
        assert isinstance(sample, dict), "Sample must be a dictionary"
        assert "image_path" in sample, "Sample must have 'image_path'"
        assert "label" in sample, "Sample must have 'label'"
        assert "defect_type" in sample, "Sample must have 'defect_type'"
        assert "mask_path" in sample, "Sample must have 'mask_path'"

    def test_test_sample_structure(self, config):
        """Test test sample dictionary structure."""
        dataset = self._create_dataset(config, "test")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        sample = dataset.samples[0]
        
        assert isinstance(sample, dict), "Sample must be a dictionary"
        assert "image_path" in sample, "Sample must have 'image_path'"
        assert "label" in sample, "Sample must have 'label'"
        assert "defect_type" in sample, "Sample must have 'defect_type'"
        assert "mask_path" in sample, "Sample must have 'mask_path'"

    # =========================================================
    # __getitem__ Output Tests
    # =========================================================

    def test_getitem_output_format(self, config):
        """Test __getitem__ returns correct format."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[0]
        
        # Check output is dictionary
        assert isinstance(item, dict), "__getitem__ must return dict"
        
        # Check required keys
        required_keys = {"image", "label", "defect_type", "mask"}
        assert set(item.keys()) == required_keys, f"Keys must be {required_keys}"

    def test_getitem_image_format(self, config):
        """Test __getitem__ returns correct image format."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[0]
        
        # Check image type and shape
        assert isinstance(item["image"], torch.Tensor), "Image must be torch.Tensor"
        assert item["image"].ndim == 3, "Image must be 3D (C, H, W)"
        assert item["image"].shape[0] == 3, "Image must have 3 channels (RGB)"
        assert item["image"].shape[1:] == (self.img_size, self.img_size), \
            f"Image size must be ({self.img_size}, {self.img_size})"

    def test_getitem_label_format(self, config):
        """Test __getitem__ returns correct label format."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[0]
        
        # Check label type
        assert isinstance(item["label"], torch.Tensor), "Label must be torch.Tensor"
        assert item["label"].dtype == torch.long, "Label must be torch.long"
        assert item["label"].ndim == 0, "Label must be scalar tensor"

    def test_getitem_defect_type_format(self, config):
        """Test __getitem__ returns correct defect_type format."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[0]
        
        # Check defect_type type
        assert isinstance(item["defect_type"], str), "defect_type must be string"
        assert len(item["defect_type"]) > 0, "defect_type must not be empty"

    def test_getitem_mask_format_normal(self, config):
        """Test __getitem__ returns None mask for normal samples."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[0]
        
        # Train samples should have no mask
        assert item["mask"] is None, "Train (normal) samples should have None mask"

    # =========================================================
    # Label Tests
    # =========================================================

    def test_train_labels_all_normal(self, config):
        """Test that all train labels are 0 (normal)."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        # Check first 10 samples (or all if less)
        num_check = min(10, len(dataset))
        for i in range(num_check):
            item = dataset[i]
            assert item["label"] == 0, f"Train sample {i} must have label 0"

    def test_test_labels_valid_range(self, config):
        """Test that test labels are in valid range."""
        dataset = self._create_dataset(config, "test")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        # Collect unique labels
        labels = {int(dataset[i]["label"]) for i in range(min(20, len(dataset)))}
        
        # Check labels are subset of expected
        assert labels.issubset(self.test_label_set), \
            f"Labels {labels} must be subset of {self.test_label_set}"

    def test_train_defect_type_normal(self, config):
        """Test that train defect types are 'normal'."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        # Check first 10 samples
        num_check = min(10, len(dataset))
        for i in range(num_check):
            item = dataset[i]
            assert item["defect_type"] == "normal", \
                f"Train sample {i} must have defect_type 'normal'"

    # =========================================================
    # Image Path Tests
    # =========================================================

    def test_image_paths_exist(self, config):
        """Test that image paths in samples exist."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        # Check first 10 samples
        num_check = min(10, len(dataset))
        for i in range(num_check):
            sample = dataset.samples[i]
            assert os.path.exists(sample["image_path"]), \
                f"Image path does not exist: {sample['image_path']}"

    def test_image_paths_are_absolute(self, config):
        """Test that image paths are absolute paths."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        sample = dataset.samples[0]
        assert os.path.isabs(sample["image_path"]), \
            "Image path must be absolute path"

    # =========================================================
    # Mask Tests
    # =========================================================

    def test_normal_samples_no_mask(self, config):
        """Test that normal samples have None mask_path."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        # All train samples should be normal
        for sample in dataset.samples[:10]:
            if sample["label"] == 0:
                assert sample["mask_path"] is None, \
                    "Normal samples must have None mask_path"

    # =========================================================
    # Transform Tests
    # =========================================================

    def test_dataset_without_transform(self, config):
        """Test dataset works without transform."""
        dataset = self.DatasetClass(
            root_dir=config[self.root_key],
            category=self.category,
            split="train",
            transform=None,
            mask_transform=None,
        )
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[0]
        assert isinstance(item["image"], torch.Tensor), \
            "Image should still be tensor with default ToTensor()"

    def test_dataset_with_custom_transform(self, config):
        """Test dataset with custom transform."""
        custom_transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])
        
        dataset = self.DatasetClass(
            root_dir=config[self.root_key],
            category=self.category,
            split="train",
            transform=custom_transform,
            mask_transform=custom_transform,
        )
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[0]
        assert item["image"].shape[1:] == (128, 128), \
            "Image should have custom transform size"

    # =========================================================
    # Helper Method Tests
    # =========================================================

    def test_count_normal_method(self, config):
        """Test count_normal() method."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        count = dataset.count_normal()
        assert isinstance(count, int), "count_normal must return int"
        assert count >= 0, "count_normal must be non-negative"
        
        # For train split, all should be normal
        assert count == len(dataset), "Train dataset should have all normal samples"

    def test_count_anomaly_method(self, config):
        """Test count_anomaly() method."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        count = dataset.count_anomaly()
        assert isinstance(count, int), "count_anomaly must return int"
        assert count >= 0, "count_anomaly must be non-negative"
        
        # For train split, should be 0
        assert count == 0, "Train dataset should have no anomaly samples"

    def test_count_methods_sum(self, config):
        """Test that count_normal + count_anomaly equals total."""
        dataset = self._create_dataset(config, "test")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        total = len(dataset)
        normal_count = dataset.count_normal()
        anomaly_count = dataset.count_anomaly()
        
        assert normal_count + anomaly_count == total, \
            "count_normal + count_anomaly must equal total samples"

    # =========================================================
    # Indexing Tests
    # =========================================================

    def test_dataset_indexing_first(self, config):
        """Test accessing first item."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[0]
        assert item is not None

    def test_dataset_indexing_last(self, config):
        """Test accessing last item."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item = dataset[len(dataset) - 1]
        assert item is not None

    def test_dataset_indexing_out_of_range(self, config):
        """Test that out of range index raises error."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

    # =========================================================
    # Consistency Tests
    # =========================================================

    def test_multiple_getitem_same_index(self, config):
        """Test that multiple calls to __getitem__ with same index return consistent results."""
        dataset = self._create_dataset(config, "train")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        item1 = dataset[0]
        item2 = dataset[0]
        
        # Should have same label and defect_type
        assert item1["label"] == item2["label"]
        assert item1["defect_type"] == item2["defect_type"]
        
        # Images should have same shape
        assert item1["image"].shape == item2["image"].shape


# =========================================================
# Dataset-Specific Test Classes
# =========================================================

class TestMVTecDataset(BaseTestDataset):
    """Test class for MVTec AD dataset."""
    
    @classmethod
    def setup_class(cls):
        from anomaly_detection.data.datasets import MVTecDataset
        cls.DatasetClass = MVTecDataset
        cls.root_key = "MVTEC_DIR"
        cls.category = "bottle"
        cls.test_label_set = {0, 1}

    def test_mvtec_specific_categories(self):
        """Test MVTec has 15 categories."""
        assert len(self.DatasetClass.CATEGORIES) == 15

    def test_mvtec_ground_truth_path(self, config):
        """Test MVTec anomaly samples have ground truth masks."""
        dataset = self._create_dataset(config, "test")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        # Find an anomaly sample
        for sample in dataset.samples:
            if sample["label"] == 1:
                assert sample["mask_path"] is not None, \
                    "Anomaly samples must have mask_path"
                assert os.path.exists(sample["mask_path"]), \
                    f"Mask file must exist: {sample['mask_path']}"
                break


class TestViSADataset(BaseTestDataset):
    """Test class for ViSA dataset."""
    
    @classmethod
    def setup_class(cls):
        from anomaly_detection.data.datasets import ViSADataset
        cls.DatasetClass = ViSADataset
        cls.root_key = "VISA_DIR"
        cls.category = "candle"
        cls.test_label_set = {0, 1}

    def test_visa_specific_categories(self):
        """Test ViSA has 12 categories."""
        assert len(self.DatasetClass.CATEGORIES) == 12

    def test_visa_csv_loaded(self, config):
        """Test ViSA dataset loads CSV successfully."""
        dataset = self._create_dataset(config, "train")
        assert hasattr(dataset, 'df'), "ViSA dataset must have df attribute"
        assert len(dataset.df) > 0, "DataFrame must not be empty"

    def test_visa_csv_filtering(self, config):
        """Test ViSA dataset filters by category correctly."""
        dataset = self._create_dataset(config, "train")
        
        # All rows should have the specified category
        categories = dataset.df["object"].unique()
        assert len(categories) == 1, "Should only have one category"
        assert categories[0] == self.category, \
            f"Category should be {self.category}"


class TestBTADDataset(BaseTestDataset):
    """Test class for BTAD dataset."""
    
    @classmethod
    def setup_class(cls):
        from anomaly_detection.data.datasets import BTADDataset
        cls.DatasetClass = BTADDataset
        cls.root_key = "BTAD_DIR"
        cls.category = "01"
        cls.test_label_set = {0, 1}

    def test_btad_specific_categories(self):
        """Test BTAD has 3 categories."""
        assert len(self.DatasetClass.CATEGORIES) == 3

    def test_btad_ground_truth_path(self, config):
        """Test BTAD anomaly samples have ground truth masks."""
        dataset = self._create_dataset(config, "test")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        # Find an anomaly sample
        for sample in dataset.samples:
            if sample["label"] == 1:
                assert sample["mask_path"] is not None, \
                    "Anomaly samples must have mask_path"
                # Note: mask file might not exist for all samples in BTAD
                break