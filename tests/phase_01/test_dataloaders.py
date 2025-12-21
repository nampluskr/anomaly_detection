# tests/phase_01/test_dataloaders.py
import pytest
import torch
import torchvision.transforms as T


class DummyDataset:
    """Dummy dataset for testing dataloaders without real data."""
    
    def __init__(self, size=100, image_size=256):
        self.size = size
        self.image_size = image_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "label": torch.tensor(idx % 2).long(),
            "defect_type": "normal" if idx % 2 == 0 else "anomaly",
            "mask": None
        }


# =========================================================
# Base Test Class for DataLoaders
# =========================================================

class BaseTestDataloader:
    """Base test class for all dataloader implementations.
    
    Subclasses must set:
        - DatasetClass: The dataset class to use
        - root_key: Config key for dataset root directory
        - category: Test category name
    """
    DatasetClass = None
    root_key = None
    category = None
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
    # Train DataLoader Tests
    # =========================================================

    def test_train_dataloader_creation(self, config):
        """Test train dataloader creation."""
        from anomaly_detection.data.dataloaders import get_train_dataloader
        
        try:
            dataset = self._create_dataset(config, "train")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        dataloader = get_train_dataloader(dataset, batch_size=8)
        
        assert dataloader is not None
        assert dataloader.drop_last is True

    def test_train_dataloader_batch_content(self, config):
        """Test train dataloader batch content."""
        from anomaly_detection.data.dataloaders import get_train_dataloader
        
        try:
            dataset = self._create_dataset(config, "train")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        dataloader = get_train_dataloader(dataset, batch_size=8)
        batch = next(iter(dataloader))
        
        # Check batch structure
        assert isinstance(batch, dict)
        assert "image" in batch
        assert "label" in batch
        assert "defect_type" in batch
        assert "mask" in batch
        
        # Check batch dimensions
        assert batch["image"].ndim == 4  # (B, C, H, W)
        assert batch["label"].ndim == 1  # (B,)
        assert batch["image"].shape[0] <= 8  # batch_size or less

    def test_train_dataloader_iteration(self, config):
        """Test train dataloader iteration."""
        from anomaly_detection.data.dataloaders import get_train_dataloader
        
        try:
            dataset = self._create_dataset(config, "train")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        dataloader = get_train_dataloader(dataset, batch_size=8)
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert isinstance(batch, dict)
        
        assert batch_count > 0

    def test_train_dataloader_drops_last_batch(self, config):
        """Test that train dataloader drops incomplete last batch."""
        from anomaly_detection.data.dataloaders import get_train_dataloader
        
        try:
            dataset = self._create_dataset(config, "train")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        # Use batch_size that doesn't divide evenly if possible
        total_samples = len(dataset)
        batch_size = 7  # Likely to have remainder
        
        if total_samples <= batch_size:
            pytest.skip("Dataset too small for drop_last test")
        
        dataloader = get_train_dataloader(dataset, batch_size=batch_size)
        batches = list(dataloader)
        
        # All batches should have exactly batch_size samples (drop_last=True)
        for batch in batches:
            assert batch["image"].shape[0] == batch_size

    def test_train_dataloader_image_format(self, config):
        """Test train dataloader returns correct image format."""
        from anomaly_detection.data.dataloaders import get_train_dataloader
        
        try:
            dataset = self._create_dataset(config, "train")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        dataloader = get_train_dataloader(dataset, batch_size=4)
        batch = next(iter(dataloader))
        
        # Check image tensor
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[1] == 3  # RGB channels
        assert batch["image"].shape[2] == self.img_size
        assert batch["image"].shape[3] == self.img_size

    def test_train_dataloader_label_format(self, config):
        """Test train dataloader returns correct label format."""
        from anomaly_detection.data.dataloaders import get_train_dataloader
        
        try:
            dataset = self._create_dataset(config, "train")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty train dataset")
        
        dataloader = get_train_dataloader(dataset, batch_size=4)
        batch = next(iter(dataloader))
        
        # Check label tensor
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["label"].dtype == torch.long
        
        # All train labels should be 0 (normal)
        assert torch.all(batch["label"] == 0)

    # =========================================================
    # Test DataLoader Tests
    # =========================================================

    def test_test_dataloader_creation(self, config):
        """Test test dataloader creation."""
        from anomaly_detection.data.dataloaders import get_test_dataloader
        
        try:
            dataset = self._create_dataset(config, "test")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        dataloader = get_test_dataloader(dataset, batch_size=8)
        
        assert dataloader is not None
        assert dataloader.drop_last is False

    def test_test_dataloader_batch_content(self, config):
        """Test test dataloader batch content."""
        from anomaly_detection.data.dataloaders import get_test_dataloader
        
        try:
            dataset = self._create_dataset(config, "test")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        dataloader = get_test_dataloader(dataset, batch_size=8)
        batch = next(iter(dataloader))
        
        # Check batch structure
        assert isinstance(batch, dict)
        assert "image" in batch
        assert "label" in batch
        assert "defect_type" in batch
        assert "mask" in batch
        
        # Check batch dimensions
        assert batch["image"].ndim == 4  # (B, C, H, W)
        assert batch["label"].ndim == 1  # (B,)

    def test_test_dataloader_iteration(self, config):
        """Test test dataloader iteration."""
        from anomaly_detection.data.dataloaders import get_test_dataloader
        
        try:
            dataset = self._create_dataset(config, "test")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        dataloader = get_test_dataloader(dataset, batch_size=8)
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert isinstance(batch, dict)
        
        assert batch_count > 0

    def test_test_dataloader_keeps_last_batch(self, config):
        """Test that test dataloader keeps incomplete last batch."""
        from anomaly_detection.data.dataloaders import get_test_dataloader
        
        try:
            dataset = self._create_dataset(config, "test")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        total_samples = len(dataset)
        batch_size = 7  # Likely to have remainder
        
        dataloader = get_test_dataloader(dataset, batch_size=batch_size)
        batches = list(dataloader)
        
        # Calculate expected number of batches
        import math
        expected_batches = math.ceil(total_samples / batch_size)
        
        assert len(batches) == expected_batches

    def test_test_dataloader_image_format(self, config):
        """Test test dataloader returns correct image format."""
        from anomaly_detection.data.dataloaders import get_test_dataloader
        
        try:
            dataset = self._create_dataset(config, "test")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        dataloader = get_test_dataloader(dataset, batch_size=4)
        batch = next(iter(dataloader))
        
        # Check image tensor
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape[1] == 3  # RGB channels
        assert batch["image"].shape[2] == self.img_size
        assert batch["image"].shape[3] == self.img_size

    def test_test_dataloader_label_format(self, config):
        """Test test dataloader returns correct label format."""
        from anomaly_detection.data.dataloaders import get_test_dataloader
        
        try:
            dataset = self._create_dataset(config, "test")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty test dataset")
        
        dataloader = get_test_dataloader(dataset, batch_size=4)
        batch = next(iter(dataloader))
        
        # Check label tensor
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["label"].dtype == torch.long

    # =========================================================
    # Train vs Test Comparison Tests
    # =========================================================

    def test_train_test_dataloader_difference(self, config):
        """Test differences between train and test dataloaders."""
        from anomaly_detection.data.dataloaders import get_train_dataloader, get_test_dataloader
        
        try:
            train_dataset = self._create_dataset(config, "train")
            test_dataset = self._create_dataset(config, "test")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            pytest.skip("Empty dataset")
        
        train_loader = get_train_dataloader(train_dataset, batch_size=8)
        test_loader = get_test_dataloader(test_dataset, batch_size=8)
        
        # Train should drop last, test should not
        assert train_loader.drop_last is True
        assert test_loader.drop_last is False

    # =========================================================
    # Config Integration Tests
    # =========================================================

    def test_dataloader_with_config_batch_size(self, config):
        """Test dataloader using batch size from config."""
        from anomaly_detection.data.dataloaders import get_train_dataloader
        
        try:
            dataset = self._create_dataset(config, "train")
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{self.DatasetClass.__name__} dataset not available")
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        # Simulate config structure
        mock_config = {"dataloader": {"train": {"batch_size": 16}}}
        
        dataloader = get_train_dataloader(
            dataset,
            batch_size=mock_config["dataloader"]["train"]["batch_size"]
        )
        
        batch = next(iter(dataloader))
        assert batch["image"].shape[0] <= 16


# =========================================================
# Dataset-Specific Test Classes
# =========================================================

class TestMVTecDataloader(BaseTestDataloader):
    """Test class for MVTec AD dataloader."""
    
    @classmethod
    def setup_class(cls):
        from anomaly_detection.data.datasets import MVTecDataset
        cls.DatasetClass = MVTecDataset
        cls.root_key = "mvtec_dir"  # lowercase
        cls.category = "bottle"


class TestViSADataloader(BaseTestDataloader):
    """Test class for ViSA dataloader."""
    
    @classmethod
    def setup_class(cls):
        from anomaly_detection.data.datasets import ViSADataset
        cls.DatasetClass = ViSADataset
        cls.root_key = "visa_dir"  # lowercase
        cls.category = "candle"


class TestBTADDataloader(BaseTestDataloader):
    """Test class for BTAD dataloader."""
    
    @classmethod
    def setup_class(cls):
        from anomaly_detection.data.datasets import BTADDataset
        cls.DatasetClass = BTADDataset
        cls.root_key = "btad_dir"  # lowercase
        cls.category = "01"


# =========================================================
# Generic DataLoader Tests (using DummyDataset)
# =========================================================

def test_get_dataloader_basic():
    """Test basic dataloader creation."""
    from anomaly_detection.data.dataloaders import get_dataloader
    
    dataset = DummyDataset(size=32)
    dataloader = get_dataloader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    assert dataloader is not None
    assert len(dataloader) == 4  # 32 / 8 = 4 batches


def test_get_dataloader_with_batch_size():
    """Test dataloader creation with different batch sizes."""
    from anomaly_detection.data.dataloaders import get_dataloader
    
    dataset = DummyDataset(size=100)
    
    # Test various batch sizes
    for batch_size in [1, 8, 16, 32]:
        dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        expected_batches = (100 + batch_size - 1) // batch_size  # Ceiling division
        assert len(dataloader) == expected_batches


def test_get_train_dataloader_basic():
    """Test training dataloader creation."""
    from anomaly_detection.data.dataloaders import get_train_dataloader
    
    dataset = DummyDataset(size=32)
    dataloader = get_train_dataloader(dataset, batch_size=8)
    
    assert dataloader is not None
    assert dataloader.drop_last is True


def test_get_test_dataloader_basic():
    """Test test/validation dataloader creation."""
    from anomaly_detection.data.dataloaders import get_test_dataloader
    
    dataset = DummyDataset(size=32)
    dataloader = get_test_dataloader(dataset, batch_size=8)
    
    assert dataloader is not None
    assert dataloader.drop_last is False


def test_train_dataloader_drops_last_batch():
    """Test that training dataloader drops incomplete last batch."""
    from anomaly_detection.data.dataloaders import get_train_dataloader
    
    dataset = DummyDataset(size=35)  # 35 / 8 = 4 batches + 3 samples
    dataloader = get_train_dataloader(dataset, batch_size=8)
    
    batches = list(dataloader)
    
    # Should have 4 complete batches (drop last 3 samples)
    assert len(batches) == 4
    
    # All batches should have exactly 8 samples
    for batch in batches:
        assert batch["image"].shape[0] == 8


def test_test_dataloader_keeps_last_batch():
    """Test that test dataloader keeps incomplete last batch."""
    from anomaly_detection.data.dataloaders import get_test_dataloader
    
    dataset = DummyDataset(size=35)  # 35 / 8 = 4 batches + 3 samples
    dataloader = get_test_dataloader(dataset, batch_size=8)
    
    batches = list(dataloader)
    
    # Should have 5 batches (keep last 3 samples)
    assert len(batches) == 5
    
    # Last batch should have 3 samples
    assert batches[-1]["image"].shape[0] == 3


def test_dataloader_config_platform_specific():
    """Test that dataloader config is platform-specific."""
    from anomaly_detection.data.dataloaders import get_dataloader_config
    import platform
    
    config = get_dataloader_config()
    
    assert "num_workers" in config
    assert "pin_memory" in config
    assert "persistent_workers" in config
    
    if platform.system() == "Windows":
        assert config["num_workers"] == 0
        assert config["pin_memory"] is False
        assert config["persistent_workers"] is False
    else:  # Linux
        assert config["num_workers"] == 8
        assert config["pin_memory"] is True
        assert config["persistent_workers"] is True


def test_dataloader_collate_function():
    """Test that dataloader properly collates samples."""
    from anomaly_detection.data.dataloaders import get_dataloader
    
    dataset = DummyDataset(size=16, image_size=256)
    dataloader = get_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    batch = next(iter(dataloader))
    
    # Images should be stacked into tensor
    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape == (4, 3, 256, 256)
    
    # Labels should be stacked into tensor
    assert isinstance(batch["label"], torch.Tensor)
    assert batch["label"].shape == (4,)
    
    # defect_type should be list of strings
    assert isinstance(batch["defect_type"], (list, tuple))
    assert len(batch["defect_type"]) == 4


def test_backward_compatibility_aliases():
    """Test backward compatibility with get_train_loader and get_test_loader."""
    from anomaly_detection.data.dataloaders import get_train_loader, get_test_loader
    
    dataset = DummyDataset(size=32)
    
    train_loader = get_train_loader(dataset, batch_size=8)
    test_loader = get_test_loader(dataset, batch_size=8)
    
    assert train_loader is not None
    assert test_loader is not None
    assert train_loader.drop_last is True
    assert test_loader.drop_last is False


def test_dataloader_with_invalid_batch_size():
    """Test dataloader behavior with invalid batch size."""
    from anomaly_detection.data.dataloaders import get_dataloader
    
    dataset = DummyDataset(size=32)
    
    # batch_size must be positive
    with pytest.raises((ValueError, TypeError)):
        dataloader = get_dataloader(dataset, batch_size=0, shuffle=False, num_workers=0)


def test_dataloader_with_negative_num_workers():
    """Test dataloader behavior with negative num_workers."""
    from anomaly_detection.data.dataloaders import get_dataloader
    
    dataset = DummyDataset(size=32)
    
    # num_workers must be non-negative
    with pytest.raises((ValueError, TypeError)):
        dataloader = get_dataloader(dataset, batch_size=8, shuffle=False, num_workers=-1)