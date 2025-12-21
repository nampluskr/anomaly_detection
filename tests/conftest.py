# tests/conftest.py
import os
import pytest
import yaml
import torch


# =========================================================
# Path Fixtures
# =========================================================

@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(tests_dir)


@pytest.fixture(scope="session")
def source_dir(project_root):
    """Get source directory."""
    return os.path.join(project_root, "src")


@pytest.fixture(scope="session")
def config_dir(project_root):
    """Get config directory."""
    return os.path.join(project_root, "configs")


@pytest.fixture(scope="session")
def output_dir(project_root):
    """Get output directory."""
    return os.path.join(project_root, "outputs")


# =========================================================
# Config Fixtures
# =========================================================

@pytest.fixture(scope="session")
def config(config_dir):
    """Load test configuration"""
    from anomaly_detection.config import load_config
    return load_config(os.path.join(config_dir, "paths.yaml"))


# =========================================================
# Device Fixtures
# =========================================================

@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


# =========================================================
# Parametrized Fixtures
# =========================================================

@pytest.fixture(params=["train", "test"])
def split(request):
    """Parametrize split (train/test)."""
    return request.param


@pytest.fixture(params=["bottle", "cable", "capsule"])
def mvtec_category(request):
    """Parametrize MVTec categories (sample)."""
    return request.param


# =========================================================
# Helper Functions (not fixtures)
# =========================================================

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "phase_00: Setup tests"
    )
    config.addinivalue_line(
        "markers", "phase_01: Data pipeline tests"
    )
    config.addinivalue_line(
        "markers", "requires_dataset: Tests requiring actual dataset"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )