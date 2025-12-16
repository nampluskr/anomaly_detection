import pytest
from anomaly_detection.config import load_config


@pytest.fixture(scope="session")
def config():
    return load_config("configs/paths.yaml")
