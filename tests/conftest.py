# tests/conftest.py
import os
import pytest


@pytest.fixture(scope="session")
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def config_dir(project_root):
    return os.path.join(project_root, "configs")


@pytest.fixture(scope="session")
def source_dir(project_root):
    return os.path.join(project_root, "src")


@pytest.fixture(scope="session")
def output_dir(project_root):
    return os.path.join(project_root, "outputs")


@pytest.fixture(scope="function")
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"