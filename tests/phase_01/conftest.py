# tests/phase_01/conftest.py
import pytest
import torchvision.transforms as T


@pytest.fixture(scope="module")
def default_transform():
    """Default transform for Phase 01 tests."""
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])


@pytest.fixture(scope="module")
def default_mask_transform():
    """Default mask transform for Phase 01 tests."""
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])


@pytest.fixture(params=["bottle", "cable", "capsule"])
def mvtec_category(request):
    """Parametrize MVTec categories."""
    return request.param


@pytest.fixture(params=["candle", "capsules", "cashew"])
def visa_category(request):
    """Parametrize ViSA categories."""
    return request.param


@pytest.fixture(params=["01", "02", "03"])
def btad_category(request):
    """Parametrize BTAD categories."""
    return request.param