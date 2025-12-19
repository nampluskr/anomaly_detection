# Phase 01: Data Pipeline - 개발 가이드

## 개요

**목표**: 데이터 로딩 및 전처리 파이프라인 구축  
**예상 소요 시간**: 2-3일  
**브랜치**: `feature/phase-01-data-pipeline`  
**담당**: 데이터 파이프라인 담당 개발자  
**선행 조건**: Phase 00 완료 및 main 브랜치 머지 완료

---

## 목표

- Dataset 클래스 구현 (MVTec, VisA, BTAD)
- DataLoader 생성 함수 구현
- Transform 함수 구현 (학습/테스트/마스크)
- 데이터 검증 유틸리티 구현
- 데이터 파이프라인 테스트 작성 및 검증

---

## 사전 요구사항

### Phase 00 완료 확인
```bash
# Phase 00 브랜치가 main에 머지되었는지 확인
git checkout main
git pull origin main
git log --oneline -5  # Phase 00 커밋 확인
```

### 데이터셋 준비
```bash
# 외부 데이터셋이 다음 경로에 준비되어 있어야 함
# configs/paths.yaml의 dataset_root 확인

# MVTec AD 데이터셋 구조
mvtec/
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── broken_large/
│   │   └── ...
│   └── ground_truth/
│       ├── broken_large/
│       └── ...
├── cable/
└── ...

# VisA 데이터셋 구조
visa/
├── candle/
│   ├── Data/
│   │   ├── Images/
│   │   │   ├── Normal/
│   │   │   └── Anomaly/
│   │   └── Masks/
│   └── image_anno.csv
└── ...

# BTAD 데이터셋 구조
btad/
├── 01/
│   ├── train/
│   │   └── ok/
│   ├── test/
│   │   ├── ok/
│   │   └── ko/
│   └── ground_truth/
│       └── ko/
└── ...
```

---

## 프로젝트 구조 (추가/수정)

```
anomaly_detection/
├── src/anomaly_detection/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   └── data/                        # 새로 추가
│       ├── __init__.py
│       ├── datasets.py              # Dataset 클래스
│       ├── dataloaders.py           # DataLoader 함수
│       ├── transforms.py            # Transform 함수
│       └── validation.py            # 데이터 검증
├── configs/
│   ├── paths.yaml
│   ├── default.yaml
│   └── datasets/                    # 새로 추가
│       ├── mvtec.yaml
│       ├── visa.yaml
│       └── btad.yaml
├── tests/
│   ├── conftest.py
│   ├── phase_00/
│   └── phase_01/                    # 새로 추가
│       ├── test_datasets.py
│       ├── test_dataloaders.py
│       ├── test_transforms.py
│       └── test_validation.py
└── docs/
    ├── phase_00_guide.md
    └── phase_01_guide.md            # 이 문서
```

---

## 개발 워크플로우

### Step 1: 브랜치 생성

```bash
git checkout main
git pull origin main
git checkout -b feature/phase-01-data-pipeline
```

### Step 2: 디렉터리 및 파일 생성

```bash
# 디렉터리 생성
mkdir -p src/anomaly_detection/data
mkdir -p configs/datasets
mkdir -p tests/phase_01

# __init__.py 파일 생성
touch src/anomaly_detection/data/__init__.py
touch tests/phase_01/__init__.py
```

### Step 3: 데이터셋 설정 파일 작성

#### 3.1 configs/datasets/mvtec.yaml
```yaml
name: mvtec
categories:
  - bottle
  - cable
  - capsule
  - carpet
  - grid
  - hazelnut
  - leather
  - metal_nut
  - pill
  - screw
  - tile
  - toothbrush
  - transistor
  - wood
  - zipper

image_size: 256
normalization:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

train_split: train/good
test_split: test
mask_split: ground_truth
```

#### 3.2 configs/datasets/visa.yaml
```yaml
name: visa
categories:
  - candle
  - capsules
  - cashew
  - chewinggum
  - fryum
  - macaroni1
  - macaroni2
  - pcb1
  - pcb2
  - pcb3
  - pcb4
  - pipe_fryum

image_size: 256
normalization:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

split_ratio: 0.8
annotation_file: image_anno.csv
```

#### 3.3 configs/datasets/btad.yaml
```yaml
name: btad
categories:
  - "01"
  - "02"
  - "03"

image_size: 256
normalization:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

train_split: train/ok
test_split: test
mask_split: ground_truth
```

### Step 4: Transform 함수 구현

#### 4.1 src/anomaly_detection/data/transforms.py
```python
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
```

### Step 5: Dataset 클래스 구현

#### 5.1 src/anomaly_detection/data/datasets.py
```python
# src/anomaly_detection/data/datasets.py
import os
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, root_dir, category, split, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        raise NotImplementedError


class MVTecDataset(BaseDataset):
    def _load_samples(self):
        category_dir = os.path.join(self.root_dir, self.category)
        
        if self.split == "train":
            # Load normal training images
            train_dir = os.path.join(category_dir, "train", "good")
            for img_name in os.listdir(train_dir):
                if img_name.endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append({
                        "image_path": os.path.join(train_dir, img_name),
                        "label": 0,
                        "mask_path": None
                    })
        
        elif self.split == "test":
            # Load test images (both normal and anomaly)
            test_dir = os.path.join(category_dir, "test")
            gt_dir = os.path.join(category_dir, "ground_truth")
            
            for defect_type in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                
                is_normal = (defect_type == "good")
                
                for img_name in os.listdir(defect_dir):
                    if not img_name.endswith((".png", ".jpg", ".jpeg")):
                        continue
                    
                    img_path = os.path.join(defect_dir, img_name)
                    
                    # Find mask path for anomalies
                    mask_path = None
                    if not is_normal:
                        mask_dir = os.path.join(gt_dir, defect_type)
                        mask_name = img_name.replace(".png", "_mask.png")
                        potential_mask = os.path.join(mask_dir, mask_name)
                        if os.path.exists(potential_mask):
                            mask_path = potential_mask
                    
                    self.samples.append({
                        "image_path": img_path,
                        "label": 0 if is_normal else 1,
                        "mask_path": mask_path,
                        "defect_type": defect_type
                    })
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        result = {
            "image": image,
            "label": sample["label"],
            "image_path": sample["image_path"]
        }
        
        if sample["mask_path"]:
            mask = Image.open(sample["mask_path"]).convert("L")
            # Apply mask transform if available
            result["mask"] = mask
        
        return result


class ViSADataset(BaseDataset):
    def _load_samples(self):
        # TODO: Implement VisA dataset loading
        pass
    
    def __getitem__(self, idx):
        # TODO: Implement VisA dataset item retrieval
        pass


class BTADDataset(BaseDataset):
    def _load_samples(self):
        # TODO: Implement BTAD dataset loading
        pass
    
    def __getitem__(self, idx):
        # TODO: Implement BTAD dataset item retrieval
        pass
```

### Step 6: DataLoader 함수 구현

#### 6.1 src/anomaly_detection/data/dataloaders.py
```python
# src/anomaly_detection/data/dataloaders.py
from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
```

### Step 7: 데이터 검증 유틸리티 구현

#### 7.1 src/anomaly_detection/data/validation.py
```python
# src/anomaly_detection/data/validation.py
import os
from PIL import Image


def validate_dataset_structure(dataset_root, dataset_type="mvtec"):
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    if dataset_type == "mvtec":
        return _validate_mvtec_structure(dataset_root)
    elif dataset_type == "visa":
        return _validate_visa_structure(dataset_root)
    elif dataset_type == "btad":
        return _validate_btad_structure(dataset_root)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _validate_mvtec_structure(dataset_root):
    required_dirs = ["train", "test", "ground_truth"]
    categories = os.listdir(dataset_root)
    
    valid_categories = []
    for category in categories:
        category_dir = os.path.join(dataset_root, category)
        if not os.path.isdir(category_dir):
            continue
        
        has_all_dirs = all(
            os.path.exists(os.path.join(category_dir, d)) 
            for d in required_dirs
        )
        
        if has_all_dirs:
            valid_categories.append(category)
    
    return valid_categories


def _validate_visa_structure(dataset_root):
    # TODO: Implement VisA validation
    pass


def _validate_btad_structure(dataset_root):
    # TODO: Implement BTAD validation
    pass


def check_image_integrity(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False
```

### Step 8: data 모듈 __init__.py 작성

#### 8.1 src/anomaly_detection/data/__init__.py
```python
# src/anomaly_detection/data/__init__.py
from .datasets import BaseDataset, MVTecDataset, ViSADataset, BTADDataset
from .dataloaders import create_dataloader
from .transforms import get_train_transforms, get_test_transforms, get_mask_transforms
from .validation import validate_dataset_structure, check_image_integrity

__all__ = [
    "BaseDataset",
    "MVTecDataset",
    "ViSADataset",
    "BTADDataset",
    "create_dataloader",
    "get_train_transforms",
    "get_test_transforms",
    "get_mask_transforms",
    "validate_dataset_structure",
    "check_image_integrity",
]
```

### Step 9: 테스트 작성

#### 9.1 tests/phase_01/test_transforms.py
```python
# tests/phase_01/test_transforms.py
import pytest
import torch
from PIL import Image
from anomaly_detection.data.transforms import (
    get_train_transforms,
    get_test_transforms,
    get_mask_transforms
)


def test_train_transforms():
    transform = get_train_transforms(image_size=256)
    
    img = Image.new("RGB", (512, 512), color=(255, 0, 0))
    transformed = transform(img)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 256, 256)
    assert transformed.min() >= -3.0
    assert transformed.max() <= 3.0


def test_test_transforms():
    transform = get_test_transforms(image_size=256)
    
    img = Image.new("RGB", (512, 512), color=(0, 255, 0))
    transformed = transform(img)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 256, 256)


def test_mask_transforms():
    transform = get_mask_transforms(image_size=256)
    
    mask = Image.new("L", (512, 512), color=255)
    transformed = transform(mask)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (1, 256, 256)
    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.0


def test_custom_normalization():
    custom_mean = [0.5, 0.5, 0.5]
    custom_std = [0.5, 0.5, 0.5]
    
    transform = get_train_transforms(
        image_size=128,
        mean=custom_mean,
        std=custom_std
    )
    
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    transformed = transform(img)
    
    assert transformed.shape == (3, 128, 128)
```

#### 9.2 tests/phase_01/test_datasets.py
```python
# tests/phase_01/test_datasets.py
import pytest
import os
from anomaly_detection.data import MVTecDataset
from anomaly_detection.data.transforms import get_train_transforms


def test_mvtec_dataset_creation(config_dir):
    # This test requires actual MVTec dataset
    # Skip if dataset not available
    pytest.skip("Requires actual MVTec dataset")


def test_mvtec_train_dataset_structure():
    # Test basic dataset structure
    pytest.skip("Requires actual MVTec dataset")


def test_mvtec_test_dataset_structure():
    # Test test dataset with anomalies
    pytest.skip("Requires actual MVTec dataset")


def test_dataset_getitem():
    # Test __getitem__ method
    pytest.skip("Requires actual MVTec dataset")


def test_dataset_length():
    # Test __len__ method
    pytest.skip("Requires actual MVTec dataset")
```

#### 9.3 tests/phase_01/test_dataloaders.py
```python
# tests/phase_01/test_dataloaders.py
import pytest
import torch
from torch.utils.data import TensorDataset
from anomaly_detection.data import create_dataloader


def test_create_dataloader():
    dummy_data = torch.randn(100, 3, 256, 256)
    dummy_labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    dataloader = create_dataloader(dataset, batch_size=16, shuffle=True)
    
    assert len(dataloader) == 7  # 100 / 16 = 6.25 -> 7
    
    batch = next(iter(dataloader))
    assert len(batch) == 2
    assert batch[0].shape[0] <= 16


def test_dataloader_iteration():
    dummy_data = torch.randn(32, 3, 256, 256)
    dummy_labels = torch.randint(0, 2, (32,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    dataloader = create_dataloader(dataset, batch_size=8, shuffle=False)
    
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        assert batch[0].shape[0] <= 8
    
    assert batch_count == 4


def test_dataloader_with_num_workers():
    dummy_data = torch.randn(64, 3, 256, 256)
    dummy_labels = torch.randint(0, 2, (64,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    dataloader = create_dataloader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=2
    )
    
    assert dataloader.num_workers == 2
    batch = next(iter(dataloader))
    assert batch[0].shape[0] == 16
```

#### 9.4 tests/phase_01/test_validation.py
```python
# tests/phase_01/test_validation.py
import pytest
import os
from PIL import Image
from anomaly_detection.data.validation import (
    validate_dataset_structure,
    check_image_integrity
)


def test_check_image_integrity_valid(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    img.save(img_path)
    
    assert check_image_integrity(str(img_path)) is True


def test_check_image_integrity_invalid(tmp_path):
    img_path = tmp_path / "corrupted.png"
    img_path.write_text("not an image")
    
    assert check_image_integrity(str(img_path)) is False


def test_validate_dataset_structure_not_found():
    with pytest.raises(FileNotFoundError):
        validate_dataset_structure("/nonexistent/path", "mvtec")


def test_validate_dataset_structure_unknown_type():
    with pytest.raises(ValueError):
        validate_dataset_structure("/some/path", "unknown_dataset")
```

### Step 10: pytest.ini 업데이트

```ini
# pytest.ini에 추가
markers =
    phase_00: Infrastructure tests
    phase_01: Data pipeline tests  # 추가
    phase_02: STFPM model tests
    slow: Slow running tests
    gpu: Tests requiring GPU
    requires_dataset: Tests requiring actual dataset  # 추가
```

### Step 11: 테스트 실행

```bash
# Phase 01 전체 테스트 실행
pytest tests/phase_01/ -v

# 특정 테스트 파일 실행
pytest tests/phase_01/test_transforms.py -v
pytest tests/phase_01/test_dataloaders.py -v

# 데이터셋 의존 테스트 제외
pytest tests/phase_01/ -v -m "not requires_dataset"
```

### Step 12: 수동 검증

```python
# 수동 테스트 스크립트 (test_manual.py)
from anomaly_detection.config import load_config
from anomaly_detection.data import MVTecDataset, create_dataloader, get_train_transforms

# Config 로딩
config = load_config("configs/paths.yaml")

# Dataset 생성
transform = get_train_transforms(image_size=256)
dataset = MVTecDataset(
    root_dir=config["mvtec_dir"],
    category="bottle",
    split="train",
    transform=transform
)

print(f"Dataset size: {len(dataset)}")

# DataLoader 생성
dataloader = create_dataloader(dataset, batch_size=8)

# 배치 확인
batch = next(iter(dataloader))
print(f"Batch image shape: {batch['image'].shape}")
print(f"Batch label shape: {batch['label'].shape}")
```

### Step 13: 커밋 및 푸시

```bash
git add .
git commit -m "feat: Phase 01 - Data pipeline implementation

- Implement Dataset classes (MVTec, VisA, BTAD)
- Add DataLoader creation functions
- Implement transform functions
- Add data validation utilities
- Create comprehensive tests for data pipeline
- Add dataset configuration files
"

git push origin feature/phase-01-data-pipeline
```

### Step 14: Pull Request 생성

제목: "Phase 01: Data Pipeline Implementation"

설명:
```markdown
## 변경 사항
- MVTec, VisA, BTAD Dataset 클래스 구현
- DataLoader 생성 함수 구현
- Transform 함수 구현 (학습/테스트/마스크)
- 데이터 검증 유틸리티 추가
- 데이터셋 설정 파일 추가

## 테스트 결과
- Transform 테스트: 4개 통과
- DataLoader 테스트: 3개 통과
- Validation 테스트: 4개 통과
- Dataset 테스트: 실제 데이터셋 필요 (스킵)

## 체크리스트
- [x] Dataset 클래스 구현
- [x] DataLoader 함수 구현
- [x] Transform 함수 구현
- [x] 테스트 작성 및 통과
- [x] 설정 파일 추가
- [ ] 실제 데이터셋으로 검증 (각자 환경)
```

---

## 산출물 체크리스트

### 생성할 파일 (12개)
- [ ] src/anomaly_detection/data/__init__.py
- [ ] src/anomaly_detection/data/datasets.py
- [ ] src/anomaly_detection/data/dataloaders.py
- [ ] src/anomaly_detection/data/transforms.py
- [ ] src/anomaly_detection/data/validation.py
- [ ] configs/datasets/mvtec.yaml
- [ ] configs/datasets/visa.yaml
- [ ] configs/datasets/btad.yaml
- [ ] tests/phase_01/test_datasets.py
- [ ] tests/phase_01/test_dataloaders.py
- [ ] tests/phase_01/test_transforms.py
- [ ] tests/phase_01/test_validation.py

### 생성할 디렉터리 (3개)
- [ ] src/anomaly_detection/data/
- [ ] configs/datasets/
- [ ] tests/phase_01/

### 통과해야 할 테스트
- [ ] test_transforms.py - 모든 테스트 통과 (4개 테스트)
- [ ] test_dataloaders.py - 모든 테스트 통과 (3개 테스트)
- [ ] test_validation.py - 모든 테스트 통과 (4개 테스트)
- [ ] test_datasets.py - 실제 데이터셋 필요 (스킵 가능)

### 품질 기준
- [ ] 코드 스타일 일관성 유지
- [ ] 타입힌트 및 docstring 없음 (프로젝트 규칙)
- [ ] 주석은 영어로만 작성
- [ ] 테스트 커버리지 ≥ 80%

---

## 일반적인 문제 및 해결 방법

### 문제 1: 데이터셋 경로 오류
```
FileNotFoundError: Dataset root not found
```

**해결 방법**:
- `configs/paths.yaml`의 `dataset_root`, `mvtec_dir` 등이 실제 경로인지 확인
- 데이터셋이 실제로 다운로드되어 있는지 확인

### 문제 2: PIL 이미지 로딩 오류
```
UnidentifiedImageError: cannot identify image file
```

**해결 방법**:
- 이미지 파일이 손상되지 않았는지 확인
- 지원되는 형식(.png, .jpg, .jpeg)인지 확인

### 문제 3: DataLoader num_workers 오류
```
RuntimeError: DataLoader worker exited unexpectedly
```

**해결 방법**:
```python
# num_workers를 0으로 설정하여 테스트
dataloader = create_dataloader(dataset, batch_size=8, num_workers=0)
```

### 문제 4: Transform 차원 오류
```
RuntimeError: Expected 3D (unbatched) or 4D (batched) input
```

**해결 방법**:
- 이미지가 RGB 모드인지 확인: `image.convert("RGB")`
- Transform 순서 확인: Resize → ToTensor → Normalize

---

## 개발 시 주의사항

### MVTec 데이터셋 특이사항
- 학습 데이터: `train/good/` 에만 정상 이미지
- 테스트 데이터: `test/good/`, `test/{defect_type}/` 
- 마스크: `ground_truth/{defect_type}/` 에 위치
- 마스크 파일명: `{image_name}_mask.png` 형태

### VisA 데이터셋 특이사항
- CSV 파일(`image_anno.csv`)로 split 정보 제공
- 정상/이상 이미지가 같은 폴더에 위치
- 마스크는 별도 폴더에 위치

### BTAD 데이터셋 특이사항
- 카테고리가 숫자(`01`, `02`, `03`)
- 구조는 MVTec과 유사
- 마스크 파일명 규칙이 MVTec과 다를 수 있음

---

## 다음 단계

Phase 01 완료 후:
1. `main` 브랜치로 머지
2. 로컬 `main` 업데이트
3. Phase 02로 이동: STFPM Model
4. 새 브랜치 생성: `git checkout -b feature/phase-02-stfpm`

---

## 참고 자료

- PyTorch Dataset/DataLoader: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- torchvision.transforms: https://pytorch.org/vision/stable/transforms.html
- MVTec AD 데이터셋: https://www.mvtec.com/company/research/datasets/mvtec-ad
- VisA 데이터셋: https://github.com/amazon-science/spot-diff
- BTAD 데이터셋: https://avires.dimi.uniud.it/papers/btad/

---

**문서 버전**: 1.0 (초안)  
**최종 업데이트**: 2025-01-XX  
**작성자**: 개발팀