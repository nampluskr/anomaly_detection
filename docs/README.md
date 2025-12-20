# Anomaly Detection Framework

PyTorch 기반 Vision Anomaly Detection 모델 학습 및 평가 프레임워크

## 개요

본 프로젝트는 SOTA(State-of-the-Art) Vision Anomaly Detection 모델들을 통합하고 학습/평가할 수 있는 순수 PyTorch 기반 프레임워크입니다. Anomalib 라이브러리의 최신 모델들을 지원하며, PyTorch Lightning 의존성을 제거하여 로컬 환경 제약을 극복합니다.

### 주요 특징

- **순수 PyTorch 구현**: Lightning 의존성 제거, 로컬 환경 최적화
- **SOTA 모델 지원**: Anomalib 모델 직접 통합 (STFPM, EfficientAD 등)
- **표준 벤치마크**: MVTec AD, VisA, BTAD 데이터셋 지원
- **Hook 패턴 기반 Trainer**: 확장 가능한 학습 파이프라인
- **TDD 방식 개발**: pytest 기반 단계별 테스트

---

## 프로젝트 구조

```
anomaly_detection/
├── src/anomaly_detection/          # 핵심 소스 코드
│   ├── config.py                   # 설정 관리 (변수 치환, 경로 해석)
│   ├── utils.py                    # 유틸리티 함수
│   ├── trainer.py                  # BaseTrainer (Hook 패턴)
│   ├── evaluator.py                # 모델 평가
│   ├── data/                       # 데이터 파이프라인
│   │   ├── datasets.py             # Dataset 클래스 (MVTec, VisA, BTAD)
│   │   ├── dataloaders.py          # DataLoader 생성
│   │   └── transforms.py           # Transform 함수
│   ├── models/                     # 모델 모듈
│   │   ├── stfpm/                  # STFPM 모델
│   │   │   ├── stfpm.py            # 모델 + Loss + AnomalyMap
│   │   │   └── trainer.py          # STFPMTrainer
│   │   └── efficientad/            # EfficientAD 모델 (예정)
│   └── metrics/                    # 평가 메트릭
│       ├── auroc.py
│       └── visualization.py
├── configs/                        # 설정 파일
│   ├── paths.yaml                  # 경로 설정
│   ├── default.yaml                # 공통 설정
│   ├── datasets/                   # 데이터셋별 설정
│   │   ├── mvtec.yaml
│   │   ├── visa.yaml
│   │   └── btad.yaml
│   └── models/                     # 모델별 설정
│       ├── stfpm.yaml
│       └── efficientad.yaml
├── tests/                          # 테스트 코드
│   ├── conftest.py                 # pytest fixtures
│   ├── phase_00/                   # Setup 테스트
│   ├── phase_01/                   # Data Pipeline 테스트
│   ├── phase_02/                   # Training Framework 테스트
│   └── phase_03/                   # STFPM 모델 테스트
├── experiments/                    # 학습 스크립트
│   ├── train_stfpm.py
│   └── train_efficientad.py
├── notebooks/                      # 분석 및 평가 노트북
│   ├── eval_stfpm.ipynb
│   └── dataset_statistics.ipynb
├── outputs/                        # 학습 결과 (gitignore)
│   ├── checkpoints/
│   ├── results/
│   └── logs/
├── docs/                           # 문서
│   ├── phase_00_guide.md           # Setup 가이드
│   ├── phase_01_guide.md           # Data Pipeline 가이드
│   ├── phase_02_guide.md           # Training Framework 가이드
│   └── phase_03_guide.md           # STFPM 가이드
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md                       # 이 문서
```

---

## 시작하기

### 1. 환경 설정

#### 필수 요구사항
- Python 3.8+
- CUDA 11.0+ (GPU 사용 시)

#### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd anomaly_detection

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

#### requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.22.0
pandas>=1.4.0
pillow>=9.0.0
pyyaml>=6.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pytest>=7.0.0
```

### 2. 데이터셋 준비

#### 데이터셋 다운로드
- **MVTec AD**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **VisA**: https://github.com/amazon-science/spot-diff
- **BTAD**: https://avires.dimi.uniud.it/papers/btad/

#### 데이터셋 배치
```bash
# 외부 경로 (절대 경로)
/path/to/datasets/
├── mvtec/
│   ├── bottle/
│   ├── cable/
│   └── ...
├── visa/
│   ├── candle/
│   └── ...
└── btad/
    ├── 01/
    └── ...
```

#### 경로 설정
`configs/paths.yaml` 파일에서 실제 경로로 수정:

```yaml
dataset_root: /path/to/your/datasets  # 실제 경로로 수정
mvtec_dir: ${dataset_root}/mvtec
visa_dir: ${dataset_root}/visa
btad_dir: ${dataset_root}/btad
```

### 3. 사전학습 백본 준비

```bash
# 외부 환경에서 다운로드
python -c "
from torchvision.models import resnet18, resnet50
import torch

# ResNet18 다운로드
model = resnet18(pretrained=True)
torch.save(model.state_dict(), 'backbones/resnet18.pth')

# ResNet50 다운로드
model = resnet50(pretrained=True)
torch.save(model.state_dict(), 'backbones/resnet50.pth')
"
```

`configs/paths.yaml` 설정:
```yaml
backbone_root: /path/to/your/backbones
```

---

## 빠른 시작

### STFPM 모델 학습

```bash
# 1. STFPM 학습 (MVTec bottle)
python experiments/train_stfpm.py

# 2. 학습 모니터링
tensorboard --logdir outputs/logs

# 3. 평가
jupyter notebook notebooks/eval_stfpm.ipynb
```

### 커스텀 데이터셋 학습

```python
from anomaly_detection.config import load_experiment_config
from anomaly_detection.data import MVTecDataset, create_dataloader, get_train_transforms
from anomaly_detection.models.stfpm import STFPMModel, STFPMLoss, STFPMTrainer

# Config 로딩
config = load_experiment_config("stfpm", "mvtec")

# Dataset 생성
transform = get_train_transforms(image_size=256)
train_dataset = MVTecDataset(
    root_dir=config["mvtec_dir"],
    category="bottle",
    split="train",
    transform=transform
)

# DataLoader 생성
train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)

# 모델 및 Trainer 생성
model = STFPMModel(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
loss_fn = STFPMLoss()
trainer = STFPMTrainer(model, loss_fn, device="cuda", learning_rate=0.4)

# 학습
trainer.fit(train_loader, val_loader, num_epochs=100)
```

---

## 개발 가이드

### Phase별 개발 단계

본 프로젝트는 TDD(Test-Driven Development) 방식으로 단계적으로 개발됩니다.

#### Phase 00: Setup
- 설정 시스템 구축
- 프로젝트 구조 확립
- 테스트 프레임워크 구축

**가이드**: [docs/phase_00_guide.md](docs/phase_00_guide.md)

#### Phase 01: Data Pipeline
- Dataset 클래스 구현 (MVTec, VisA, BTAD)
- DataLoader 생성 함수
- Transform 함수

**가이드**: [docs/phase_01_guide.md](docs/phase_01_guide.md)

#### Phase 02: Training Framework
- BaseTrainer 구현 (Hook 패턴)
- Evaluator 구현
- Metrics 모듈

**가이드**: [docs/phase_02_guide.md](docs/phase_02_guide.md)

#### Phase 03: STFPM Model
- Anomalib STFPM 모델 통합
- STFPMTrainer 구현
- 학습/평가 파이프라인

**가이드**: [docs/phase_03_guide.md](docs/phase_03_guide.md)

#### Phase 04+: 추가 모델
- EfficientAD, PatchCore 등
- Phase 03 패턴 반복

---

## BaseTrainer 사용법

### Hook 패턴 기반 확장

```python
from anomaly_detection.trainer import BaseTrainer

class CustomTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx):
        """학습 단계 구현 (필수)"""
        images, labels = batch["image"], batch["label"]
        images = images.to(self.device)
        
        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """검증 단계 구현 (필수)"""
        images = batch["image"].to(self.device)
        outputs = self.model(images)
        
        return {"predictions": outputs, "labels": batch["label"]}
    
    def configure_optimizers(self):
        """Optimizer 설정 (필수)"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        return {"optimizer": optimizer, "scheduler": scheduler}
    
    # Hook 메서드 (선택적 구현)
    def on_train_epoch_end(self, epoch, train_loss):
        print(f"Epoch {epoch} - Loss: {train_loss:.4f}")
    
    def on_validation_epoch_end(self, epoch, val_metrics):
        print(f"Epoch {epoch} - AUROC: {val_metrics['auroc']:.4f}")
```

### Hook 메서드 목록

**Batch-level**:
- `on_train_batch_start(batch, batch_idx)`
- `on_train_batch_end(batch, batch_idx, outputs, loss)`
- `on_validation_batch_start(batch, batch_idx)`
- `on_validation_batch_end(batch, batch_idx, outputs)`

**Epoch-level**:
- `on_train_epoch_start(epoch)`
- `on_train_epoch_end(epoch, train_loss)`
- `on_validation_epoch_start(epoch)`
- `on_validation_epoch_end(epoch, val_metrics)`

**Fit-level**:
- `on_fit_start()`
- `on_fit_end()`

---

## 설정 시스템

### 변수 치환

```yaml
# configs/paths.yaml
project_root: ${PWD}
source_dir: ${project_root}/src
dataset_root: /path/to/datasets
mvtec_dir: ${dataset_root}/mvtec
```

### 설정 로딩

```python
from anomaly_detection.config import load_config, load_experiment_config

# 단일 설정 파일
config = load_config("configs/paths.yaml")

# 여러 설정 자동 병합
config = load_experiment_config(
    model_name="stfpm",
    dataset_name="mvtec"
)
# 자동으로 병합: paths.yaml + default.yaml + datasets/mvtec.yaml + models/stfpm.yaml
```

---

## 테스트

### 전체 테스트 실행

```bash
# 전체 테스트
pytest tests/ -v

# Phase별 테스트
pytest tests/phase_00/ -v  # Setup
pytest tests/phase_01/ -v  # Data Pipeline
pytest tests/phase_02/ -v  # Training Framework
pytest tests/phase_03/ -v  # STFPM Model

# 커버리지 확인
pytest tests/ --cov=src/anomaly_detection --cov-report=html
```

### 특정 테스트

```bash
# Config 테스트
pytest tests/phase_00/test_config.py -v

# Dataset 테스트
pytest tests/phase_01/test_datasets.py -v

# Trainer 테스트
pytest tests/phase_02/test_base_trainer.py -v
```

---

## 지원 모델

### 현재 지원
- **STFPM** (Student-Teacher Feature Pyramid Matching)
  - Paper: https://arxiv.org/abs/2103.04257
  - Backbone: ResNet18, ResNet50, Wide-ResNet50

### 개발 예정
- **EfficientAD**
- **PatchCore**
- **FastFlow**
- **Reverse Distillation**

---

## 성능 벤치마크

### MVTec AD Dataset

| Model | Backbone | Image AUROC | Pixel AUROC |
|-------|----------|-------------|-------------|
| STFPM | ResNet18 | 0.95+ | 0.97+ |
| STFPM | ResNet50 | 0.96+ | 0.98+ |

**테스트 환경**: NVIDIA RTX 3090, PyTorch 2.0

---

## 제약 사항 및 해결 방안

### 1. 로컬 환경 제약

**문제**: 방화벽으로 인한 외부 네트워크 접근 제한

**해결**:
- 사전학습 가중치 사전 다운로드 (`backbones/` 폴더)
- 보조 데이터셋 사전 배치
- 오프라인 설치 지원

### 2. Lightning 의존성

**문제**: PyTorch Lightning 설치 및 호환성 문제

**해결**:
- 순수 PyTorch 기반 BaseTrainer 구현
- Hook 패턴으로 확장성 제공
- Lightning 완전 제거

---

## Git 워크플로우

### 브랜치 전략

```bash
main                           # 안정 버전
├── feature/phase-00-setup     # Phase 00 개발
├── feature/phase-01-data-pipeline
├── feature/phase-02-training-framework
├── feature/phase-03-stfpm
└── feature/phase-04-efficientad
```

### 커밋 메시지 규칙

```
feat: Phase XX - 기능 추가
fix: Phase XX - 버그 수정
test: Phase XX - 테스트 추가
docs: Phase XX - 문서 업데이트
refactor: Phase XX - 리팩토링
```

**예시**:
```bash
git commit -m "feat: Phase 02 - Implement BaseTrainer with hook pattern"
git commit -m "test: Phase 03 - Add STFPM model tests"
git commit -m "docs: Update README with training examples"
```

---

## 코딩 규칙

### Python 스타일

- **타입힌트**: 사용하지 않음 (Anomalib 모델 파일 제외)
- **Docstring**: 사용하지 않음 (Anomalib 모델 파일 제외)
- **주석**: 영어로만 작성, 필요한 곳에만 최소화
- **파일 첫 줄**: `# src/anomaly_detection/파일명.py` 형식으로 경로 주석

### 네이밍 규칙

**파일명**:
- 모듈: `datasets.py`, `dataloaders.py` (소문자 + 복수형)
- 스크립트: `train_stfpm.py` (동사_모델명)

**클래스명**:
- Dataset: `MVTecDataset`, `ViSADataset`
- Model: `STFPMModel`, `EfficientADModel`
- Trainer: `BaseTrainer`, `STFPMTrainer`
- Loss: `STFPMLoss`

**함수명**:
- `create_dataloader()`, `get_train_transforms()`
- `load_config()`, `compute_auroc()`

---

## 문제 해결

### 자주 발생하는 문제

#### 1. Import Error
```
ModuleNotFoundError: No module named 'anomaly_detection'
```

**해결**:
```bash
# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 또는 개발 모드 설치
pip install -e .
```

#### 2. Config 경로 오류
```
FileNotFoundError: Config file not found
```

**해결**:
- 프로젝트 루트에서 실행 확인
- `configs/paths.yaml`에 절대 경로 설정

#### 3. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**해결**:
- Batch size 감소: `batch_size: 16`
- Mixed precision training 사용

#### 4. 사전학습 가중치 다운로드 실패
```
URLError: urlopen error
```

**해결**:
- 외부 환경에서 사전 다운로드
- `backbones/` 폴더에 수동 배치

---

## 기여 가이드

### 새 모델 추가

1. **브랜치 생성**
```bash
git checkout -b feature/phase-XX-modelname
```

2. **모델 디렉터리 생성**
```bash
mkdir -p src/anomaly_detection/models/modelname
mkdir -p configs/models
mkdir -p tests/phase_XX
```

3. **필수 파일 작성**
```
models/modelname/
├── __init__.py
├── modelname.py      # 모델 + Loss
└── trainer.py        # ModelTrainer(BaseTrainer)
```

4. **설정 파일**
```yaml
# configs/models/modelname.yaml
model:
  name: modelname
  backbone: resnet18
  
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
```

5. **테스트 작성**
```python
# tests/phase_XX/test_modelname.py
def test_model_creation()
def test_model_forward()
def test_model_training()
```

6. **학습 스크립트**
```python
# experiments/train_modelname.py
```

7. **평가 노트북**
```python
# notebooks/eval_modelname.ipynb
```

---

## 라이선스

MIT License

---

## 참고 자료

### 논문
- **STFPM**: Wang et al., "Student-Teacher Feature Pyramid Matching for Anomaly Detection" (BMVC 2021)
- **MVTec AD**: Bergmann et al., "MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" (CVPR 2019)

### 프로젝트
- **Anomalib**: https://github.com/openvinotoolkit/anomalib
- **PyTorch**: https://pytorch.org
- **scikit-learn**: https://scikit-learn.org

### 문서
- [Phase 00: Setup 가이드](docs/phase_00_guide.md)
- [Phase 01: Data Pipeline 가이드](docs/phase_01_guide.md)
- [Phase 02: Training Framework 가이드](docs/phase_02_guide.md)
- [Phase 03: STFPM 가이드](docs/phase_03_guide.md)

---

## 연락처

**프로젝트 관리자**: 개발팀  
**이슈 트래킹**: GitHub Issues  
**문서 업데이트**: 2025-01-XX

---

## 변경 이력

### v0.1.0 (2025-01-XX)
- Phase 00: Setup 완료
- Phase 01: Data Pipeline 완료
- Phase 02: Training Framework 완료
- Phase 03: STFPM 모델 통합 (진행 중)

---

**본 README는 프로젝트 진행에 따라 지속적으로 업데이트됩니다.**