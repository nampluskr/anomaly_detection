# Phase 02: Training Framework - 개발 가이드

## 개요

**목표**: 모델 학습 및 평가를 위한 Base Framework 구축  
**예상 소요 시간**: 2-3일  
**브랜치**: `feature/phase-02-training-framework`  
**담당**: Training Framework 담당 개발자  
**선행 조건**: Phase 00, Phase 01 완료 및 main 브랜치 머지 완료

---

## 목표

- BaseTrainer 클래스 설계 및 구현 (Hook 패턴)
- Evaluator 구현 (모델 평가 함수)
- Metrics 모듈 구현 (AUROC, F1 등)
- 더미 모델로 전체 파이프라인 테스트
- Phase 03 이후 모든 모델에서 재사용 가능한 기반 구축

---

## 사전 요구사항

### Phase 01 완료 확인
```bash
git checkout main
git pull origin main
git log --oneline -10  # Phase 01 커밋 확인
```

### 필요한 라이브러리 확인
```bash
# scikit-learn (metrics)
pip list | grep scikit-learn

# 없다면 설치
pip install scikit-learn
```

---

## 프로젝트 구조 (추가/수정)

```
anomaly_detection/
├── src/anomaly_detection/
│   ├── config.py
│   ├── utils.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── dataloaders.py
│   │   └── transforms.py
│   ├── trainer.py               # 새로 추가 (BaseTrainer)
│   ├── evaluator.py             # 새로 추가 (평가 함수)
│   └── metrics/                 # 새로 추가
│       ├── __init__.py
│       ├── auroc.py
│       └── visualization.py
├── configs/
│   ├── paths.yaml
│   ├── default.yaml
│   └── datasets/
├── tests/
│   ├── conftest.py
│   ├── phase_00/
│   ├── phase_01/
│   └── phase_02/                # 새로 추가
│       ├── test_base_trainer.py
│       ├── test_evaluator.py
│       └── test_metrics.py
└── docs/
    ├── phase_00_guide.md
    ├── phase_01_guide.md
    └── phase_02_guide.md        # 이 문서
```

---

## 개발 워크플로우

### Step 1: 브랜치 생성

```bash
git checkout main
git pull origin main
git checkout -b feature/phase-02-training-framework
```

### Step 2: 디렉터리 및 파일 생성

```bash
# 디렉터리 생성
mkdir -p src/anomaly_detection/metrics
mkdir -p tests/phase_02

# __init__.py 파일 생성
touch src/anomaly_detection/metrics/__init__.py
```

### Step 3: BaseTrainer 설계

#### 3.1 BaseTrainer 핵심 개념

**설계 원칙**:
- Hook 패턴으로 확장성 제공
- 순수 PyTorch (Lightning 의존성 제거)
- 모델별 커스터마이징 최소화

**Hook 종류**:
```
Batch-level:
- on_train_batch_start(batch, batch_idx)
- on_train_batch_end(batch, batch_idx, outputs, loss)
- on_validation_batch_start(batch, batch_idx)
- on_validation_batch_end(batch, batch_idx, outputs)

Epoch-level:
- on_train_epoch_start(epoch)
- on_train_epoch_end(epoch, train_loss)
- on_validation_epoch_start(epoch)
- on_validation_epoch_end(epoch, val_metrics)

Fit-level:
- on_fit_start()
- on_fit_end()
```

**추상 메서드** (서브클래스에서 반드시 구현):
```python
def training_step(self, batch, batch_idx):
    """학습 단계. dict with 'loss' key 반환"""
    raise NotImplementedError

def validation_step(self, batch, batch_idx):
    """검증 단계. 평가용 outputs 반환"""
    raise NotImplementedError

def configure_optimizers(self):
    """Optimizer 및 Scheduler 설정"""
    raise NotImplementedError
```

### Step 4: BaseTrainer 구현

#### 4.1 src/anomaly_detection/trainer.py

**구현 요구사항**:
- Hook 메서드 10개 (빈 구현, 서브클래스에서 오버라이드)
- 추상 메서드 3개 (training_step, validation_step, configure_optimizers)
- train_epoch, validate_epoch, fit 메서드
- checkpoint 저장/로드 기능
- best model 자동 저장

**주요 메서드**:
```python
class BaseTrainer:
    def __init__(self, model, loss_fn=None, device="cuda", checkpoint_dir="checkpoints")
    
    # Hook methods (override in subclass)
    def on_train_batch_start(self, batch, batch_idx)
    def on_train_batch_end(self, batch, batch_idx, outputs, loss)
    def on_train_epoch_start(self, epoch)
    def on_train_epoch_end(self, epoch, train_loss)
    def on_validation_batch_start(self, batch, batch_idx)
    def on_validation_batch_end(self, batch, batch_idx, outputs)
    def on_validation_epoch_start(self, epoch)
    def on_validation_epoch_end(self, epoch, val_metrics)
    def on_fit_start(self)
    def on_fit_end(self)
    
    # Abstract methods (must implement in subclass)
    def training_step(self, batch, batch_idx)
    def validation_step(self, batch, batch_idx)
    def configure_optimizers(self)
    
    # Core methods
    def train_epoch(self, train_loader)
    def validate_epoch(self, val_loader)
    def fit(self, train_loader, val_loader, num_epochs)
    def compute_metrics(self, outputs)
    
    # Utility methods
    def save_checkpoint(self, epoch, metrics)
    def load_checkpoint(self, checkpoint_path)
    def _is_better(self, current_metrics)
```

**구현 시 주의사항**:
- `fit()` 에서 `configure_optimizers()` 호출하여 optimizer/scheduler 설정
- `train_epoch()` 에서 progress bar (tqdm) 사용
- checkpoint는 `epoch_N.pth` 와 `best_model.pth` 형태로 저장
- `_is_better()` 는 서브클래스에서 오버라이드 (기본값: True)

### Step 5: Evaluator 구현

#### 5.1 src/anomaly_detection/evaluator.py

**목적**: 학습된 모델의 성능 평가

**함수 시그니처**:
```python
def evaluate(model, dataloader, device="cuda"):
    """
    Returns:
        metrics: dict (image_auroc, image_f1, threshold)
        scores: numpy array
        labels: numpy array
        predictions: dict (optional, model-specific)
    """
```

**구현 내용**:
- 모델을 eval 모드로 설정
- 전체 데이터셋에 대해 추론
- anomaly score 수집
- AUROC, F1 계산
- threshold 계산 (adaptive)

**주의사항**:
- 모델의 forward pass 출력 형식에 의존하지 않도록 범용적으로 설계
- torch.no_grad() 사용
- tqdm으로 progress 표시

### Step 6: Metrics 모듈 구현

#### 6.1 src/anomaly_detection/metrics/__init__.py
```python
# src/anomaly_detection/metrics/__init__.py
from .auroc import compute_auroc, compute_auroc_per_class
from .visualization import plot_roc_curve, plot_confusion_matrix

__all__ = [
    "compute_auroc",
    "compute_auroc_per_class",
    "plot_roc_curve",
    "plot_confusion_matrix",
]
```

#### 6.2 src/anomaly_detection/metrics/auroc.py

**함수**:
```python
def compute_auroc(scores, labels):
    """Compute image-level AUROC"""
    
def compute_auroc_per_class(scores, labels, class_names):
    """Compute AUROC for each class"""
```

**구현**:
- sklearn.metrics.roc_auc_score 사용
- 에러 처리 (단일 클래스만 있는 경우)

#### 6.3 src/anomaly_detection/metrics/visualization.py

**함수**:
```python
def plot_roc_curve(scores, labels, save_path=None):
    """Plot ROC curve"""
    
def plot_confusion_matrix(predictions, labels, save_path=None):
    """Plot confusion matrix"""
```

**구현**:
- matplotlib 사용
- 저장 기능 포함

### Step 7: 더미 모델로 테스트

#### 7.1 tests/phase_02/test_base_trainer.py

**목적**: BaseTrainer가 정상 동작하는지 검증

**테스트 시나리오**:
1. 더미 모델로 BaseTrainer 상속
2. training_step, validation_step 구현
3. train_epoch, validate_epoch 실행
4. checkpoint 저장/로드 확인
5. Hook 메서드 호출 확인

**더미 모델 예시**:
```python
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

class DummyTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        outputs = self.model(x)
        return {"outputs": outputs, "targets": y}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return {"optimizer": optimizer}
```

**테스트 케이스**:
```python
def test_base_trainer_creation()
def test_trainer_train_epoch()
def test_trainer_validate_epoch()
def test_trainer_fit()
def test_checkpoint_save_load()
def test_hooks_called()
def test_best_model_selection()
```

#### 7.2 tests/phase_02/test_evaluator.py

**테스트 케이스**:
```python
def test_evaluate_basic()
def test_evaluate_metrics_computation()
def test_evaluate_with_perfect_predictions()
def test_evaluate_with_random_predictions()
```

#### 7.3 tests/phase_02/test_metrics.py

**테스트 케이스**:
```python
def test_compute_auroc()
def test_compute_auroc_perfect_score()
def test_compute_auroc_random_score()
def test_compute_auroc_single_class_error()
def test_plot_roc_curve()
def test_plot_confusion_matrix()
```

### Step 8: pytest.ini 업데이트

```ini
# pytest.ini에 추가
markers =
    phase_00: Setup tests
    phase_01: Data pipeline tests
    phase_02: Training framework tests  # 추가
    phase_03: STFPM model tests
    slow: Slow running tests
    gpu: Tests requiring GPU
```

### Step 9: 테스트 실행

```bash
# Phase 02 전체 테스트
pytest tests/phase_02/ -v

# 특정 테스트 파일
pytest tests/phase_02/test_base_trainer.py -v
pytest tests/phase_02/test_evaluator.py -v
pytest tests/phase_02/test_metrics.py -v

# CPU 전용 테스트
pytest tests/phase_02/ -v -m "not gpu"
```

### Step 10: 통합 테스트 (Data Pipeline + Trainer)

#### 10.1 tests/phase_02/test_integration.py

**목적**: Phase 01 + Phase 02 통합 동작 확인

**테스트 시나리오**:
```python
def test_full_pipeline_with_dummy_model():
    # 1. Config 로딩
    config = load_config("configs/paths.yaml")
    
    # 2. Dataset 생성 (Phase 01)
    transform = get_train_transforms(image_size=256)
    # 더미 데이터로 대체 (실제 MVTec 없이 테스트)
    
    # 3. DataLoader 생성 (Phase 01)
    loader = create_dataloader(dataset, batch_size=8)
    
    # 4. 더미 모델 및 Trainer 생성 (Phase 02)
    model = DummyModel()
    trainer = DummyTrainer(model, loss_fn, device="cpu")
    
    # 5. 학습 실행
    trainer.fit(loader, loader, num_epochs=2)
    
    # 6. 평가 실행
    metrics = evaluate(model, loader, device="cpu")
    
    # 7. 검증
    assert "image_auroc" in metrics
```

### Step 11: 문서화

#### 11.1 docs/trainer_design.md 작성 (선택)

**내용**:
- BaseTrainer 설계 철학
- Hook 패턴 상세 설명
- 서브클래스 구현 가이드
- 예제 코드

#### 11.2 README.md 업데이트

**추가 섹션**:
```markdown
## Training Framework

본 프로젝트는 순수 PyTorch 기반의 학습 프레임워크를 제공합니다.

### BaseTrainer

- Hook 패턴 기반 확장 가능한 Trainer
- PyTorch Lightning 의존성 제거
- 모델별 커스터마이징 최소화

### 사용 예시

\`\`\`python
from anomaly_detection.trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx):
        # 학습 로직 구현
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        # 검증 로직 구현
        return outputs
    
    def configure_optimizers(self):
        # Optimizer 설정
        return {"optimizer": optimizer}
\`\`\`
```

### Step 12: 커밋 및 푸시

```bash
git add .
git commit -m "feat: Phase 02 - Training framework implementation

- Implement BaseTrainer with hook pattern
- Add Evaluator for model evaluation
- Implement metrics module (AUROC, visualization)
- Add comprehensive tests with dummy model
- Integrate with Phase 01 data pipeline
"

git push origin feature/phase-02-training-framework
```

### Step 13: Pull Request 생성

제목: "Phase 02: Training Framework Implementation"

설명:
```markdown
## 변경 사항
- BaseTrainer 구현 (Hook 패턴)
- Evaluator 구현
- Metrics 모듈 추가 (AUROC, visualization)
- 더미 모델로 전체 파이프라인 검증

## 테스트 결과
- BaseTrainer 테스트: 7개 통과
- Evaluator 테스트: 4개 통과
- Metrics 테스트: 6개 통과
- 통합 테스트: 1개 통과

## Hook 패턴
- 10개 Hook 메서드 제공
- Batch/Epoch/Fit 레벨 커스터마이징 가능
- Phase 03 이후 모든 모델에서 재사용

## 체크리스트
- [x] BaseTrainer 구현
- [x] Evaluator 구현
- [x] Metrics 모듈 구현
- [x] 더미 모델 테스트
- [x] 통합 테스트 통과
- [x] 문서 작성
```

---

## 산출물 체크리스트

### 생성할 파일 (8개)
- [ ] src/anomaly_detection/trainer.py
- [ ] src/anomaly_detection/evaluator.py
- [ ] src/anomaly_detection/metrics/__init__.py
- [ ] src/anomaly_detection/metrics/auroc.py
- [ ] src/anomaly_detection/metrics/visualization.py
- [ ] tests/phase_02/test_base_trainer.py
- [ ] tests/phase_02/test_evaluator.py
- [ ] tests/phase_02/test_metrics.py
- [ ] tests/phase_02/test_integration.py (선택)

### 생성할 디렉터리 (2개)
- [ ] src/anomaly_detection/metrics/
- [ ] tests/phase_02/

### 통과해야 할 테스트
- [ ] test_base_trainer.py - 7개 테스트
- [ ] test_evaluator.py - 4개 테스트
- [ ] test_metrics.py - 6개 테스트
- [ ] test_integration.py - 1개 테스트 (선택)

### 품질 기준
- [ ] 더미 모델로 전체 파이프라인 동작 확인
- [ ] Hook 메서드 모두 호출 확인
- [ ] Checkpoint 저장/로드 정상 동작
- [ ] 코드 커버리지 ≥ 85%

---

## 일반적인 문제 및 해결 방법

### 문제 1: Hook 메서드가 호출되지 않음
```
Hook method not called during training
```

**해결 방법**:
- train_epoch, validate_epoch 메서드에서 hook 호출 위치 확인
- 서브클래스에서 super() 호출 여부 확인

### 문제 2: configure_optimizers 반환 형식 오류
```
TypeError: 'Optimizer' object is not subscriptable
```

**해결 방법**:
```python
# 잘못된 예
def configure_optimizers(self):
    return optimizer  # X

# 올바른 예
def configure_optimizers(self):
    return {"optimizer": optimizer}  # O
```

### 문제 3: validation_step 출력 형식 불일치
```
KeyError: 'outputs'
```

**해결 방법**:
- validation_step은 자유 형식 dict 반환
- compute_metrics에서 필요한 key 확인

### 문제 4: Checkpoint 로드 실패
```
RuntimeError: Error(s) in loading state_dict
```

**해결 방법**:
```python
# strict=False 옵션 사용
checkpoint = torch.load(path)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
```

---

## BaseTrainer Hook 패턴 상세

### Hook 실행 순서

```
fit()
├── on_fit_start()
├── for epoch in range(num_epochs):
│   ├── train_epoch()
│   │   ├── on_train_epoch_start(epoch)
│   │   ├── for batch in train_loader:
│   │   │   ├── on_train_batch_start(batch, batch_idx)
│   │   │   ├── training_step(batch, batch_idx)
│   │   │   └── on_train_batch_end(batch, batch_idx, outputs, loss)
│   │   └── on_train_epoch_end(epoch, train_loss)
│   ├── validate_epoch()
│   │   ├── on_validation_epoch_start(epoch)
│   │   ├── for batch in val_loader:
│   │   │   ├── on_validation_batch_start(batch, batch_idx)
│   │   │   ├── validation_step(batch, batch_idx)
│   │   │   └── on_validation_batch_end(batch, batch_idx, outputs)
│   │   └── on_validation_epoch_end(epoch, val_metrics)
│   └── save_checkpoint(epoch, val_metrics)
└── on_fit_end()
```

### Hook 사용 예시

#### 예시 1: Learning Rate Logging
```python
class MyTrainer(BaseTrainer):
    def on_train_epoch_start(self, epoch):
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} - Learning Rate: {current_lr}")
```

#### 예시 2: Gradient Clipping
```python
class MyTrainer(BaseTrainer):
    def on_train_batch_end(self, batch, batch_idx, outputs, loss):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

#### 예시 3: Early Stopping
```python
class MyTrainer(BaseTrainer):
    def __init__(self, *args, patience=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_count = 0
        self.best_score = None
    
    def on_validation_epoch_end(self, epoch, val_metrics):
        score = val_metrics.get("image_auroc", 0)
        
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        
        if self.no_improve_count >= self.patience:
            print(f"Early stopping at epoch {epoch}")
            raise StopIteration
```

---

## Phase 03 연계 사항

Phase 02 완료 후 Phase 03에서:

### 1. BaseTrainer 상속
```python
from anomaly_detection.trainer import BaseTrainer

class STFPMTrainer(BaseTrainer):
    # STFPM-specific implementation
    pass
```

### 2. 필수 구현 메서드
```python
def training_step(self, batch, batch_idx):
    # STFPM loss 계산
    return {"loss": loss}

def validation_step(self, batch, batch_idx):
    # STFPM anomaly score 계산
    return outputs

def configure_optimizers(self):
    # STFPM optimizer (SGD + StepLR)
    return {"optimizer": optimizer, "scheduler": scheduler}
```

### 3. Hook 활용 (선택)
```python
def on_train_epoch_end(self, epoch, train_loss):
    print(f"STFPM - Epoch {epoch} Loss: {train_loss:.4f}")

def on_validation_epoch_end(self, epoch, val_metrics):
    auroc = val_metrics.get("image_auroc", 0)
    print(f"STFPM - Epoch {epoch} AUROC: {auroc:.4f}")
```

---

## 다음 단계

Phase 02 완료 후:
1. `main` 브랜치로 머지
2. 로컬 `main` 업데이트
3. Phase 03로 이동: STFPM Model
4. 새 브랜치 생성: `git checkout -b feature/phase-03-stfpm`

---

## 참고 자료

- PyTorch Training Loop: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- Hook Pattern: https://refactoring.guru/design-patterns/observer
- scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-01-XX  
**작성자**: 개발팀