# Phase 02: STFPM Model - 개발 가이드

## 개요

**목표**: STFPM 모델 통합 및 순수 PyTorch 학습 파이프라인 구축  
**예상 소요 시간**: 3-5일  
**브랜치**: `feature/phase-02-stfpm`  
**담당**: STFPM 모델 담당 개발자  
**선행 조건**: Phase 00, Phase 01 완료 및 main 브랜치 머지 완료

---

## 목표

- Anomalib STFPM 모델 복사 및 통합
- Lightning 의존성 제거 (순수 PyTorch 구현)
- BaseTrainer 클래스 설계 및 구현 (Hook 패턴)
- STFPMTrainer 구현 (BaseTrainer 상속)
- 학습/검증/평가 파이프라인 구축
- MVTec 데이터셋으로 성능 검증

---

## Anomalib STFPM 구조 분석

### Anomalib STFPM 디렉터리 구조
```
anomalib/models/stfpm/
├── __init__.py
├── torch_model.py           # 순수 PyTorch 모델 (복사 대상)
├── lightning_model.py       # Lightning 래퍼 (재구현 대상)
├── loss.py                  # Loss 함수 (stfpm.py에 통합)
└── anomaly_map.py           # Anomaly Map 생성 (stfpm.py에 통합)
```

### 복사 및 통합 전략
1. **torch_model.py** → `src/anomaly_detection/models/stfpm/stfpm.py` (그대로 복사)
2. **loss.py** → `stfpm.py`에 클래스/함수 통합
3. **anomaly_map.py** → `stfpm.py`에 클래스/함수 통합
4. **lightning_model.py** → `STFPMTrainer`로 재구현 (순수 PyTorch)

---

## 사전 요구사항

### Phase 01 완료 확인
```bash
git checkout main
git pull origin main
git log --oneline -10  # Phase 01 커밋 확인
```

### Anomalib 설치 (참조용)
```bash
# 별도 환경에서 Anomalib 설치 (복사 작업용)
pip install anomalib

# 또는 GitHub에서 직접 확인
# https://github.com/openvinotoolkit/anomalib
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
│   ├── models/                      # 새로 추가
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseModel
│   │   └── stfpm/
│   │       ├── __init__.py
│   │       └── stfpm.py             # STFPM 모델 통합
│   ├── trainer.py                   # BaseTrainer
│   ├── evaluator.py                 # Evaluator
│   └── metrics/                     # 새로 추가
│       ├── __init__.py
│       ├── auroc.py
│       └── visualization.py
├── configs/
│   ├── paths.yaml
│   ├── default.yaml
│   ├── datasets/
│   └── models/                      # 새로 추가
│       └── stfpm.yaml
├── tests/
│   ├── conftest.py
│   ├── phase_00/
│   ├── phase_01/
│   └── phase_02/                    # 새로 추가
│       ├── test_stfpm_model.py
│       ├── test_base_trainer.py
│       ├── test_stfpm_trainer.py
│       └── test_evaluation.py
├── experiments/
│   └── train_stfpm.py               # 학습 스크립트
├── notebooks/
│   └── eval_stfpm.ipynb             # 평가 노트북
└── docs/
    ├── phase_00_guide.md
    ├── phase_01_guide.md
    └── phase_02_guide.md            # 이 문서
```

---

## 개발 워크플로우

### Step 1: 브랜치 생성

```bash
git checkout main
git pull origin main
git checkout -b feature/phase-02-stfpm
```

### Step 2: 디렉터리 및 파일 생성

```bash
# 디렉터리 생성
mkdir -p src/anomaly_detection/models/stfpm
mkdir -p src/anomaly_detection/metrics
mkdir -p configs/models
mkdir -p tests/phase_02
mkdir -p notebooks

# __init__.py 파일 생성
touch src/anomaly_detection/models/__init__.py
touch src/anomaly_detection/models/stfpm/__init__.py
touch src/anomaly_detection/metrics/__init__.py
touch tests/phase_02/__init__.py
```

### Step 3: STFPM 모델 설정 파일 작성

#### 3.1 configs/models/stfpm.yaml
```yaml
model:
  name: stfpm
  backbone: resnet18
  layers:
    - layer1
    - layer2
    - layer3

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.4
  weight_decay: 0.0001
  optimizer: sgd
  
evaluation:
  threshold_method: adaptive
  image_metrics: [auroc, f1]
  pixel_metrics: [auroc, f1]
```

### Step 4: Anomalib STFPM 모델 복사 및 통합

#### 4.1 Anomalib에서 STFPM 파일 복사

```bash
# 1. Anomalib 저장소 확인 (별도 환경 또는 GitHub)
# https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/image/stfpm

# 2. 다음 파일들을 참조하여 stfpm.py 생성
# - torch_model.py (STFPMModel)
# - loss.py (STFPMLoss)
# - anomaly_map.py (AnomalyMapGenerator)
```

#### 4.2 src/anomaly_detection/models/stfpm/stfpm.py 생성

**중요**: Anomalib의 다음 3개 파일 내용을 **하나의 파일**로 통합:

1. `anomaly_map.py` → `AnomalyMapGenerator` 클래스 복사
2. `loss.py` → `STFPMLoss` 클래스 복사  
3. `torch_model.py` → `STFPMModel` 클래스 복사

**파일 구조**:
```python
# src/anomaly_detection/models/stfpm/stfpm.py

# 첫 줄에 파일 경로 주석 추가
# src/anomaly_detection/models/stfpm/stfpm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Anomalib에서 필요한 컴포넌트 import
# (TimmFeatureExtractor, Tiler 등은 별도 구현 또는 복사 필요)

# ============================================================================
# Copied from anomalib/src/anomalib/models/image/stfpm/anomaly_map.py
# ============================================================================
class AnomalyMapGenerator(nn.Module):
    # 원본 코드 그대로 복사 (타입힌트, docstring 유지)
    ...

# ============================================================================
# Copied from anomalib/src/anomalib/models/image/stfpm/loss.py
# ============================================================================
class STFPMLoss(nn.Module):
    # 원본 코드 그대로 복사 (타입힌트, docstring 유지)
    ...

# ============================================================================
# Copied from anomalib/src/anomalib/models/image/stfpm/torch_model.py
# ============================================================================
class STFPMModel(nn.Module):
    # 원본 코드 그대로 복사 (타입힌트, docstring 유지)
    ...
```

**주의사항**:
- Anomalib 코드의 **타입힌트와 docstring은 그대로 유지** (프로젝트 규칙 예외)
- `TimmFeatureExtractor`, `Tiler` 등 의존성 컴포넌트도 함께 복사 필요
- 첫 줄에 `# src/anomaly_detection/models/stfpm/stfpm.py` 주석 추가

### Step 5: BaseTrainer 설계 및 구현

#### 5.1 src/anomaly_detection/trainer.py

**참고**: 첨부된 문서의 `AnomalyTrainer`를 참조하여 BaseTrainer 설계

```python
# src/anomaly_detection/trainer.py
import os
import torch
from tqdm import tqdm


class BaseTrainer:
    """Base trainer class with hook pattern for customization.
    
    Hook methods:
        - on_train_batch_start(batch, batch_idx)
        - on_train_batch_end(batch, batch_idx, outputs, loss)
        - on_train_epoch_start(epoch)
        - on_train_epoch_end(epoch, train_loss)
        - on_validation_batch_start(batch, batch_idx)
        - on_validation_batch_end(batch, batch_idx, outputs)
        - on_validation_epoch_start(epoch)
        - on_validation_epoch_end(epoch, val_metrics)
        - on_fit_start()
        - on_fit_end()
    """
    
    def __init__(self, model, loss_fn=None, device="cuda", checkpoint_dir="checkpoints"):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.to(device)
        self.current_epoch = 0
        self.best_metric = None
    
    # Hook methods (override in subclasses)
    def on_train_batch_start(self, batch, batch_idx):
        pass
    
    def on_train_batch_end(self, batch, batch_idx, outputs, loss):
        pass
    
    def on_train_epoch_start(self, epoch):
        pass
    
    def on_train_epoch_end(self, epoch, train_loss):
        pass
    
    def on_validation_batch_start(self, batch, batch_idx):
        pass
    
    def on_validation_batch_end(self, batch, batch_idx, outputs):
        pass
    
    def on_validation_epoch_start(self, epoch):
        pass
    
    def on_validation_epoch_end(self, epoch, val_metrics):
        pass
    
    def on_fit_start(self):
        pass
    
    def on_fit_end(self):
        pass
    
    # Core methods (must override in subclasses)
    def training_step(self, batch, batch_idx):
        """Single training step. Must return dict with 'loss' key."""
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        """Single validation step. Return outputs for metric computation."""
        raise NotImplementedError
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler. Must return dict with 'optimizer' and optional 'scheduler'."""
        raise NotImplementedError
    
    # Training loop
    def train_epoch(self, train_loader):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        
        self.on_train_epoch_start(self.current_epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            self.on_train_batch_start(batch, batch_idx)
            
            self.optimizer.zero_grad()
            
            # Call training_step
            outputs = self.training_step(batch, batch_idx)
            loss = outputs['loss']
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            self.on_train_batch_end(batch, batch_idx, outputs, loss)
            
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.on_train_epoch_end(self.current_epoch, avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """Run one validation epoch."""
        self.model.eval()
        all_outputs = []
        
        self.on_validation_epoch_start(self.current_epoch)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
            for batch_idx, batch in enumerate(pbar):
                self.on_validation_batch_start(batch, batch_idx)
                
                outputs = self.validation_step(batch, batch_idx)
                all_outputs.append(outputs)
                
                self.on_validation_batch_end(batch, batch_idx, outputs)
        
        metrics = self.compute_metrics(all_outputs)
        self.on_validation_epoch_end(self.current_epoch, metrics)
        
        return metrics
    
    def compute_metrics(self, outputs):
        """Compute metrics from validation outputs. Override in subclass."""
        return {}
    
    def fit(self, train_loader, val_loader, num_epochs):
        """Main training loop."""
        self.on_fit_start()
        
        # Configure optimizer and scheduler
        opt_config = self.configure_optimizers()
        self.optimizer = opt_config['optimizer']
        self.scheduler = opt_config.get('scheduler', None)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics)
        
        self.on_fit_end()
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }, checkpoint_path)
        
        # Save best model
        if self.best_metric is None or self._is_better(metrics):
            self.best_metric = metrics
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics
            }, best_path)
    
    def _is_better(self, current_metrics):
        """Check if current metrics are better. Override in subclass."""
        return True
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("metrics", {})
```

### Step 6: STFPMTrainer 구현

#### 6.1 src/anomaly_detection/models/stfpm/trainer.py 생성

**참고**: 첨부된 문서의 마지막 부분 `STFPMTrainer`를 이 파일로 분리

```python
# src/anomaly_detection/models/stfpm/trainer.py
import torch
from anomaly_detection.trainer import BaseTrainer
from sklearn.metrics import roc_auc_score


class STFPMTrainer(BaseTrainer):
    """STFPM-specific trainer implementation.
    
    Implements:
        - training_step: STFPM loss computation
        - validation_step: anomaly map and score computation
        - configure_optimizers: SGD with StepLR
        - compute_metrics: AUROC calculation
    """
    
    def __init__(
        self,
        model,
        loss_fn=None,
        device="cuda",
        checkpoint_dir="checkpoints",
        learning_rate=0.4
    ):
        super().__init__(model, loss_fn, device, checkpoint_dir)
        
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_scores = []
    
    def training_step(self, batch, batch_idx):
        """STFPM training step."""
        images = batch["image"].to(self.device)
        
        # Forward pass (training mode returns features)
        teacher_features, student_features = self.model(images)
        
        # Compute loss
        loss = self.loss_fn(teacher_features, student_features)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """STFPM validation step."""
        images = batch["image"].to(self.device)
        labels = batch["label"]
        
        # Forward pass (eval mode returns predictions)
        outputs = self.model(images)
        
        return {
            "pred_score": outputs["pred_score"].cpu(),
            "anomaly_map": outputs["anomaly_map"].cpu(),
            "label": labels
        }
    
    def configure_optimizers(self):
        """Configure SGD optimizer with StepLR scheduler."""
        optimizer = torch.optim.SGD(
            self.model.student_model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.5
        )
        
        return {"optimizer": optimizer, "scheduler": scheduler}
    
    def compute_metrics(self, outputs):
        """Compute AUROC from validation outputs."""
        all_scores = torch.cat([out["pred_score"] for out in outputs])
        all_labels = torch.cat([out["label"] for out in outputs])
        
        image_auroc = roc_auc_score(all_labels.numpy(), all_scores.numpy())
        
        return {"image_auroc": image_auroc}
    
    def _is_better(self, current_metrics):
        """Check if current AUROC is better than best."""
        if self.best_metric is None:
            return True
        return current_metrics["image_auroc"] > self.best_metric.get("image_auroc", 0)
    
    # Hook implementations
    def on_train_epoch_end(self, epoch, train_loss):
        """Log training loss."""
        self.train_losses.append(train_loss)
        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
    
    def on_validation_epoch_end(self, epoch, val_metrics):
        """Log validation metrics."""
        self.val_scores.append(val_metrics)
        auroc = val_metrics.get("image_auroc", 0)
        print(f"Epoch {epoch} - Val AUROC: {auroc:.4f}")
```

#### 6.2 src/anomaly_detection/models/stfpm/__init__.py

```python
# src/anomaly_detection/models/stfpm/__init__.py
from .stfpm import STFPMModel, STFPMLoss, AnomalyMapGenerator
from .trainer import STFPMTrainer

__all__ = [
    "STFPMModel",
    "STFPMLoss", 
    "AnomalyMapGenerator",
    "STFPMTrainer",
]
```

### Step 7: Evaluator 구현

#### 7.1 src/anomaly_detection/evaluator.py
```python
# src/anomaly_detection/evaluator.py
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np


def evaluate(model, dataloader, device="cuda"):
    """Evaluate model on test dataset.
    
    Returns:
        metrics: dict with AUROC and F1 scores
        scores: numpy array of anomaly scores
        labels: numpy array of labels
        anomaly_maps: torch tensor of anomaly maps
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    all_maps = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"]
            
            # Forward pass (eval mode returns predictions)
            outputs = model(images)
            
            # Extract predictions
            pred_score = outputs["pred_score"]
            anomaly_map = outputs["anomaly_map"]
            
            all_scores.extend(pred_score.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_maps.append(anomaly_map.cpu())
    
    # Compute metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    image_auroc = roc_auc_score(all_labels, all_scores)
    
    # Compute F1 at optimal threshold
    threshold = np.percentile(all_scores, 95)
    predictions = (all_scores >= threshold).astype(int)
    f1 = f1_score(all_labels, predictions)
    
    metrics = {
        "image_auroc": image_auroc,
        "image_f1": f1,
        "threshold": threshold
    }
    
    return metrics, all_scores, all_labels, torch.cat(all_maps)
```

### Step 8: 학습 스크립트 작성

#### 8.1 experiments/train_stfpm.py
```python
# experiments/train_stfpm.py
import sys
sys.path.insert(0, "src")

import torch
from anomaly_detection.config import load_experiment_config
from anomaly_detection.data import MVTecDataset, create_dataloader, get_train_transforms, get_test_transforms
from anomaly_detection.models.stfpm import STFPMModel, STFPMLoss, STFPMTrainer


def main():
    # Load config
    config = load_experiment_config("stfpm", "mvtec")
    
    category = "bottle"
    
    # Create datasets
    train_transform = get_train_transforms(image_size=config.get("image_size", 256))
    test_transform = get_test_transforms(image_size=config.get("image_size", 256))
    
    train_dataset = MVTecDataset(
        root_dir=config["mvtec_dir"],
        category=category,
        split="train",
        transform=train_transform
    )
    
    test_dataset = MVTecDataset(
        root_dir=config["mvtec_dir"],
        category=category,
        split="test",
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    
    val_loader = create_dataloader(
        test_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )
    
    # Create model
    model = STFPMModel(
        backbone=config["model"]["backbone"],
        layers=config["model"]["layers"]
    )
    
    # Create loss
    loss_fn = STFPMLoss()
    
    # Create trainer
    device = config["training"]["device"]
    checkpoint_dir = f"{config['checkpoint_dir']}/stfpm/{category}"
    
    trainer = STFPMTrainer(
        model=model,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        learning_rate=config["training"]["learning_rate"]
    )
    
    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["training"]["num_epochs"]
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()
```

### Step 9: 테스트 작성

#### 9.1 tests/phase_02/test_stfpm_model.py
```python
# tests/phase_02/test_stfpm_model.py
import pytest
import torch
from anomaly_detection.models.stfpm.stfpm import STFPMModel, STFPMLoss, compute_anomaly_map


def test_stfpm_model_creation():
    model = STFPMModel(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    assert model is not None
    assert model.backbone == "resnet18"


def test_stfpm_forward():
    model = STFPMModel(backbone="resnet18")
    images = torch.randn(4, 3, 256, 256)
    
    outputs = model(images)
    
    assert "teacher_features" in outputs
    assert "student_features" in outputs
    assert len(outputs["teacher_features"]) == 3
    assert len(outputs["student_features"]) == 3


def test_stfpm_loss():
    criterion = STFPMLoss()
    
    teacher_features = {
        "layer1": torch.randn(4, 64, 64, 64),
        "layer2": torch.randn(4, 128, 32, 32)
    }
    student_features = {
        "layer1": torch.randn(4, 64, 64, 64),
        "layer2": torch.randn(4, 128, 32, 32)
    }
    
    loss = criterion(teacher_features, student_features)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_compute_anomaly_map():
    teacher_features = {
        "layer1": torch.randn(2, 64, 64, 64),
    }
    student_features = {
        "layer1": torch.randn(2, 64, 64, 64),
    }
    
    anomaly_map = compute_anomaly_map(teacher_features, student_features, (256, 256))
    
    assert anomaly_map.shape == (2, 1, 256, 256)
    assert anomaly_map.min() >= 0


def test_stfpm_teacher_frozen():
    model = STFPMModel(backbone="resnet18")
    
    for param in model.teacher_model.parameters():
        assert param.requires_grad is False


def test_stfpm_student_trainable():
    model = STFPMModel(backbone="resnet18")
    
    trainable_params = sum(p.requires_grad for p in model.student_model.parameters())
    assert trainable_params > 0
```

#### 9.2 tests/phase_02/test_base_trainer.py
```python
# tests/phase_02/test_base_trainer.py
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from anomaly_detection.trainer import BaseTrainer
from anomaly_detection.data import create_dataloader


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)


class SimpleTrainer(BaseTrainer):
    def train_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        
        return outputs, loss
    
    def validation_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        outputs = self.model(x)
        
        return {"outputs": outputs, "targets": y}


def test_base_trainer_creation():
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    trainer = SimpleTrainer(model, optimizer, criterion, device="cpu")
    
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.current_epoch == 0


def test_trainer_train_epoch():
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    trainer = SimpleTrainer(model, optimizer, criterion, device="cpu")
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    dataset = TensorDataset(x, y)
    loader = create_dataloader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    loss = trainer.train_epoch(loader)
    
    assert isinstance(loss, float)
    assert loss >= 0


def test_trainer_hooks_called():
    class HookTestTrainer(SimpleTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hook_calls = []
        
        def on_train_epoch_start(self, epoch):
            self.hook_calls.append("on_train_epoch_start")
        
        def on_train_epoch_end(self, epoch, train_loss):
            self.hook_calls.append("on_train_epoch_end")
    
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    trainer = HookTestTrainer(model, optimizer, criterion, device="cpu")
    
    x = torch.randn(16, 10)
    y = torch.randn(16, 1)
    dataset = TensorDataset(x, y)
    loader = create_dataloader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    trainer.train_epoch(loader)
    
    assert "on_train_epoch_start" in trainer.hook_calls
    assert "on_train_epoch_end" in trainer.hook_calls
```

### Step 10: 테스트 실행

```bash
# Phase 02 전체 테스트
pytest tests/phase_02/ -v

# 특정 테스트
pytest tests/phase_02/test_stfpm_model.py -v
pytest tests/phase_02/test_base_trainer.py -v

# GPU 필요한 테스트 제외
pytest tests/phase_02/ -v -m "not gpu"
```

### Step 11: 학습 실행

```bash
# STFPM 학습 실행
python experiments/train_stfpm.py

# 특정 카테고리 학습 (스크립트 수정 필요)
# python experiments/train_stfpm.py --category bottle
```

### Step 12: 평가 노트북 작성

#### 12.1 notebooks/eval_stfpm.ipynb
```python
# Cell 1: Setup
import sys
sys.path.insert(0, "../src")

import torch
import matplotlib.pyplot as plt
from anomaly_detection.config import load_experiment_config
from anomaly_detection.data import MVTecDataset, create_dataloader, get_test_transforms
from anomaly_detection.models.stfpm.stfpm import STFPMModel
from anomaly_detection.evaluator import evaluate

# Cell 2: Load config
config = load_experiment_config("stfpm", "mvtec")
category = "bottle"

# Cell 3: Load model
model = STFPMModel(
    backbone=config["model"]["backbone"],
    layers=config["model"]["layers"]
)

checkpoint_path = f"{config['checkpoint_dir']}/stfpm/{category}/best_model.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.to("cuda")
model.eval()

# Cell 4: Load test dataset
test_transform = get_test_transforms(image_size=256)
test_dataset = MVTecDataset(
    root_dir=config["mvtec_dir"],
    category=category,
    split="test",
    transform=test_transform
)

test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Cell 5: Evaluate
metrics, scores, labels, anomaly_maps = evaluate(model, test_loader, device="cuda")

print(f"Image AUROC: {metrics['image_auroc']:.4f}")
print(f"Image F1: {metrics['image_f1']:.4f}")

# Cell 6: Visualize results
# TODO: Add visualization code
```

### Step 13: 커밋 및 푸시

```bash
git add .
git commit -m "feat: Phase 02 - STFPM model implementation

- Copy and integrate STFPM model from Anomalib
- Implement BaseTrainer with hook pattern
- Implement STFPMTrainer (Lightning-free)
- Add training and evaluation pipeline
- Create comprehensive tests
- Add STFPM training script and evaluation notebook
"

git push origin feature/phase-02-stfpm
```

### Step 14: Pull Request 생성

제목: "Phase 02: STFPM Model Implementation"

설명:
```markdown
## 변경 사항
- Anomalib STFPM 모델 복사 및 통합
- Lightning 의존성 제거 (순수 PyTorch)
- BaseTrainer 구현 (Hook 패턴)
- STFPMTrainer 구현
- 학습/평가 파이프라인 구축

## 테스트 결과
- STFPM 모델 테스트: 6개 통과
- BaseTrainer 테스트: 3개 통과
- 학습 테스트: MVTec bottle 카테고리 성공

## 성능 (MVTec bottle)
- Image AUROC: 0.95+ (목표)
- Pixel AUROC: 0.97+ (목표)

## 체크리스트
- [x] STFPM 모델 통합
- [x] BaseTrainer 구현
- [x] STFPMTrainer 구현
- [x] 테스트 작성 및 통과
- [x] 학습 스크립트 작성
- [x] 평가 노트북 작성
- [ ] 전체 15개 카테고리 검증 (선택)
```

---

## 산출물 체크리스트

### 생성할 파일 (14개)
- [ ] src/anomaly_detection/models/__init__.py
- [ ] src/anomaly_detection/models/stfpm/__init__.py
- [ ] src/anomaly_detection/models/stfpm/stfpm.py (Anomalib 복사)
- [ ] src/anomaly_detection/models/stfpm/trainer.py
- [ ] src/anomaly_detection/trainer.py
- [ ] src/anomaly_detection/evaluator.py
- [ ] src/anomaly_detection/metrics/__init__.py
- [ ] configs/models/stfpm.yaml
- [ ] tests/phase_02/test_stfpm_model.py
- [ ] tests/phase_02/test_base_trainer.py
- [ ] tests/phase_02/test_stfpm_trainer.py
- [ ] tests/phase_02/test_evaluation.py
- [ ] experiments/train_stfpm.py
- [ ] notebooks/eval_stfpm.ipynb

### 생성할 디렉터리 (4개)
- [ ] src/anomaly_detection/models/stfpm/
- [ ] src/anomaly_detection/metrics/
- [ ] tests/phase_02/
- [ ] configs/models/

### Anomalib 복사 작업
- [ ] stfpm.py에 AnomalyMapGenerator 복사
- [ ] stfpm.py에 STFPMLoss 복사
- [ ] stfpm.py에 STFPMModel 복사
- [ ] 필요한 컴포넌트(TimmFeatureExtractor, Tiler) 복사

### 통과해야 할 테스트
- [ ] test_stfpm_model.py - 6개 테스트
- [ ] test_base_trainer.py - 3개 테스트
- [ ] test_stfpm_trainer.py - 5개 테스트 (TODO)
- [ ] test_evaluation.py - 3개 테스트 (TODO)

### 품질 기준
- [ ] Image AUROC ≥ 0.90 (최소 5개 카테고리)
- [ ] Pixel AUROC ≥ 0.95 (최소 5개 카테고리)
- [ ] 단일 배치 overfitting 성공
- [ ] 코드 스타일 일관성

---

## 일반적인 문제 및 해결 방법

### 문제 1: 사전학습 가중치 다운로드 실패
```
URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
```

**해결 방법**:
```python
# 외부 환경에서 사전 다운로드
import torch
from torchvision.models import resnet18

model = resnet18(pretrained=True)
torch.save(model.state_dict(), "backbones/resnet18.pth")

# 로컬에서 로드
model = resnet18(pretrained=False)
model.load_state_dict(torch.load("backbones/resnet18.pth"))
```

### 문제 2: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**해결 방법**:
- Batch size 줄이기: `batch_size: 16`
- Gradient checkpointing 사용
- Mixed precision training (AMP) 사용

### 문제 3: Loss가 감소하지 않음
```
Loss stays constant or increases
```

**해결 방법**:
- Learning rate 조정: `0.4 → 0.1`
- Teacher 모델이 제대로 freeze 되었는지 확인
- Optimizer 설정 확인 (SGD with momentum)

### 문제 4: Feature 차원 불일치
```
RuntimeError: The size of tensor a (64) must match the size of tensor b (128)
```

**해결 방법**:
- layers 설정 확인
- 모델과 trainer의 layers가 일치하는지 확인

---

## BaseTrainer Hook 패턴 설명

### Hook 메서드 종류

#### Batch-level Hooks
```python
def on_train_batch_start(self, batch, batch_idx):
    # 학습 배치 시작 전 실행
    # 예: 배치 데이터 전처리, 로깅
    pass

def on_train_batch_end(self, batch, batch_idx, outputs, loss):
    # 학습 배치 종료 후 실행
    # 예: 배치별 메트릭 수집, 그래디언트 클리핑
    pass
```

#### Epoch-level Hooks
```python
def on_train_epoch_start(self, epoch):
    # 학습 에폭 시작 전 실행
    # 예: Learning rate 스케줄링, 로깅 초기화
    pass

def on_train_epoch_end(self, epoch, train_loss):
    # 학습 에폭 종료 후 실행
    # 예: 에폭별 메트릭 로깅, 체크포인트 저장
    pass
```

#### Fit-level Hooks
```python
def on_fit_start(self):
    # 전체 학습 시작 전 실행
    # 예: 실험 디렉터리 생성, 설정 저장
    pass

def on_fit_end(self):
    # 전체 학습 종료 후 실행
    # 예: 최종 결과 저장, 정리 작업
    pass
```

### Hook 사용 예시

```python
class CustomTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []
    
    def on_train_batch_end(self, batch, batch_idx, outputs, loss):
        # 배치별 loss 저장
        self.losses.append(loss.item())
    
    def on_train_epoch_end(self, epoch, train_loss):
        # 에폭별 평균 loss 출력
        print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
        
        # Loss 그래프 저장
        import matplotlib.pyplot as plt
        plt.plot(self.losses)
        plt.savefig(f"loss_epoch_{epoch}.png")
        self.losses = []
```

---

## Phase 03 이후 개발 가이드

Phase 02의 구조를 다른 모델에 재사용:

### 1. 모델별 디렉터리 생성
```bash
mkdir -p src/anomaly_detection/models/efficientad
mkdir -p configs/models
mkdir -p tests/phase_03
```

### 2. Anomalib 모델 복사
```python
# src/anomaly_detection/models/efficientad/efficientad.py
# Copy from anomalib/models/efficientad/torch_model.py
# Integrate loss.py, anomaly_map.py
```

### 3. 모델별 Trainer 구현
```python
# src/anomaly_detection/models/efficientad/trainer.py
class EfficientADTrainer(BaseTrainer):
    def train_step(self, batch):
        # EfficientAD-specific training logic
        pass
    
    def validation_step(self, batch):
        # EfficientAD-specific validation logic
        pass
```

### 4. 설정 파일 작성
```yaml
# configs/models/efficientad.yaml
model:
  name: efficientad
  # EfficientAD-specific configs
```

### 5. 테스트 작성
```python
# tests/phase_03/test_efficientad_model.py
# tests/phase_03/test_efficientad_trainer.py
```

### 6. 학습 스크립트
```python
# experiments/train_efficientad.py
```

---

## 다음 단계

Phase 02 완료 후:
1. `main` 브랜치로 머지
2. 로컬 `main` 업데이트
3. Phase 03로 이동: EfficientAD Model
4. 새 브랜치 생성: `git checkout -b feature/phase-03-efficientad`

---

## 참고 자료

- STFPM 논문: https://arxiv.org/abs/2103.04257
- Anomalib GitHub: https://github.com/openvinotoolkit/anomalib
- PyTorch Lightning → PyTorch 마이그레이션 가이드
- Hook Pattern 설계: https://en.wikipedia.org/wiki/Hooking

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-01-XX  
**작성자**: 개발팀