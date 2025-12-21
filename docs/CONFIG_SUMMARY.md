# Configuration Files Summary

## 생성된 파일 목록 (7개)

### 1. configs/paths.yaml
**목적**: 프로젝트 경로 설정
**주요 내용**:
- project_root, source_dir, config_dir, output_dir
- dataset_root, mvtec_dir, visa_dir, btad_dir
- backbone_root
- checkpoint_dir, result_dir, log_dir

**중요**: 실제 경로로 수정 필요
```yaml
dataset_root: /path/to/datasets  # 수정 필요
backbone_root: /path/to/backbones  # 수정 필요
```

---

### 2. configs/default.yaml
**목적**: 공통 기본 설정
**주요 내용**:
- data: image_size, normalization
- **dataloader**: train/test 설정
- training: num_epochs, early_stopping, checkpoint
- evaluation: metrics, threshold
- device: cuda/cpu
- seed: 42

**DataLoader 설정**:
```yaml
dataloader:
  train:
    batch_size: 32
    shuffle: true
    drop_last: true
  
  test:
    batch_size: 32
    shuffle: false
    drop_last: false
  
  # 플랫폼 기본값 오버라이드 (선택)
  override:
    num_workers: 4
    pin_memory: true
```

---

### 3. configs/datasets/mvtec.yaml
**목적**: MVTec AD 데이터셋 설정
**주요 내용**:
- 15개 카테고리 목록
- default_category: bottle
- image_format: png
- mask 설정
- 데이터셋별 transform 설정
- DataLoader 오버라이드 (선택)

---

### 4. configs/datasets/visa.yaml
**목적**: ViSA 데이터셋 설정
**주요 내용**:
- 12개 카테고리 목록
- default_category: candle
- csv_file: split_csv/1cls.csv
- image_format: JPG
- mask 설정

---

### 5. configs/datasets/btad.yaml
**목적**: BTAD 데이터셋 설정
**주요 내용**:
- 3개 카테고리 (01, 02, 03)
- default_category: "01"
- image_format: mixed
- mask 설정

---

### 6. configs/models/stfpm.yaml
**목적**: STFPM 모델 설정
**주요 내용**:
- backbone: resnet18
- layers: [layer1, layer2, layer3]
- optimizer: SGD (lr=0.4)
- scheduler: StepLR
- loss weights
- evaluation settings

---

### 7. configs/README.md
**목적**: 설정 파일 사용 가이드
**내용**:
- 디렉터리 구조
- 각 파일 설명
- 사용법 예시
- DataLoader 설정 가이드
- 변수 치환 방법
- 우선순위 규칙
- 예제 및 트러블슈팅

---

## 폴더 구조

```
configs/
├── paths.yaml              # 경로 설정
├── default.yaml            # 공통 기본 설정
├── datasets/               # 데이터셋별 설정
│   ├── mvtec.yaml
│   ├── visa.yaml
│   └── btad.yaml
├── models/                 # 모델별 설정
│   ├── stfpm.yaml
│   └── efficientad.yaml (추가 예정)
└── README.md              # 사용 가이드
```

---

## DataLoader 설정 상세

### 플랫폼별 기본값 (자동 감지)

**Windows**:
```yaml
num_workers: 0
pin_memory: false
persistent_workers: false
```

**Linux**:
```yaml
num_workers: 8
pin_memory: true
persistent_workers: true
```

### Train vs Test 설정

**Train**:
```yaml
batch_size: 32
shuffle: true
drop_last: true
```

**Test**:
```yaml
batch_size: 32
shuffle: false
drop_last: false
```

### 설정 오버라이드 방법

#### 방법 1: default.yaml에서 오버라이드
```yaml
# configs/default.yaml
dataloader:
  override:
    num_workers: 4
    pin_memory: true
```

#### 방법 2: 모델/데이터셋별 오버라이드
```yaml
# configs/models/stfpm.yaml
dataloader:
  train:
    batch_size: 64  # 기본값 32에서 변경
```

#### 방법 3: 코드에서 오버라이드
```python
train_loader = get_train_dataloader(
    dataset,
    batch_size=config["dataloader"]["train"]["batch_size"],
    num_workers=4  # 명시적 오버라이드
)
```

---

## 사용 예시

### 1. 설정 로딩

```python
from anomaly_detection.config import load_experiment_config

# 자동으로 병합: paths + default + mvtec + stfpm
config = load_experiment_config("stfpm", "mvtec")

# 사용
batch_size = config["dataloader"]["train"]["batch_size"]
learning_rate = config["training"]["optimizer"]["lr"]
```

### 2. DataLoader 생성

```python
from anomaly_detection.data.dataloaders import get_train_dataloader, get_test_dataloader

# 플랫폼 자동 감지 + config 설정 사용
train_loader = get_train_dataloader(
    train_dataset,
    batch_size=config["dataloader"]["train"]["batch_size"]
)

test_loader = get_test_dataloader(
    test_dataset,
    batch_size=config["dataloader"]["test"]["batch_size"]
)
```

### 3. 설정 커스터마이징

```python
# GPU 메모리 부족 시
config["dataloader"]["train"]["batch_size"] = 16

# CPU 전용 학습
config["device"]["type"] = "cpu"
config["dataloader"]["override"]["num_workers"] = 4
config["dataloader"]["override"]["pin_memory"] = False
```

---

## 우선순위

설정 병합 시 우선순위 (낮음 → 높음):
```
paths.yaml
    ↓
default.yaml
    ↓
datasets/mvtec.yaml
    ↓
models/stfpm.yaml
```

**예시**:
- default.yaml: `batch_size: 32`
- stfpm.yaml: `batch_size: 64`
- **최종 결과**: `batch_size = 64`

---

## 변수 치환

### 지원 문법
```yaml
project_root: ${PWD}
output_dir: ${project_root}/outputs
checkpoint_dir: ${output_dir}/checkpoints
```

### 특수 변수
- `${PWD}`: 현재 작업 디렉터리
- `${HOME}`: 홈 디렉터리
- `${USER}`: 사용자명
- 기타 환경 변수

### 다단계 치환
```yaml
a: /base
b: ${a}/middle
c: ${b}/end
# 결과: c = /base/middle/end
```

---

## 파일 복사 위치

생성된 파일들을 다음 위치에 배치:

```bash
# 프로젝트 루트에서
mkdir -p configs/datasets configs/models

# 파일 복사
cp paths.yaml configs/
cp default.yaml configs/
cp mvtec.yaml configs/datasets/
cp visa.yaml configs/datasets/
cp btad.yaml configs/datasets/
cp stfpm.yaml configs/models/
cp configs_README.md configs/README.md
```

---

## 주요 설정값 요약

### Data
- image_size: 256
- normalization: ImageNet mean/std

### DataLoader
- train batch_size: 32
- test batch_size: 32
- num_workers: 플랫폼별 (0 or 8)
- pin_memory: 플랫폼별 (false or true)

### Training
- num_epochs: 100
- early_stopping patience: 10
- checkpoint interval: 10

### STFPM
- backbone: resnet18
- optimizer: SGD (lr=0.4)
- scheduler: StepLR (step=40, gamma=0.1)

---

## 다음 단계

1. **paths.yaml 수정**: 실제 데이터셋/백본 경로 설정
2. **config.py 구현**: load_config, load_experiment_config 함수
3. **테스트**: config 로딩 및 변수 치환 검증
4. **통합**: DataLoader와 config 연동

---

모든 설정 파일이 준비되었습니다!
