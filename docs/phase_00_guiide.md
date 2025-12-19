# Phase 00: Infrastructure Setup - 개발 가이드

## 개요

**목표**: 프로젝트 인프라 구축 및 설정 시스템 확립  
**예상 소요 시간**: 1-2일  
**브랜치**: `feature/phase-00-infrastructure`  
**담당**: 전체 개발자 (공통 작업)

---

## 목표

- 설정 관리 시스템 구축
- 프로젝트 디렉터리 구조 확립
- pytest 기반 테스트 프레임워크 구축
- 필수 프로젝트 파일 생성 (.gitignore, pytest.ini, README.md)
- 모든 인프라 컴포넌트 검증

---

## 사전 요구사항

### 시스템 요구사항
- Python 3.8+
- Git
- 가상환경 (venv 또는 conda)

### 필수 라이브러리
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

---

## 프로젝트 구조

```
anomaly_detection/
├── src/anomaly_detection/
│   ├── __init__.py
│   ├── config.py
│   └── utils.py
├── configs/
│   ├── paths.yaml
│   └── default.yaml
├── tests/
│   ├── conftest.py
│   └── phase_00/
│       ├── test_config.py
│       └── test_structure.py
├── experiments/
├── notebooks/
├── outputs/
├── docs/
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 개발 워크플로우

### Step 1: 브랜치 생성

```bash
git checkout main
git pull origin main
git checkout -b feature/phase-00-infrastructure
```

### Step 2: 디렉터리 구조 생성

```bash
# 디렉터리 생성
mkdir -p src/anomaly_detection
mkdir -p configs/datasets
mkdir -p configs/models
mkdir -p tests/phase_00
mkdir -p experiments
mkdir -p notebooks
mkdir -p outputs
mkdir -p docs

# __init__.py 파일 생성 (빈 파일)
touch src/anomaly_detection/__init__.py
```

### Step 3: 핵심 파일 생성

#### 3.1 .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/

# Virtual Environment
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# Jupyter
.ipynb_checkpoints/

# Project specific
outputs/
*.pth
*.ckpt
*.log

# OS
.DS_Store
Thumbs.db
```

#### 3.2 pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings

markers =
    phase_00: Infrastructure tests
    phase_01: Data pipeline tests
    phase_02: STFPM model tests
    slow: Slow running tests
    gpu: Tests requiring GPU
```

#### 3.3 requirements.txt
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

#### 3.4 configs/paths.yaml
```yaml
# 경로 설정
project_root: ${PWD}

# 소스 코드
source_dir: ${project_root}/src
package_dir: ${source_dir}/anomaly_detection

# 외부 데이터 (절대 경로 - 각자 환경에 맞게 수정)
dataset_root: /path/to/your/datasets
mvtec_dir: ${dataset_root}/mvtec
visa_dir: ${dataset_root}/visa
btad_dir: ${dataset_root}/btad

# 사전학습 백본 가중치 (절대 경로 - 각자 환경에 맞게 수정)
backbone_root: /path/to/your/backbones

# 실험 출력
output_root: ${project_root}/outputs
checkpoint_dir: ${output_root}/checkpoints
result_dir: ${output_root}/results
log_dir: ${output_root}/logs

# 기타
experiment_dir: ${project_root}/experiments
notebook_dir: ${project_root}/notebooks
docs_dir: ${project_root}/docs
```

**중요**: `dataset_root`와 `backbone_root`를 실제 경로로 수정해야 합니다.

#### 3.5 configs/default.yaml
```yaml
# 공통 학습 설정
training:
  batch_size: 32
  num_workers: 4
  device: cuda
  seed: 42

# 공통 평가 설정
evaluation:
  batch_size: 32
  metrics: [auroc, f1]
  save_visualizations: true
```

### Step 4: 설정 시스템 구현

#### 4.1 src/anomaly_detection/config.py
실제 구현 파일 참조.

주요 기능:
- 변수 치환: `${variable_name}`
- 환경 변수 지원: `${ENV_VAR}`
- 중첩 딕셔너리 지원
- 자동 경로 해석
- 깊은 병합 지원

#### 4.2 src/anomaly_detection/utils.py
```python
import os
import logging
from datetime import datetime

def setup_logging(log_dir, name="train"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def get_experiment_dir(output_root, model_name, dataset_name, category):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{dataset_name}_{category}_{timestamp}"
    exp_dir = os.path.join(output_root, "experiments", model_name, exp_name)
    
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    return exp_dir
```

### Step 5: 테스트 프레임워크 생성

#### 5.1 tests/conftest.py
```python
import pytest
import os

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
```

#### 5.2 tests/phase_00/test_config.py
실제 구현 파일 참조.

테스트 항목:
- 설정 파일 로딩
- 변수 치환
- 경로 해석
- 중첩 딕셔너리
- 환경 변수
- 설정 병합

#### 5.3 tests/phase_00/test_structure.py
실제 구현 파일 참조.

테스트 항목:
- 필수 디렉터리 존재 확인
- 필수 파일 존재 확인
- 파일 내용 검증
- 경로 설정 검증
- 디렉터리 권한 확인

### Step 6: 의존성 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### Step 7: 테스트 실행

```bash
# Phase 00 전체 테스트 실행
pytest tests/phase_00/ -v

# 특정 테스트 파일 실행
pytest tests/phase_00/test_config.py -v
pytest tests/phase_00/test_structure.py -v

# 커버리지 확인
pytest tests/phase_00/ --cov=src/anomaly_detection --cov-report=html
```

### Step 8: 구현 검증

```bash
# Python import 테스트
python -c "from anomaly_detection.config import load_config; print('Import 성공')"

# Config 로딩 테스트
python -c "from anomaly_detection.config import load_config; cfg = load_config('configs/paths.yaml'); print('Config 로딩 완료:', cfg.keys())"

# 프로젝트 구조 확인
tree -L 2 anomaly_detection/
```

### Step 9: 커밋 및 푸시

```bash
# 상태 확인
git status

# 파일 추가
git add .

# 커밋
git commit -m "feat: Phase 00 - Infrastructure setup complete

- 변수 치환 기능을 포함한 config 시스템 구현
- 프로젝트 디렉터리 구조 생성
- pytest 테스트 프레임워크 추가
- .gitignore 및 pytest.ini 설정
- config 및 구조에 대한 포괄적인 테스트 추가
"

# 푸시
git push origin feature/phase-00-infrastructure
```

### Step 10: Pull Request 생성

1. GitHub 저장소로 이동
2. `feature/phase-00-infrastructure`에서 `main`으로 Pull Request 생성
3. 제목: "Phase 00: Infrastructure Setup"
4. 설명:
   ```
   ## 변경 사항
   - 설정 관리 시스템 구현
   - 프로젝트 디렉터리 구조 구축
   - pytest 테스트 프레임워크 생성
   - 필수 프로젝트 파일 추가
   
   ## 테스트 결과
   - 전체 37개 테스트 통과
   - 테스트 커버리지: 95%+
   
   ## 체크리스트
   - [x] 모든 디렉터리 생성 완료
   - [x] 모든 파일이 올바른 위치에 생성
   - [x] 모든 테스트 통과
   - [x] 문서 작성 완료
   ```

---

## 산출물 체크리스트

### 생성할 파일 (14개)
- [ ] .gitignore
- [ ] pytest.ini
- [ ] README.md
- [ ] requirements.txt
- [ ] configs/paths.yaml
- [ ] configs/default.yaml
- [ ] src/anomaly_detection/__init__.py
- [ ] src/anomaly_detection/config.py
- [ ] src/anomaly_detection/utils.py
- [ ] tests/conftest.py
- [ ] tests/phase_00/test_config.py
- [ ] tests/phase_00/test_structure.py
- [ ] docs/phase_00_guide.md

### 생성할 디렉터리 (9개)
- [ ] src/anomaly_detection/
- [ ] configs/datasets/
- [ ] configs/models/
- [ ] tests/phase_00/
- [ ] experiments/
- [ ] notebooks/
- [ ] outputs/
- [ ] docs/

### 통과해야 할 테스트
- [ ] test_config.py - 모든 테스트 통과 (19개 테스트)
- [ ] test_structure.py - 모든 테스트 통과 (18개 테스트)
- [ ] 총 37개 테스트 통과

### 품질 기준
- [ ] 테스트 커버리지 ≥ 90%
- [ ] Lint 에러 없음
- [ ] 모든 경로 올바르게 설정
- [ ] 문서 작성 완료

---

## 일반적인 문제 및 해결 방법

### 문제 1: Import 에러
```
ModuleNotFoundError: No module named 'anomaly_detection'
```

**해결 방법**:
```bash
# src를 PYTHONPATH에 추가
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 또는 개발 모드로 설치 (권장)
pip install -e .
```

### 문제 2: Config 경로 에러
```
FileNotFoundError: Config file not found: configs/paths.yaml
```

**해결 방법**:
- 프로젝트 루트에서 테스트 실행 확인
- 파일 존재 확인: `ls configs/paths.yaml`

### 문제 3: 절대 경로 불일치
```
AssertionError: Path 'dataset_root' is not absolute
```

**해결 방법**:
- `configs/paths.yaml`을 실제 절대 경로로 수정
- `/path/to/your/datasets`를 `/mnt/d/deep_learning/datasets` 같은 실제 경로로 변경

### 문제 4: outputs/ 권한 거부
```
PermissionError: [Errno 13] Permission denied: 'outputs/test_write.txt'
```

**해결 방법**:
```bash
# 디렉터리 권한 수정
chmod -R 755 outputs/
```

---

## 테스트 커버리지 리포트

예상 커버리지:
```
src/anomaly_detection/config.py    95%+
src/anomaly_detection/utils.py     90%+
전체                                 90%+
```

---

## 다음 단계

Phase 00 완료 후:
1. `main` 브랜치로 머지
2. 로컬 `main` 업데이트: `git checkout main && git pull`
3. Phase 01로 이동: Data Pipeline
4. 새 브랜치 생성: `git checkout -b feature/phase-01-data-pipeline`

---

## 참고 자료

- pytest 문서: https://docs.pytest.org/
- PyYAML 문서: https://pyyaml.org/wiki/PyYAMLDocumentation
- Python 프로젝트 구조 모범 사례: https://docs.python-guide.org/writing/structure/

---

## 주의사항

- 이것은 모든 개발자를 위한 공통 기반입니다
- Phase 00이 완료되고 머지될 때까지 Phase 01로 진행하지 마세요
- 모든 후속 Phase는 이 인프라에 의존합니다
- 이 브랜치를 깨끗하고 잘 테스트된 상태로 유지하세요

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-01-XX  
**작성자**: 개발팀