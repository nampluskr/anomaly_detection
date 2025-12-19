## Anomaly Detection

### 목표
- SOTA Vision Anomaly Detection 평가 및 학습 프레임워크 구축
- Anomalib 라이브러리의 최신 SOTA 모델 지원 (torch_model.py 그대로 사용)
- Lightning 기반 학습/평가 엔진 구현을 순수 PyTorch로 전환 (lightning_model.py 대체 구현)
- 표준 벤치마크 데이터셋(MVTec, VisA, BTAD)에서 모델 성능 검증
- pytest 기반 TDD 방식으로 개발 (브랜치 기반 단계별 확장 구현)

### 제약 사항

#### 1. 로컬 환경 제약 극복

로컬 실행 환경으로 보안 정책에 따른 방화벽으로 외부 네트워크 접근 제한

**문제점:**
- **사전학습 가중치 다운로드 불가**: ResNet, Wide ResNet, DINOv2 등의 백본 가중치를 자동으로 다운로드할 수 없음
- **보조 데이터셋 접근 불가**: DRAEM의 DTD, EfficientAD의 Imagenette2 등 필수 데이터셋 다운로드 불가
- **라이브러리 설치 제한**: PyPI 접근 제한으로 의존성 관리 복잡

**해결 방안:**
- 외부 환경에서 모든 가중치 파일을 사전 다운로드하여 `backbones/` 폴더에 저장
- 보조 데이터셋을 `datasets/` 폴더에 사전 배치
- 최소한의 핵심 라이브러리만 사용하여 의존성 최소화

#### 2. Lightning 의존성 제거

Anomalib은 PyTorch Lightning을 학습/평가 엔진(래퍼)로 사용하지만 로컬 환경에서는 사용 제한

**문제점:**
- **라이브러리 호환성 문제**: Lightning과 관련 의존성 패키지 간 버전 충돌
- **불필요한 복잡성**: 단순 학습 파이프라인에 과도한 추상화
- **설치 오류**: 오프라인 환경에서 Lightning 설치 실패

**해결 방안:**
- 순수 PyTorch만을 사용한 학습 파이프라인 구현
- BaseTrainer 클래스를 통한 통합 학습 인터페이스 제공
- Hook 패턴을 통한 모델별 커스터마이징 지원


## Project Stucture
```
projects/
├── datasets/                       # 이상 탐지 데이터셋 저장 디렉터리
│   ├── mvtec/                      # MVtec (train/test/ground_truth)
│   ├── visa/                       # ViSA (images/annotations)
│   └── btad/                       # BTAD
├── backbones/                      # 사전 학습 백본 가중치 저장
│   └── *.pth
└── anomaly_detection/              # 프로젝트 메인 디렉터리
```

### 프로젝트 루트 폴더 구조
```
anomaly_detection/                  # 프로젝트 루트 (Git 저장소)
│
├── src/                            # 소스 코드
│   └── anomaly_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── datasets.py
│       ├── dataloader.py          # DataLoader 생성 함수
│       ├── trainer.py
│       ├── evaluator.py
│       ├── utils.py
│       ├── models/
│       ├── transforms/
│       └── metrics/
│
├── configs/                        # 설정 파일
│   ├── default.yaml
│   ├── paths.yaml
│   ├── datasets/
│   │   ├── mvtec.yaml
│   │   ├── visa.yaml
│   │   └── btad.yaml
│   └── models/
│       ├── stfpm.yaml
│       └── efficientad.yaml
│
├── tests/                          # 테스트 코드
│   ├── conftest.py
│   ├── phase_00/
│   │   ├── test_config.py
│   │   └── test_structure.py
│   ├── phase_01/
│   │   ├── test_dataset.py
│   │   ├── test_dataloader.py
│   │   ├── test_transforms.py
│   │   └── test_validation.py
│   └── phase_02/
│       ├── test_stfpm_model.py
│       ├── test_stfpm_trainer.py
│       └── test_stfpm_evaluation.py
│
├── experiments/                    # 실험 스크립트
│   ├── train_stfpm.py
│   └── train_efficientad.py
│
├── notebooks/                      # Jupyter 노트북
│   ├── eval_stfpm.ipynb
│   └── eval_efficientad.ipynb
│
├── outputs/                        # 실험 출력물 (Git 제외)
│
├── docs/                           # 문서
│
├── .gitignore
├── pytest.ini
└── README.md
```