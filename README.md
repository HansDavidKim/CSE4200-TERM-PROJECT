# CSE4200 Term Project: PPO-based Recommender System

## 📋 프로젝트 개요
이 프로젝트는 **PPO (Proximal Policy Optimization)** 기반 강화학습 추천 시스템을 구현하고, **GRU4Rec** (순차 추천 모델) 및 **Hybrid (CBF)** 모델과 성능을 비교합니다. RecSim 시뮬레이션 환경을 사용하여 사용자 관심사 변화(drift)가 있는 동적 환경에서 모델을 평가합니다.

### 주요 성과
- **PPO가 GRU4Rec의 98-99% 성능 달성** (모든 drift 환경에서)
- **Hybrid 모델 대비 15-18점 우수한 성능**
- **Drift 환경에 강건한 성능** (drift 0.1 → 1.0에서도 안정적)

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정

**⚠️ 중요: Python 3.11+ 버전 필요 (tomllib 지원)**

```bash
# Conda 환경 생성 (권장)
conda create -n rec_test python=3.11 -y
conda activate rec_test

# 의존성 패키지 설치
pip install -r requirements.txt
```

또는 기존 Python 3.11+ 환경에서:
```bash
pip install -r requirements.txt
```

### 2. 전체 벤치마크 실행
```bash
python run_full_benchmark.py
```

이 명령어 하나로 다음이 자동 실행됩니다:
- ✅ 3가지 drift 환경(0.1, 0.5, 1.0)에 대한 데이터 생성
- ✅ GRU4Rec 및 PPO 모델 학습
- ✅ 3가지 모델(GRU4Rec, PPO, Hybrid) 평가
- ✅ 결과를 CSV/JSON으로 저장 (`experiments/benchmark/`)

**예상 소요 시간**: 약 30-40분 (CPU 기준)

### 3. 결과 확인
```bash
# CSV 결과 확인
cat experiments/benchmark/benchmark_results_final.csv

# 또는 Jupyter Notebook 사용
jupyter notebook experiment.ipynb
```

---

## 📊 벤치마크 결과 요약

| Drift | Model | Avg Reward | Coverage | CTR |
|:---:|:---:|:---:|:---:|:---:|
| **0.1** | GRU4Rec | **176.10** | 1.000 | 0.690 |
| 0.1 | PPO | 172.71 | 0.132 | 0.592 |
| 0.1 | Hybrid | 157.55 | 0.982 | 0.567 |
| **0.5** | GRU4Rec | **173.13** | 1.000 | 0.662 |
| 0.5 | PPO | 172.46 | 0.130 | 0.572 |
| 0.5 | Hybrid | 155.57 | 0.976 | 0.559 |
| **1.0** | GRU4Rec | **174.12** | 1.000 | 0.625 |
| 1.0 | PPO | 169.88 | 0.134 | 0.505 |
| 1.0 | Hybrid | 151.24 | 0.982 | 0.488 |

**핵심 인사이트**:
- PPO는 GRU4Rec과 거의 동등한 성능 (차이 < 5점)
- PPO는 Hybrid 대비 15-18점 우수
- Coverage는 낮지만 높은 보상을 달성하는 전략적 선택

---

## 🗂️ 프로젝트 구조

```
CSE4200-TERM-PROJECT/
├── main.py                      # CLI 진입점
├── run_full_benchmark.py        # 전체 벤치마크 자동화 스크립트
├── experiment.ipynb             # Jupyter Notebook (대화형 실행)
├── requirements.txt             # 의존성 패키지
├── configs/
│   └── default.toml            # 기본 설정
├── recommender/
│   ├── gru4rec.py              # GRU4Rec 모델
│   ├── ppo_agent.py            # PPO 에이전트
│   ├── train.py                # 학습 로직
│   ├── evaluate.py             # 평가 로직
│   └── utils.py                # 유틸리티
├── rec_sim/                     # RecSim 시뮬레이션 환경
├── matrix_factorization/        # 데이터 전처리
└── experiments/                 # 실험 결과 (자동 생성)
    ├── raw_dataset/            # 원본 데이터셋
    ├── dataset/                # 전처리된 데이터셋
    ├── baseline/               # GRU4Rec 모델
    ├── ppo/                    # PPO 모델
    └── benchmark/              # 평가 결과 (CSV/JSON)
```

---

## 🔧 개별 명령어 사용법

### 데이터 생성
```bash
python main.py generate-data --num-users 500 --num-items 500 --steps 30 --drift-scale 0.1
```

### 데이터 전처리
```bash
python main.py process-dataset --input-file dataset/data.csv --output-dir preprocessed
```

### 모델 학습

**GRU4Rec:**
```bash
python main.py train gru4rec --input-dir preprocessed --epochs 5 --device cpu
```

**PPO:**
```bash
python main.py train ppo --num-items 500 --epochs 10 --drift-scale 0.1 --max-steps 100 --device cpu
```

### 모델 평가

**GRU4Rec:**
```bash
python main.py evaluate gru4rec --model-path experiments/baseline/gru4rec_0_1 --drift 0.1 --max-steps 150
```

**PPO:**
```bash
python main.py evaluate ppo --model-path experiments/ppo/best_0_1/ppo_model_epoch_10.pth --drift 0.1 --max-steps 150
```

**Hybrid:**
```bash
python main.py evaluate hybrid --model-path experiments/baseline/gru4rec_0_1 --drift 0.1 --max-steps 150
```

---

## 📝 주요 하이퍼파라미터

### PPO 학습
- `--epochs`: 학습 에포크 수 (기본: 10)
- `--max-steps`: 에피소드당 최대 스텝 (기본: 100)
- `--top-k`: Top-K 샘플링 (기본: 7)
- `--similarity-coef`: 유사도 기반 soft labeling 계수 (기본: 0.50)
- `--entropy-coef`: 엔트로피 정규화 계수 (기본: 0.15)
- `--drift-scale`: 환경 drift 강도 (0.1, 0.5, 1.0)

### GRU4Rec 학습
- `--epochs`: 학습 에포크 수 (기본: 5)
- `--batch-size`: 배치 크기 (기본: 256)
- `--embedding-dim`: 임베딩 차원 (기본: 64)
- `--hidden-size`: 은닉층 크기 (기본: 128)

---

## 📈 실험 재현 가이드

### Option 1: 자동화 스크립트 (권장)
```bash
python run_full_benchmark.py
```

### Option 2: Jupyter Notebook
```bash
jupyter notebook experiment.ipynb
```
노트북에서 셀을 순차적으로 실행하면 됩니다.

### Option 3: 수동 실행
```bash
# 1. 데이터 생성 및 전처리
python main.py generate-data --num-users 500 --num-items 500 --steps 30 --drift-scale 0.1 --output-dir experiments/raw_dataset/0_1
python main.py process-dataset --input-file experiments/raw_dataset/0_1/data_500_500_30.csv --output-dir experiments/dataset/0_1

# 2. GRU4Rec 학습
python main.py train gru4rec --input-dir experiments/dataset/0_1 --epochs 5 --output-dir experiments/baseline/gru4rec_0_1

# 3. PPO 학습
python main.py train ppo --num-items 500 --epochs 10 --drift-scale 0.1 --max-steps 100 --output-dir experiments/ppo/best_0_1

# 4. 평가
python main.py evaluate gru4rec --model-path experiments/baseline/gru4rec_0_1 --drift 0.1 --max-steps 150 --output-dir experiments/benchmark/gru4rec_0_1
python main.py evaluate ppo --model-path experiments/ppo/best_0_1/ppo_model_epoch_10.pth --drift 0.1 --max-steps 150 --output-dir experiments/benchmark/ppo_0_1
python main.py evaluate hybrid --model-path experiments/baseline/gru4rec_0_1 --drift 0.1 --max-steps 150 --output-dir experiments/benchmark/hybrid_0_1
```

---

## 🎯 핵심 기술

### PPO Agent
- **Actor-Critic 아키텍처**: GRU 기반 상태 표현 + DeepSets 슬레이트 인코딩
- **Top-K Sampling**: 다양성과 성능의 균형
- **Similarity-based Soft Labeling**: 클릭된 아이템과 유사한 아이템에 대한 학습 신호 전파
- **Behavior Cloning**: GRU4Rec 임베딩으로 초기화

### GRU4Rec
- **순차 모델링**: GRU를 사용한 사용자 행동 시퀀스 학습
- **LightGCN 초기화**: 그래프 기반 임베딩으로 cold-start 완화

### Hybrid (CBF)
- **Content-Based Filtering**: 아이템 임베딩 기반 유사도 계산
- **EMA 업데이트**: 사용자 프로필의 점진적 업데이트

---

## 📚 참고 문헌
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- GRU4Rec: Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks" (2016)
- RecSim: Ie et al., "RecSim: A Configurable Simulation Platform for Recommender Systems" (2019)

