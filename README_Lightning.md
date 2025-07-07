# PDAN PyTorch Lightning Training

이 프로젝트는 PyTorch Lightning을 기반으로 한 최신 PDAN (Pyramidal Dilated Attention Network) 학습 시스템입니다.

## 주요 특징

- **PyTorch Lightning 기반**: 최신 딥러닝 프레임워크를 사용한 효율적인 학습
- **Multi-GPU 지원**: 분산 학습을 통한 빠른 훈련
- **자동 혼합 정밀도 (AMP)**: 메모리 효율성과 학습 속도 향상
- **고급 콜백 시스템**: 체크포인트, 조기 종료, 학습률 모니터링
- **다양한 로깅**: TensorBoard, Wandb 지원
- **구조화된 설정 관리**: Hydra를 통한 설정 파일 관리

## 설치

```bash
# 필수 패키지 설치
pip install -r requirements_lightning.txt

# 또는 conda를 사용하는 경우
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning wandb tensorboard hydra-core omegaconf
```

## 사용법

### 1. 기본 학습

```bash
# RGB 모드로 기본 학습
python train_pdan_lightning.py --mode rgb --batch_size 8 --max_epochs 50

# Flow 모드로 학습
python train_pdan_lightning.py --mode flow --batch_size 4 --max_epochs 50

# Skeleton 모드로 학습
python train_pdan_lightning.py --mode skeleton --batch_size 8 --max_epochs 50
```

### 2. 고급 설정을 사용한 학습

```bash
# 멀티 스테이지 모델로 학습
python train_pdan_lightning.py \
    --mode rgb \
    --num_stages 2 \
    --num_layers 8 \
    --num_channels 1024 \
    --batch_size 8 \
    --lr 0.0001 \
    --max_epochs 50 \
    --optimizer adamw \
    --scheduler cosine \
    --gpus 1 \
    --precision 16-mixed \
    --logger both \
    --wandb_project your_project_name
```

### 3. 멀티 GPU 학습

```bash
# 2개의 GPU로 분산 학습
python train_pdan_lightning.py \
    --mode rgb \
    --batch_size 16 \
    --gpus 2 \
    --strategy ddp \
    --precision 16-mixed \
    --max_epochs 50
```

### 4. Hydra 설정 파일 사용

```bash
# 기본 설정으로 학습
python train_hydra.py

# 설정 오버라이드
python train_hydra.py \
    model.num_stages=2 \
    training.batch_size=16 \
    training.lr=0.0002 \
    data.mode=flow \
    system.gpus=2
```

### 5. 사전 정의된 스크립트 실행

```bash
# 다양한 설정으로 학습 실행
./run_lightning_training.sh
```

### 6. 모델 테스트

```bash
# 사전 훈련된 모델로 테스트
python train_pdan_lightning.py \
    --test_only true \
    --ckpt_path ./lightning_logs/PDAN/experiment/checkpoints/best.ckpt \
    --mode rgb \
    --batch_size 1
```

## 설정 파일 구조

`config.yaml` 파일을 통해 모든 하이퍼파라미터를 관리할 수 있습니다:

```yaml
model:
  num_stages: 1
  num_layers: 5
  num_channels: 512
  input_channels: 1024
  num_classes: 157

training:
  batch_size: 8
  lr: 0.0001
  max_epochs: 50
  optimizer: "adamw"
  scheduler: "cosine"

data:
  mode: "rgb"
  rgb_root: "/path/to/rgb/features"
  flow_root: "/path/to/flow/features"
  skeleton_root: "/path/to/skeleton/features"

system:
  gpus: 1
  precision: "16-mixed"
```

## 주요 파라미터

### 모델 파라미터
- `num_stages`: 정제 스테이지 수 (기본값: 1)
- `num_layers`: 레이어 수 (기본값: 5)
- `num_channels`: 피처 채널 수 (기본값: 512)
- `input_channels`: 입력 피처 차원 (기본값: 1024, skeleton 모드에서는 256)
- `num_classes`: 클래스 수 (기본값: 157)

### 학습 파라미터
- `batch_size`: 배치 크기 (기본값: 8)
- `lr`: 학습률 (기본값: 0.0001)
- `max_epochs`: 최대 에포크 수 (기본값: 50)
- `optimizer`: 옵티마이저 (adamw, adam, sgd)
- `scheduler`: 학습률 스케줄러 (cosine, plateau, none)

### 시스템 파라미터
- `gpus`: 사용할 GPU 수 (기본값: 1)
- `precision`: 정밀도 (16-mixed, 32, bf16-mixed)
- `strategy`: 분산 학습 전략 (auto, ddp, ddp_spawn)

## 로깅 및 모니터링

### TensorBoard
```bash
# TensorBoard 실행
tensorboard --logdir lightning_logs
```

### Wandb
Wandb 사용을 위해서는 먼저 로그인이 필요합니다:
```bash
wandb login
```

## 체크포인트 관리

- 최고 성능 모델: `./lightning_logs/PDAN/experiment/checkpoints/best.ckpt`
- 최신 모델: `./lightning_logs/PDAN/experiment/checkpoints/last.ckpt`
- Top-K 모델들: 자동으로 저장됨

## 디버깅

개발 중에는 다음 옵션들을 사용할 수 있습니다:

```bash
# 빠른 개발 실행 (각 단계마다 몇 배치만 실행)
python train_pdan_lightning.py --fast_dev_run true

# 제한된 배치로 실행
python train_pdan_lightning.py --limit_train_batches 0.1 --limit_val_batches 0.1
```

## 성능 최적화 팁

1. **메모리 효율성**: `precision=16-mixed` 사용
2. **속도 향상**: 멀티 GPU 사용 (`gpus=2`, `strategy=ddp`)
3. **데이터 로딩**: `num_workers` 조정
4. **배치 크기**: GPU 메모리에 맞게 조정
5. **그래디언트 클리핑**: 안정적인 학습을 위해 `grad_clip=1.0` 사용

## 트러블슈팅

### 일반적인 문제들

1. **CUDA 메모리 부족**
   - 배치 크기를 줄이거나 `precision=16-mixed` 사용
   - `gradient_checkpointing` 활성화

2. **느린 데이터 로딩**
   - `num_workers` 값을 조정
   - `pin_memory=True` 사용

3. **분산 학습 문제**
   - 방화벽 설정 확인
   - `strategy=ddp_spawn` 시도

## 예제 결과

성공적인 학습 후에는 다음과 같은 결과를 얻을 수 있습니다:

```
Epoch 49: 100%|██████████| 1250/1250 [05:23<00:00, 3.86it/s, v_num=0, train_loss=0.125, val_loss=0.138, val_map=75.3]
Testing: 100%|██████████| 500/500 [02:15<00:00, 3.69it/s]
Test mAP: 76.8%
```

## 기여하기

이 프로젝트에 기여하고 싶으시다면:

1. 이슈를 생성하여 문제를 보고하거나 기능을 제안하세요
2. 코드 개선사항을 제안하세요
3. 문서화 개선에 도움을 주세요

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
