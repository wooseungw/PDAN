#!/bin/bash

# P# Default configuration
STAGE=1
BLOCK=5
NUM_CHANNEL=512
INPUT_CHANNEL=1024
NUM_CLASSES=157
BATCH_SIZE=4  # 1에서 4로 증가
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
NUM_WORKERS=8  # 워커 수 증가
MAX_EPOCHS=100
GPUS=2
PRECISION=32 Training Script
# Usage: ./run_lightning_training.sh

set -e  # Exit on any error

echo "Starting PDAN Training with PyTorch Lightning..."

# Set CUDA environment variables for compatibility
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

# Disable CUDNN optimizations to avoid compatibility issues
export CUDNN_DETERMINISTIC=1

# Default configuration
STAGE=1
BLOCK=5
NUM_CHANNEL=512
INPUT_CHANNEL=1024
NUM_CLASSES=157
BATCH_SIZE=16
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
NUM_WORKERS=4
MAX_EPOCHS=100
GPUS=2
PRECISION=32

# Data paths
DATA_ROOT="/data/1_personal/4_SWWOO/actiondetect/PDAN/data"
TRAIN_SPLIT="/data/1_personal/4_SWWOO/actiondetect/PDAN/data/charades.json"
VAL_SPLIT="/data/1_personal/4_SWWOO/actiondetect/PDAN/data/charades.json"

# Logging
EXPERIMENT_NAME="pdan_rgb_experiment_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./lightning_logs"

echo "Configuration:"
echo "  Model: stage=$STAGE, block=$BLOCK, channels=$NUM_CHANNEL"
echo "  Training: batch_size=$BATCH_SIZE, lr=$LEARNING_RATE, epochs=$MAX_EPOCHS"
echo "  Data: $DATA_ROOT"
echo "  GPUs: $GPUS, Precision: $PRECISION"
echo "  Experiment: $EXPERIMENT_NAME"
echo ""

# Create log directory
mkdir -p $LOG_DIR

# Run training
python train_pdan_lightning.py \
    --stage $STAGE \
    --block $BLOCK \
    --num_channel $NUM_CHANNEL \
    --input_channel $INPUT_CHANNEL \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_epochs $MAX_EPOCHS \
    --num_workers $NUM_WORKERS \
    --data_root "$DATA_ROOT" \
    --train_split "$TRAIN_SPLIT" \
    --val_split "$VAL_SPLIT" \
    --gpus $GPUS \
    --precision $PRECISION \
    --experiment_name "$EXPERIMENT_NAME" \
    --log_dir "$LOG_DIR"

echo ""
echo "Training completed!"
echo "Logs saved to: $LOG_DIR/$EXPERIMENT_NAME"
echo "Checkpoints saved to: $LOG_DIR/checkpoints"
