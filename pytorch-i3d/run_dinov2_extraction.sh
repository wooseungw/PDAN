#!/bin/bash

echo "Starting DINOv2 Feature Extraction..."

# 환경 설정
export CUDA_VISIBLE_DEVICES=0

# 기본 설정
MODEL_SIZE="dinov2_vitb14"  # dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
ROOT_DIR="/data/1_personal/4_SWWOO/actiondetect/PDAN/Charades_v1_rgb"
SPLIT_FILE="/data/1_personal/4_SWWOO/actiondetect/PDAN/charades.json"
SAVE_DIR="/data/1_personal/4_SWWOO/actiondetect/PDAN/data/dinov2_features"
BATCH_SIZE=1

echo "Configuration:"
echo "  Model: $MODEL_SIZE"
echo "  Data Root: $ROOT_DIR"
echo "  Split File: $SPLIT_FILE"
echo "  Save Directory: $SAVE_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# DINOv2 특징 추출 실행
python extract_features_dinov2.py \
    -model_size $MODEL_SIZE \
    -root "$ROOT_DIR" \
    -split_file "$SPLIT_FILE" \
    -save_dir "$SAVE_DIR" \
    -batch_size $BATCH_SIZE \
    -gpu 0

echo ""
echo "DINOv2 feature extraction completed!"
echo "Features saved to: $SAVE_DIR"
