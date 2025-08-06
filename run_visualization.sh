#!/bin/bash

echo "PDAN Model Visualization Examples"
echo "================================="

# 환경 설정
export CUDA_VISIBLE_DEVICES=1

# 기본 경로 설정
CHECKPOINT_PATH="/data/1_personal/4_SWWOO/actiondetect/PDAN/pdan_original_25.33.ckpt"
DATA_ROOT="/data/1_personal/4_SWWOO/actiondetect/PDAN/data"
JSON_PATH="/data/1_personal/4_SWWOO/actiondetect/PDAN/charades.json"
OUTPUT_DIR="/data/1_personal/4_SWWOO/actiondetect/PDAN/visualizations"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Data Root: $DATA_ROOT"
echo "  JSON Path: $JSON_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# 출력 디렉토리 생성
mkdir -p $OUTPUT_DIR

echo "1. Analyzing multiple videos (default 5 videos)..."
python visualize_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --json_path "$JSON_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_videos 5 \
    --threshold 0.5

echo ""
echo "2. Analyzing a specific video..."
# 특정 비디오 ID로 분석 (예시)
SPECIFIC_VIDEO="0C5IQ"  # 실제 존재하는 비디오 ID로 변경 필요

python visualize_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --json_path "$JSON_PATH" \
    --output_dir "$OUTPUT_DIR/specific" \
    --video_id "$SPECIFIC_VIDEO" \
    --threshold 0.3

echo ""
echo "3. High confidence analysis (threshold=0.7)..."
python visualize_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --json_path "$JSON_PATH" \
    --output_dir "$OUTPUT_DIR/high_confidence" \
    --max_videos 3 \
    --threshold 0.7

echo ""
echo "4. Detailed analysis with more videos..."
python visualize_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --json_path "$JSON_PATH" \
    --output_dir "$OUTPUT_DIR/detailed" \
    --max_videos 10 \
    --threshold 0.4

echo ""
echo "Visualization completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - {video_id}_predictions.png: GT vs Prediction comparison"
echo "  - {video_id}_heatmap.png: Confidence heatmap"
