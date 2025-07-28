#!/bin/bash

# PDAN Model Evaluation Script
# Usage: ./run_evaluation.sh

set -e  # Exit on any error

echo "Starting PDAN Model Evaluation..."

# Configuration
CHECKPOINT_PATH="./lightning_logs/checkpoints/pdan-epoch=39-val_map=10.39.ckpt"
DATA_ROOT="/data/1_personal/4_SWWOO/actiondetect/PDAN/data"
SPLIT_FILE="/data/1_personal/4_SWWOO/actiondetect/PDAN/data/charades.json"
OUTPUT_DIR="./evaluation_results_$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Data root: $DATA_ROOT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la ./lightning_logs/checkpoints/ || echo "No checkpoints directory found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Running evaluation..."
python evaluate_pdan.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --split_file "$SPLIT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo ""
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - evaluation_summary.json: Overall evaluation metrics"
echo "  - class_ap_scores.csv: AP scores per action class"
echo "  - video_results_summary.csv: Per-video evaluation summary"
echo "  - map_per_class.png: mAP visualization per class"
echo "  - precision_recall_curves.png: P-R curves for top classes"
echo "  - prediction_statistics.png: Various prediction statistics"
echo "  - temporal_predictions_*.png: Temporal predictions for sample videos"
echo ""
echo "You can view the visualizations using:"
echo "  eog $OUTPUT_DIR/*.png  # or your preferred image viewer"