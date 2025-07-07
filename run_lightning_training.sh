#!/bin/bash

# PyTorch Lightning PDAN Training Scripts
# Various training configurations and examples

echo "=== PyTorch Lightning PDAN Training ==="

# Basic RGB training
echo "1. Basic RGB Training..."
python train_pdan_lightning.py \
    --mode rgb \
    --batch_size 8 \
    --lr 0.0001 \
    --max_epochs 50 \
    --num_stages 1 \
    --num_layers 5 \
    --num_channels 512 \
    --optimizer adamw \
    --scheduler cosine \
    --gpus 1 \
    --precision 16-mixed \
    --project_name PDAN \
    --exp_name rgb_basic \
    --logger tensorboard \
    --save_top_k 3 \
    --early_stopping_patience 15

# Enhanced Flow training with multi-stage
echo "2. Enhanced Flow Training (Multi-stage)..."
python train_pdan_lightning.py \
    --mode flow \
    --batch_size 4 \
    --lr 0.0001 \
    --max_epochs 50 \
    --num_stages 2 \
    --num_layers 8 \
    --num_channels 1024 \
    --optimizer adamw \
    --scheduler cosine \
    --gpus 1 \
    --precision 16-mixed \
    --project_name PDAN \
    --exp_name flow_enhanced \
    --logger both \
    --wandb_project pdan-flow-detection \
    --save_top_k 5 \
    --early_stopping_patience 20

# Skeleton training
echo "3. Skeleton Training..."
python train_pdan_lightning.py \
    --mode skeleton \
    --batch_size 8 \
    --lr 0.0001 \
    --max_epochs 50 \
    --num_stages 1 \
    --num_layers 5 \
    --num_channels 512 \
    --input_channels 256 \
    --optimizer adamw \
    --scheduler plateau \
    --gpus 1 \
    --precision 16-mixed \
    --project_name PDAN \
    --exp_name skeleton_basic \
    --logger tensorboard \
    --save_top_k 3

# Multi-GPU training (if available)
echo "4. Multi-GPU Training..."
python train_pdan_lightning.py \
    --mode rgb \
    --batch_size 16 \
    --lr 0.0002 \
    --max_epochs 50 \
    --num_stages 2 \
    --num_layers 8 \
    --num_channels 1024 \
    --optimizer adamw \
    --scheduler cosine \
    --gpus 2 \
    --strategy ddp \
    --precision 16-mixed \
    --project_name PDAN \
    --exp_name rgb_multigpu \
    --logger both \
    --wandb_project pdan-multigpu \
    --save_top_k 5

# Fast development run (for debugging)
echo "5. Fast Development Run..."
python train_pdan_lightning.py \
    --mode rgb \
    --batch_size 2 \
    --lr 0.0001 \
    --max_epochs 5 \
    --num_stages 1 \
    --num_layers 3 \
    --num_channels 256 \
    --optimizer adamw \
    --scheduler cosine \
    --gpus 1 \
    --precision 32 \
    --project_name PDAN \
    --exp_name debug \
    --logger tensorboard \
    --fast_dev_run true \
    --limit_train_batches 0.1 \
    --limit_val_batches 0.1

# Testing with pretrained model
echo "6. Testing with pretrained model..."
python train_pdan_lightning.py \
    --test_only true \
    --ckpt_path ./lightning_logs/PDAN/rgb_basic/checkpoints/best.ckpt \
    --mode rgb \
    --batch_size 1 \
    --project_name PDAN \
    --exp_name test_rgb \
    --logger tensorboard \
    --gpus 1

echo "=== Training Scripts Completed ==="
