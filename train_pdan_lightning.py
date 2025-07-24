#!/usr/bin/env python3
"""
PDAN Training with PyTorch Lightning
Action Detection using Pyramid Dilated Attention Network
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PDAN import PDAN
from apmeter import APMeter
from charades_i3d_per_video import MultiThumos as Dataset
from charades_i3d_per_video import mt_collate_fn as collate_fn


class PDANLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for PDAN model"""
    
    def __init__(self, 
                 stage=1, 
                 block=5, 
                 num_channel=512, 
                 input_channel=1024, 
                 num_classes=157,
                 learning_rate=0.001,
                 weight_decay=1e-4):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model parameters
        self.stage = stage
        self.block = block
        self.num_channel = num_channel
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize model
        self.model = PDAN(stage, block, num_channel, input_channel, num_classes)
        
        # Initialize AP meters
        self.train_apm = APMeter()
        self.val_apm = APMeter()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total trainable parameters: {total_params:,}')
        
    def forward(self, inputs, mask):
        """Forward pass"""
        return self.model(inputs, mask)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # maximize validation mAP
            factor=0.5, 
            patience=8,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_map',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def prepare_data_and_masks(self, batch):
        """Prepare input data and masks"""
        inputs, mask, labels, other = batch
        
        # Calculate mask statistics
        mask_list = torch.sum(mask, 1)
        batch_size = mask.size(0)
        seq_len = mask.size(1)
        
        # Create new mask for model
        mask_new = torch.zeros((batch_size, self.num_classes, seq_len), 
                              dtype=torch.float32, device=mask.device)
        
        for i in range(batch_size):
            valid_len = int(mask_list[i])
            if valid_len > 0:
                mask_new[i, :, :valid_len] = 1.0
        
        # Prepare inputs (remove spatial dimensions)
        inputs = inputs.squeeze(3).squeeze(3)
        
        return inputs, mask, mask_new, labels, other
    
    def compute_loss_and_metrics(self, batch, stage='train'):
        """Compute loss and metrics for training/validation"""
        inputs, mask, mask_new, labels, other = self.prepare_data_and_masks(batch)
        
        # Forward pass
        activation = self.forward(inputs, mask_new)
        
        # Get final outputs
        outputs_final = activation[-1]  # Take last stage output
        outputs_final = outputs_final.permute(0, 2, 1)  # [B, T, C]
        
        # Calculate probabilities and loss
        probs = F.sigmoid(outputs_final) * mask.unsqueeze(2)
        loss = F.binary_cross_entropy_with_logits(
            outputs_final, labels, reduction='sum'
        )
        loss = loss / torch.sum(mask)  # Normalize by valid timesteps
        
        # Calculate accuracy (dummy metric for now)
        accuracy = torch.sum(mask) / torch.sum(mask)  # Always 1.0, placeholder
        
        return {
            'loss': loss,
            'probs': probs,
            'labels': labels,
            'accuracy': accuracy,
            'outputs': outputs_final
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        results = self.compute_loss_and_metrics(batch, 'train')
        
        # Add to AP meter
        probs_np = results['probs'].detach().cpu().numpy()
        labels_np = results['labels'].detach().cpu().numpy()
        
        if probs_np.shape[0] > 0:  # Check if batch is not empty
            self.train_apm.add(probs_np[0], labels_np[0])
        
        # Log metrics with explicit batch size
        current_batch_size = batch[0].size(0) if hasattr(batch[0], 'size') else 1
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
        self.log('train_acc', results['accuracy'], on_step=True, on_epoch=True, batch_size=current_batch_size)
        
        return results['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        results = self.compute_loss_and_metrics(batch, 'val')
        
        # Add to AP meter
        probs_np = results['probs'].detach().cpu().numpy()
        labels_np = results['labels'].detach().cpu().numpy()
        
        if probs_np.shape[0] > 0:  # Check if batch is not empty
            self.val_apm.add(probs_np[0], labels_np[0])
        
        # Log metrics with explicit batch size
        current_batch_size = batch[0].size(0) if hasattr(batch[0], 'size') else 1
        self.log('val_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
        self.log('val_acc', results['accuracy'], on_step=False, on_epoch=True, batch_size=current_batch_size)
        
        return results['loss']
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        try:
            train_ap = self.train_apm.value()
            if train_ap is not None and len(train_ap) > 0:
                if isinstance(train_ap, torch.Tensor):
                    train_map = 100 * torch.mean(train_ap)
                else:
                    train_map = 100 * torch.mean(torch.as_tensor(train_ap, dtype=torch.float32))
                self.log('train_map', train_map, on_epoch=True, prog_bar=True)
                print(f'Train mAP: {train_map:.2f}%')
        except Exception as e:
            print(f'Error calculating train mAP: {e}')
        
        # Reset AP meter
        self.train_apm.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        try:
            val_ap = self.val_apm.value()
            if val_ap is not None and len(val_ap) > 0:
                if isinstance(val_ap, torch.Tensor):
                    val_map = 100 * torch.mean(val_ap)
                else:
                    val_map = 100 * torch.mean(torch.as_tensor(val_ap, dtype=torch.float32))
                self.log('val_map', val_map, on_epoch=True, prog_bar=True)
                print(f'Val mAP: {val_map:.2f}%')
            else:
                # Fallback: use validation loss as metric
                print('Warning: Could not calculate val mAP, using loss-based metric')
                self.log('val_map', 0.0, on_epoch=True, prog_bar=True)
        except Exception as e:
            print(f'Error calculating val mAP: {e}')
            self.log('val_map', 0.0, on_epoch=True, prog_bar=True)
        
        # Reset AP meter
        self.val_apm.reset()


class PDANDataModule(pl.LightningDataModule):
    """PyTorch Lightning Data Module for PDAN"""
    
    def __init__(self, 
                 train_split_path,
                 val_split_path,
                 data_root,
                 batch_size=1,
                 num_workers=4,
                 num_classes=157):
        super().__init__()
        
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        
    def setup(self, stage=None):
        """Setup datasets"""
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = Dataset(
                self.train_split_path, 
                'training', 
                self.data_root, 
                self.batch_size, 
                self.num_classes
            )
            
            # Validation dataset
            self.val_dataset = Dataset(
                self.val_split_path, 
                'testing', 
                self.data_root, 
                1,  # Validation batch size = 1
                self.num_classes
            )
            
        print(f'Training samples: {len(self.train_dataset) if hasattr(self, "train_dataset") else 0}')
        print(f'Validation samples: {len(self.val_dataset) if hasattr(self, "val_dataset") else 0}')
    
    def train_dataloader(self):
        """Training dataloader"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else 2  # 데이터 프리페칭 최적화
        )
    
    def val_dataloader(self):
        """Validation dataloader"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='PDAN Training with PyTorch Lightning')
    
    # Model arguments
    parser.add_argument('--stage', type=int, default=1, help='Number of stages')
    parser.add_argument('--block', type=int, default=5, help='Number of blocks per stage')
    parser.add_argument('--num_channel', type=int, default=512, help='Number of channels')
    parser.add_argument('--input_channel', type=int, default=1024, help='Input channel dimension')
    parser.add_argument('--num_classes', type=int, default=157, help='Number of action classes')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, 
                       default='/data/1_personal/4_SWWOO/actiondetect/PDAN/data',
                       help='Root directory for data')
    parser.add_argument('--train_split', type=str,
                       default='/data/1_personal/4_SWWOO/actiondetect/PDAN/data/charades.json',
                       help='Training split JSON file')
    parser.add_argument('--val_split', type=str,
                       default='/data/1_personal/4_SWWOO/actiondetect/PDAN/data/charades.json',
                       help='Validation split JSON file')
    
    # Training configuration
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Training precision')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Logging
    parser.add_argument('--experiment_name', type=str, default='pdan_experiment', help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='./lightning_logs', help='Directory for logs')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Create data module
    data_module = PDANDataModule(
        train_split_path=args.train_split,
        val_split_path=args.val_split,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes
    )
    
    # Create model
    model = PDANLightningModule(
        stage=args.stage,
        block=args.block,
        num_channel=args.num_channel,
        input_channel=args.input_channel,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{args.log_dir}/checkpoints',
        filename='pdan-{epoch:02d}-{val_map:.2f}',
        monitor='val_map',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 and torch.cuda.is_available() else 'cpu',
        devices=args.gpus if args.gpus > 0 and torch.cuda.is_available() else 'auto',
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=1,
        val_check_interval=1.0,  # Validate every epoch
        log_every_n_steps=5  # 10에서 5로 감소 (더 자주 로깅)
    )
    
    # Print configuration
    print("=" * 60)
    print("PDAN Training Configuration")
    print("=" * 60)
    print(f"Model: stage={args.stage}, block={args.block}, channels={args.num_channel}")
    print(f"Data: {args.data_root}")
    print(f"Training: batch_size={args.batch_size}, lr={args.learning_rate}, epochs={args.max_epochs}")
    print(f"Hardware: {args.gpus} GPU(s), precision={args.precision}")
    print("=" * 60)
    
    # Start training
    try:
        # Handle checkpoint resuming in the new way
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
        else:
            trainer.fit(model, data_module)
            
        print("Training completed successfully!")
        
        # Print best checkpoint path
        if checkpoint_callback.best_model_path:
            print(f"Best model saved at: {checkpoint_callback.best_model_path}")
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
