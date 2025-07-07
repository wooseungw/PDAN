#!/usr/bin/env python3
"""
PyTorch Lightning based PDAN Training Script
Modern implementation with advanced features
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.types import STEP_OUTPUT
import numpy as np

# Custom imports
from PDAN import PDAN
from charades_i3d_per_video import MultiThumos as Dataset
from charades_i3d_per_video import mt_collate_fn as collate_fn
from apmeter import APMeter


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class PDANDataModule(pl.LightningDataModule):
    """PyTorch Lightning Data Module for PDAN."""
    
    def __init__(
        self,
        mode: str = 'rgb',
        batch_size: int = 8,
        num_workers: int = 8,
        train_split: str = './data/charades.json',
        val_split: str = './data/charades.json',
        rgb_root: str = '/Path/to/charades_feat_rgb',
        flow_root: str = '/Path/to/charades_feat_flow',
        skeleton_root: str = '/Path/to/charades_feat_pose',
        num_classes: int = 157,
        **kwargs
    ):
        super().__init__()
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.skeleton_root = skeleton_root
        self.num_classes = num_classes
        
        # Select data root based on mode
        if mode == 'rgb':
            self.data_root = rgb_root
        elif mode == 'flow':
            self.data_root = flow_root
        elif mode == 'skeleton':
            self.data_root = skeleton_root
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = Dataset(
                self.train_split,
                'training',
                self.data_root,
                self.batch_size,
                self.num_classes
            )
            
            # Validation dataset
            self.val_dataset = Dataset(
                self.val_split,
                'testing',
                self.data_root,
                1,  # Batch size 1 for validation
                self.num_classes
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = Dataset(
                self.val_split,
                'testing',
                self.data_root,
                1,
                self.num_classes
            )
        
        if stage == "predict" or stage is None:
            self.predict_dataset = Dataset(
                self.val_split,
                'testing',
                self.data_root,
                1,
                self.num_classes
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def predict_dataloader(self):
        """Return prediction dataloader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
        )


class PDANLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for PDAN."""
    
    def __init__(
        self,
        num_stages: int = 1,
        num_layers: int = 5,
        num_channels: int = 512,
        input_channels: int = 1024,
        num_classes: int = 157,
        lr: float = 0.0001,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        ap_type: str = 'wap',
        optimizer: str = 'adamw',
        scheduler: str = 'cosine',
        grad_clip: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.ap_type = ap_type
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.grad_clip = grad_clip
        
        # Build model
        self.model = PDAN(
            num_stages=num_stages,
            num_layers=num_layers,
            num_f_maps=num_channels,
            dim=input_channels,
            num_classes=num_classes
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Metrics
        self.train_apm = APMeter()
        self.val_apm = APMeter()
        self.test_apm = APMeter()
        
        # Best validation mAP
        self.best_val_map = 0.0
        
        # Store validation predictions for final evaluation
        self.val_predictions = {}
        self.test_predictions = {}
        
        # Automatic gradient clipping
        self.automatic_optimization = True
        
        # Log model architecture
        self.log_model_info()
    
    def log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: PDAN")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Stages: {self.num_stages}, Layers: {self.num_layers}")
        print(f"Channels: {self.num_channels}, Input channels: {self.input_channels}")
        print(f"Number of classes: {self.num_classes}")
    
    def forward(self, x, mask):
        """Forward pass."""
        return self.model(x, mask)
    
    def prepare_data_batch(self, batch):
        """Prepare data batch for training/validation."""
        inputs, mask, labels, other = batch
        
        # Prepare mask for model
        mask_list = torch.sum(mask, 1)
        mask_new = torch.zeros((mask.size(0), self.num_classes, mask.size(1)))
        
        for i in range(mask.size(0)):
            if mask_list[i] > 0:
                mask_new[i, :, :int(mask_list[i])] = 1.0
        
        mask_new = mask_new.to(self.device)
        
        # Remove spatial dimensions
        inputs = inputs.squeeze(3).squeeze(3)
        
        return inputs, mask, mask_new, labels, other
    
    def compute_loss(self, outputs, labels, mask):
        """Compute loss."""
        # Use final stage output
        final_output = outputs[-1].permute(0, 2, 1)
        
        # Binary cross entropy loss
        loss = self.criterion(final_output, labels)
        
        # Mask the loss
        loss = loss * mask.unsqueeze(2)
        
        # Average over valid positions
        valid_positions = torch.sum(mask)
        if valid_positions > 0:
            loss = torch.sum(loss) / valid_positions
        else:
            loss = torch.sum(loss)
        
        return loss, final_output
    
    def compute_metrics(self, outputs, labels, mask, apm, phase='train'):
        """Compute metrics."""
        # Get probabilities
        probs = torch.sigmoid(outputs) * mask.unsqueeze(2)
        
        # Add to AP meter
        for i in range(probs.size(0)):
            apm.add(probs[i].detach().cpu().numpy(), labels[i].detach().cpu().numpy())
        
        # Calculate mAP
        if self.ap_type == 'wap':
            map_score = 100 * apm.value()
        else:
            ap_values = 100 * apm.value()
            valid_aps = ap_values[ap_values > 0]
            map_score = valid_aps.mean() if len(valid_aps) > 0 else 0.0
        
        return map_score
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        inputs, mask, mask_new, labels, other = self.prepare_data_batch(batch)
        
        # Forward pass
        outputs = self(inputs, mask_new)
        
        # Compute loss
        loss, final_output = self.compute_loss(outputs, labels, mask)
        
        # Compute metrics
        map_score = self.compute_metrics(final_output, labels, mask, self.train_apm, 'train')
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_map', map_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        inputs, mask, mask_new, labels, other = self.prepare_data_batch(batch)
        
        # Forward pass
        outputs = self(inputs, mask_new)
        
        # Compute loss
        loss, final_output = self.compute_loss(outputs, labels, mask)
        
        # Compute metrics
        map_score = self.compute_metrics(final_output, labels, mask, self.val_apm, 'val')
        
        # Store predictions
        probs = torch.sigmoid(final_output) * mask.unsqueeze(2)
        video_id = other[0][0]
        self.val_predictions[video_id] = probs.squeeze().detach().cpu().numpy().T
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_map', map_score, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        inputs, mask, mask_new, labels, other = self.prepare_data_batch(batch)
        
        # Forward pass
        outputs = self(inputs, mask_new)
        
        # Compute loss
        loss, final_output = self.compute_loss(outputs, labels, mask)
        
        # Compute metrics
        map_score = self.compute_metrics(final_output, labels, mask, self.test_apm, 'test')
        
        # Store predictions
        probs = torch.sigmoid(final_output) * mask.unsqueeze(2)
        video_id = other[0][0]
        self.test_predictions[video_id] = probs.squeeze().detach().cpu().numpy().T
        
        # Logging
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_map', map_score, on_step=False, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Calculate final mAP
        if self.ap_type == 'wap':
            train_map = 100 * self.train_apm.value()
        else:
            ap_values = 100 * self.train_apm.value()
            valid_aps = ap_values[ap_values > 0]
            train_map = valid_aps.mean() if len(valid_aps) > 0 else 0.0
        
        self.log('train_map_epoch', train_map, on_epoch=True)
        
        # Reset AP meter
        self.train_apm.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Calculate final mAP
        if self.ap_type == 'wap':
            val_map = 100 * self.val_apm.value()
        else:
            ap_values = 100 * self.val_apm.value()
            valid_aps = ap_values[ap_values > 0]
            val_map = valid_aps.mean() if len(valid_aps) > 0 else 0.0
        
        self.log('val_map_epoch', val_map, on_epoch=True)
        
        # Update best validation mAP
        if val_map > self.best_val_map:
            self.best_val_map = val_map
        
        self.log('best_val_map', self.best_val_map, on_epoch=True)
        
        # Reset AP meter
        self.val_apm.reset()
        
        # Clear predictions to save memory
        self.val_predictions.clear()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Calculate final mAP
        if self.ap_type == 'wap':
            test_map = 100 * self.test_apm.value()
        else:
            ap_values = 100 * self.test_apm.value()
            valid_aps = ap_values[ap_values > 0]
            test_map = valid_aps.mean() if len(valid_aps) > 0 else 0.0
        
        self.log('test_map_epoch', test_map, on_epoch=True)
        
        # Reset AP meter
        self.test_apm.reset()
        
        # Save test predictions
        if self.test_predictions:
            results_path = Path(self.trainer.default_root_dir) / 'test_predictions.json'
            with open(results_path, 'w') as f:
                json.dump({k: v.tolist() for k, v in self.test_predictions.items()}, f, indent=2)
            print(f"Test predictions saved to: {results_path}")
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # Scheduler
        if self.scheduler_name.lower() == 'cosine':
            # Cosine annealing with warmup
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return epoch / self.warmup_epochs
                else:
                    return 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / 
                                            (self.max_epochs - self.warmup_epochs)))
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif self.scheduler_name.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=8,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_map_epoch',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        else:
            return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Configure gradient clipping."""
        if self.grad_clip > 0:
            self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PDAN Training with PyTorch Lightning')
    
    # Model parameters
    parser.add_argument('--num_stages', type=int, default=1, help='Number of refinement stages')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--num_channels', type=int, default=512, help='Number of feature channels')
    parser.add_argument('--input_channels', type=int, default=1024, help='Input feature dimension')
    parser.add_argument('--num_classes', type=int, default=157, help='Number of classes')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'], help='Scheduler')
    
    # Data parameters
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'flow', 'skeleton'], help='Input modality')
    parser.add_argument('--rgb_root', type=str, default='/Path/to/charades_feat_rgb', help='RGB features path')
    parser.add_argument('--flow_root', type=str, default='/Path/to/charades_feat_flow', help='Flow features path')
    parser.add_argument('--skeleton_root', type=str, default='/Path/to/charades_feat_pose', help='Skeleton features path')
    parser.add_argument('--train_split', type=str, default='./data/charades.json', help='Training split file')
    parser.add_argument('--val_split', type=str, default='./data/charades.json', help='Validation split file')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    
    # System parameters
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['16-mixed', '32', 'bf16-mixed'], help='Precision')
    parser.add_argument('--strategy', type=str, default='auto', help='Training strategy')
    
    # Logging and checkpointing
    parser.add_argument('--project_name', type=str, default='PDAN', help='Project name for logging')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='./lightning_logs', help='Directory to save logs and checkpoints')
    parser.add_argument('--logger', type=str, default='tensorboard', choices=['tensorboard', 'wandb', 'both'], help='Logger to use')
    parser.add_argument('--wandb_project', type=str, default='pdan-action-detection', help='Wandb project name')
    
    # Callbacks
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save_top_k', type=int, default=3, help='Save top k checkpoints')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Validation frequency')
    
    # Evaluation
    parser.add_argument('--test_only', type=str2bool, default=False, help='Only run testing')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path for testing')
    parser.add_argument('--ap_type', type=str, default='wap', choices=['wap', 'map'], help='AP calculation type')
    
    # Debug
    parser.add_argument('--fast_dev_run', type=str2bool, default=False, help='Fast development run')
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='Limit training batches')
    parser.add_argument('--limit_val_batches', type=float, default=1.0, help='Limit validation batches')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = get_args()
    
    # Set random seed
    pl.seed_everything(42, workers=True)
    
    # Initialize data module
    data_module = PDANDataModule(
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        rgb_root=args.rgb_root,
        flow_root=args.flow_root,
        skeleton_root=args.skeleton_root,
        num_classes=args.num_classes
    )
    
    # Adjust input channels for skeleton mode
    if args.mode == 'skeleton':
        args.input_channels = 256
    
    # Initialize model
    model = PDANLightningModule(
        num_stages=args.num_stages,
        num_layers=args.num_layers,
        num_channels=args.num_channels,
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        ap_type=args.ap_type,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        grad_clip=args.grad_clip
    )
    
    # Initialize loggers
    loggers = []
    if args.logger in ['tensorboard', 'both']:
        tb_logger = TensorBoardLogger(
            save_dir=args.save_dir,
            name=args.project_name,
            version=args.exp_name
        )
        loggers.append(tb_logger)
    
    if args.logger in ['wandb', 'both']:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.exp_name,
            save_dir=args.save_dir
        )
        loggers.append(wandb_logger)
    
    # Initialize callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_map_epoch',
        mode='max',
        save_top_k=args.save_top_k,
        filename='pdan-{epoch:02d}-{val_map_epoch:.2f}',
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_map_epoch',
        mode='max',
        patience=args.early_stopping_patience,
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        num_nodes=args.nodes,
        precision=args.precision,
        strategy=args.strategy,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        deterministic=True,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None
    )
    
    # Run training or testing
    if args.test_only:
        if args.ckpt_path is None:
            raise ValueError("Checkpoint path must be provided for testing")
        trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, data_module)
        
        # Test with best checkpoint
        trainer.test(model, data_module, ckpt_path='best')
    
    print("Training/Testing completed!")


if __name__ == '__main__':
    main()
