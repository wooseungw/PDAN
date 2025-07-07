#!/usr/bin/env python3
"""
Advanced PDAN Training Script with Hydra Configuration
Modern PyTorch Lightning implementation with configuration management
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Import our modules
from train_pdan_lightning import PDANLightningModule, PDANDataModule


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
    # Print configuration
    print("=" * 80)
    print("PDAN Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set random seed
    pl.seed_everything(42, workers=True)
    
    # Initialize data module
    data_module = PDANDataModule(
        mode=cfg.data.mode,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        rgb_root=cfg.data.rgb_root,
        flow_root=cfg.data.flow_root,
        skeleton_root=cfg.data.skeleton_root,
        num_classes=cfg.model.num_classes
    )
    
    # Adjust input channels for skeleton mode
    input_channels = cfg.model.input_channels
    if cfg.data.mode == 'skeleton':
        input_channels = 256
    
    # Initialize model
    model = PDANLightningModule(
        num_stages=cfg.model.num_stages,
        num_layers=cfg.model.num_layers,
        num_channels=cfg.model.num_channels,
        input_channels=input_channels,
        num_classes=cfg.model.num_classes,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        ap_type=cfg.training.ap_type,
        optimizer=cfg.training.optimizer,
        scheduler=cfg.training.scheduler,
        grad_clip=cfg.training.grad_clip
    )
    
    # Initialize loggers
    loggers = []
    if cfg.logging.logger in ['tensorboard', 'both']:
        tb_logger = TensorBoardLogger(
            save_dir=cfg.logging.save_dir,
            name=cfg.logging.project_name,
            version=cfg.logging.exp_name
        )
        loggers.append(tb_logger)
    
    if cfg.logging.logger in ['wandb', 'both']:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.exp_name,
            save_dir=cfg.logging.save_dir,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        loggers.append(wandb_logger)
    
    # Initialize callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_map_epoch',
        mode='max',
        save_top_k=cfg.callbacks.save_top_k,
        filename='pdan-{epoch:02d}-{val_map_epoch:.2f}',
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_map_epoch',
        mode='max',
        patience=cfg.callbacks.early_stopping_patience,
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        devices=cfg.system.gpus,
        num_nodes=cfg.system.nodes,
        precision=cfg.system.precision,
        strategy=cfg.system.strategy,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.callbacks.check_val_every_n_epoch,
        fast_dev_run=cfg.debug.fast_dev_run,
        limit_train_batches=cfg.debug.limit_train_batches,
        limit_val_batches=cfg.debug.limit_val_batches,
        deterministic=True,
        gradient_clip_val=cfg.training.grad_clip if cfg.training.grad_clip > 0 else None
    )
    
    # Run training
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Test with best checkpoint
    print("Starting testing with best checkpoint...")
    trainer.test(model, data_module, ckpt_path='best')
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
