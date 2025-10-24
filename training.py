"""
训练和评估模块
Training and evaluation module for DEKP model

This module provides:
1. Model training with early stopping
2. Model evaluation and metrics calculation
3. Model saving and loading
4. Training visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from dekp_model import DEKPModel, DEKPLoss, create_dekp_model
from data_processing import create_data_loaders, KineticDataProcessor

logger = logging.getLogger(__name__)


class DEKPTrainer:
    """
    DEKP模型训练器
    """
    
    def __init__(self,
                 model: DEKPModel,
                 train_loader,
                 val_loader,
                 test_loader,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Initialize optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize loss function
        self.criterion = DEKPLoss(
            km_weight=config.get('km_weight', 1.0),
            kcat_weight=config.get('kcat_weight', 1.0),
            ki_weight=config.get('ki_weight', 1.0),
            use_log_space=config.get('use_log_space', True)
        )
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Initialize early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 10)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'runs/dekp'))
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """
        设置优化器
        
        Returns:
            Optimizer instance
        """
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        设置学习率调度器
        
        Returns:
            Scheduler instance or None
        """
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('max_epochs', 100)
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self) -> Tuple[float, Dict]:
        """
        训练一个epoch
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                predictions = self.model(
                    batch['protein_sequences'],
                    batch['molecular_graphs'].x,
                    batch['molecular_graphs'].edge_index,
                    batch['molecular_graphs'].batch
                )
                
                # Compute loss
                loss = self.criterion(predictions, batch['kinetic_params'])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                all_predictions.append(predictions.detach().cpu())
                all_targets.append(batch['kinetic_params'].detach().cpu())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate_epoch(self) -> Tuple[float, Dict]:
        """
        验证一个epoch
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                batch = self._move_batch_to_device(batch)
                
                try:
                    predictions = self.model(
                        batch['protein_sequences'],
                        batch['molecular_graphs'].x,
                        batch['molecular_graphs'].edge_index,
                        batch['molecular_graphs'].batch
                    )
                    
                    # Compute loss
                    loss = self.criterion(predictions, batch['kinetic_params'])
                    
                    # Update metrics
                    total_loss += loss.item()
                    all_predictions.append(predictions.cpu())
                    all_targets.append(batch['kinetic_params'].cpu())
                    
                except Exception as e:
                    logger.warning(f"Error in validation batch: {e}")
                    continue
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """
        将批次数据移动到设备
        
        Args:
            batch: Batch data
            
        Returns:
            Batch data on device
        """
        device_batch = {}
        for key, value in batch.items():
            if key == 'protein_sequences':
                device_batch[key] = value
            elif key == 'molecular_graphs' and value is not None:
                device_batch[key] = value.to(self.device)
            elif torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _calculate_metrics(self, predictions: List[torch.Tensor], 
                          targets: List[torch.Tensor]) -> Dict:
        """
        计算评估指标
        
        Args:
            predictions: List of prediction tensors
            targets: List of target tensors
            
        Returns:
            Dictionary of metrics
        """
        # Concatenate all predictions and targets
        all_preds = torch.cat(predictions, dim=0).numpy()
        all_targets = torch.cat(targets, dim=0).numpy()
        
        metrics = {}
        
        # Calculate metrics for each parameter
        param_names = ['Km', 'Kcat', 'Ki']
        
        for i, param_name in enumerate(param_names):
            pred = all_preds[:, i]
            target = all_targets[:, i]
            
            # Calculate metrics
            mse = mean_squared_error(target, pred)
            mae = mean_absolute_error(target, pred)
            r2 = r2_score(target, pred)
            
            # Calculate correlation
            correlation = np.corrcoef(target, pred)[0, 1]
            
            metrics[f'{param_name}_MSE'] = mse
            metrics[f'{param_name}_MAE'] = mae
            metrics[f'{param_name}_R2'] = r2
            metrics[f'{param_name}_Correlation'] = correlation
        
        # Overall metrics
        overall_mse = mean_squared_error(all_targets.flatten(), all_preds.flatten())
        overall_mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
        overall_r2 = r2_score(all_targets.flatten(), all_preds.flatten())
        
        metrics['Overall_MSE'] = overall_mse
        metrics['Overall_MAE'] = overall_mae
        metrics['Overall_R2'] = overall_r2
        
        return metrics
    
    def train(self, max_epochs: int = 100) -> Dict:
        """
        训练模型
        
        Args:
            max_epochs: Maximum number of epochs
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Log to tensorboard
            self._log_to_tensorboard(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val R2: {val_metrics['Overall_R2']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def _log_to_tensorboard(self, epoch: int, train_loss: float, val_loss: float,
                           train_metrics: Dict, val_metrics: Dict):
        """
        记录到TensorBoard
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Log losses
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Log R2 scores
        self.writer.add_scalar('R2/Train', train_metrics['Overall_R2'], epoch)
        self.writer.add_scalar('R2/Validation', val_metrics['Overall_R2'], epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    def evaluate(self) -> Dict:
        """
        在测试集上评估模型
        
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("Evaluating model on test set")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_raw_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = self._move_batch_to_device(batch)
                
                try:
                    predictions = self.model(
                        batch['protein_sequences'],
                        batch['molecular_graphs'].x,
                        batch['molecular_graphs'].edge_index,
                        batch['molecular_graphs'].batch
                    )
                    
                    all_predictions.append(predictions.cpu())
                    all_targets.append(batch['kinetic_params'].cpu())
                    all_raw_targets.append(batch['raw_kinetic_params'].cpu())
                    
                except Exception as e:
                    logger.warning(f"Error in test batch: {e}")
                    continue
        
        # Calculate metrics
        test_metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # Log test metrics
        logger.info("Test Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return test_metrics
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: Path to model file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: str = None):
        """
        绘制训练历史
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # R2 plot
        train_r2 = [m['Overall_R2'] for m in self.train_metrics]
        val_r2 = [m['Overall_R2'] for m in self.val_metrics]
        axes[0, 1].plot(train_r2, label='Train R2')
        axes[0, 1].plot(val_r2, label='Validation R2')
        axes[0, 1].set_title('Training and Validation R2')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R2 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Parameter-specific R2
        param_names = ['Km', 'Kcat', 'Ki']
        for i, param in enumerate(param_names):
            val_param_r2 = [m[f'{param}_R2'] for m in self.val_metrics]
            axes[1, 0].plot(val_param_r2, label=f'{param} R2')
        axes[1, 0].set_title('Parameter-specific R2 (Validation)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R2 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        lr_values = []
        for group in self.optimizer.param_groups:
            lr_values.append(group['lr'])
        axes[1, 1].plot(lr_values)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


def train_dekp_model(data_file: str, config_file: str = None) -> DEKPTrainer:
    """
    训练DEKP模型的便捷函数
    
    Args:
        data_file: Path to training data file
        config_file: Path to configuration file
        
    Returns:
        Trained DEKP trainer
    """
    # Load configuration
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'max_epochs': 100,
            'patience': 10,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'grad_clip': 1.0,
            'log_dir': 'runs/dekp'
        }
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_file,
        batch_size=config['batch_size']
    )
    
    # Create model
    model = create_dekp_model(config)
    
    # Create trainer
    trainer = DEKPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # Train model
    history = trainer.train(max_epochs=config['max_epochs'])
    
    # Evaluate model
    test_metrics = trainer.evaluate()
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    return trainer


if __name__ == "__main__":
    # Example usage
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'max_epochs': 50,
        'patience': 10,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'grad_clip': 1.0,
        'log_dir': 'runs/dekp_example'
    }
    
    # This would be used with actual data
    # trainer = train_dekp_model('enzyme_kinetic_data.csv', config)
    print("DEKP training module loaded successfully!")
