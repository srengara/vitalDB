import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import yaml
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from data_loader import MultiChannelDataLoader
from model import build_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for multi-channel early fusion models.
    """
    
    def __init__(self, config: dict, device: str = 'cuda'):
        """
        Args:
            config: Configuration dictionary
            device: Device to use (cuda or cpu)
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.log_dir = Path(config['logging']['log_dir'])
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        
        logger.info(f"Using device: {self.device}")
    
    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Build optimizer based on config"""
        optimizer_type = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        l2_reg = self.config['training']['l2_regularization']
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_reg)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        logger.info(f"Built optimizer: {optimizer_type} with lr={lr}")
        return optimizer
    
    def build_loss_function(self) -> nn.Module:
        """Build loss function based on config"""
        loss_type = self.config['training']['loss_function']
        
        if loss_type == 'mse':
            loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            loss_fn = nn.L1Loss()
        elif loss_type == 'binary_crossentropy':
            loss_fn = nn.BCELoss()
        elif loss_type == 'categorical_crossentropy':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
        
        logger.info(f"Built loss function: {loss_type}")
        return loss_fn
    
    def build_scheduler(self, optimizer: optim.Optimizer):
        """Build learning rate scheduler"""
        if not self.config['training']['use_scheduler']:
            return None
        
        scheduler_type = self.config['training']['scheduler_type']
        params = self.config['training']['scheduler_params']
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=params['factor'],
                patience=params['patience'],
                min_lr=params['min_lr']
            )
        elif scheduler_type == 'exponential':
            scheduler = ExponentialLR(optimizer, gamma=params.get('gamma', 0.95))
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config['training']['epochs'])
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        logger.info(f"Built scheduler: {scheduler_type}")
        return scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
        """
        Train for one epoch.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)
            
            # Reshape outputs to match targets
            outputs = outputs.view(-1)
            y = y.view(-1)

            loss = loss_fn(outputs, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {self.epoch} [{batch_idx + 1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, model: nn.Module, val_loader: DataLoader, 
                loss_fn: nn.Module) -> Tuple[float, Dict]:
        """
        Validate model.
        
        Args:
            model: Neural network model
            val_loader: Validation data loader
            loss_fn: Loss function
            
        Returns:
            Average validation loss, metrics dictionary
        """
        model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = model(x)
                outputs = outputs.view(-1)
                y = y.view(-1)

                loss = loss_fn(outputs, y)
                total_loss += loss.item()

                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Calculate additional metrics
        all_outputs = np.concatenate([o.reshape(-1) for o in all_outputs])
        all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
        
        mae = np.mean(np.abs(all_outputs - all_targets))
        
        metrics = {
            'val_loss': avg_loss,
            'val_mae': mae
        }
        
        return avg_loss, metrics
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }
        
        filename = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            filename = "best_model.pt"
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
             val_loader: DataLoader, test_loader: DataLoader):
        """
        Complete training loop.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
        """
        model = model.to(self.device)
        optimizer = self.build_optimizer(model)
        loss_fn = self.build_loss_function()
        scheduler = self.build_scheduler(optimizer)
        
        epochs = self.config['training']['epochs']
        use_early_stopping = self.config['training']['use_early_stopping']
        early_stopping_patience = self.config['training']['early_stopping_patience']
        
        logger.info("=" * 50)
        logger.info("Starting Training")
        logger.info("=" * 50)
        
        for epoch in range(epochs):
            self.epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, loss_fn)
            
            # Validate
            if (epoch + 1) % self.config['validation']['validation_frequency'] == 0:
                val_loss, metrics = self.validate(model, val_loader, loss_fn)
                
                logger.info(f"\nEpoch {self.epoch}/{epochs}")
                logger.info(f"Train Loss: {train_loss:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}")
                logger.info(f"Val MAE: {metrics['val_mae']:.4f}\n")
                
                # Learning rate scheduling
                if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif scheduler is not None:
                    scheduler.step()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if self.config['logging']['save_best_model']:
                        self.save_checkpoint(model, optimizer, self.epoch, is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if use_early_stopping and self.patience_counter >= early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered after {self.epoch} epochs")
                    break
                
                # Save periodic checkpoints
                if self.epoch % self.config['logging']['save_frequency'] == 0:
                    self.save_checkpoint(model, optimizer, self.epoch, is_best=False)
        
        logger.info("\n" + "=" * 50)
        logger.info("Training Complete")
        logger.info("=" * 50)
        
        # Test
        self.test(model, test_loader, loss_fn)
    
    def test(self, model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module):
        """
        Test model on test set.
        
        Args:
            model: Neural network model
            test_loader: Test data loader
            loss_fn: Loss function
        """
        model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        logger.info("\nEvaluating on test set...")
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = model(x)
                outputs = outputs.view(-1)
                y = y.view(-1)

                loss = loss_fn(outputs, y)
                total_loss += loss.item()

                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / len(test_loader)

        all_outputs = np.concatenate([o.reshape(-1) for o in all_outputs])
        all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
        
        mae = np.mean(np.abs(all_outputs - all_targets))
        mse = np.mean((all_outputs - all_targets) ** 2)
        
        logger.info(f"\nTest Results:")
        logger.info(f"Test Loss: {avg_loss:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        logger.info(f"Test MSE: {mse:.4f}")


def main():
    """Main training script"""
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    np.random.seed(config['dataset']['random_seed'])
    torch.manual_seed(config['dataset']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['dataset']['random_seed'])
    
    logger.info("Configuration loaded")
    logger.info(f"Model type: {config['model']['type']}")
    logger.info(f"Channels: {config['dataset']['channels']}")
    
    # Load data
    data_loader = MultiChannelDataLoader(
        config['dataset']['csv_path'],
        config
    )
    
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    logger.info("Data loaded successfully")
    
    # Build model
    model = build_model(config)
    logger.info(f"\nModel Architecture:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}\n")
    
    # Train
    device = config['hardware']['device']
    trainer = Trainer(config, device=device)
    trainer.train(model, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()
