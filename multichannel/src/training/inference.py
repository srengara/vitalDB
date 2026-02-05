import torch
import torch.nn as nn
import yaml
import numpy as np
from pathlib import Path
import logging

from model import build_model
from data_loader import MultiChannelDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """
    Inference class for making predictions with trained model.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, 
                 device: str = 'cuda'):
        """
        Args:
            config_path: Path to config.yaml
            checkpoint_path: Path to best_model.pt
            device: Device to use (cuda or cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = build_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {checkpoint_path}")
        
        # Load data loader for preprocessing
        self.data_loader = MultiChannelDataLoader(
            self.config['dataset']['csv_path'],
            self.config
        )
        # Fit scaler on original data
        self.data_loader.load_data()
        self.data_loader.preprocess_data(self.data_loader.load_data())
    
    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions on a batch of data.
        
        Args:
            data: Shape (batch_size, sequence_length, num_channels)
            
        Returns:
            Predictions array
        """
        data_tensor = torch.from_numpy(data).float().to(self.device)
        
        with torch.no_grad():
            predictions = self.model(data_tensor)
        
        return predictions.cpu().numpy()
    
    def predict_single(self, window: np.ndarray) -> float:
        """
        Make prediction on single window.
        
        Args:
            window: Shape (sequence_length, num_channels)
            
        Returns:
            Single prediction value
        """
        window = np.expand_dims(window, axis=0)  # Add batch dimension
        pred = self.predict_batch(window)
        return pred[0, 0]


def main():
    """Example usage"""
    
    # Initialize inference
    inference = ModelInference(
        config_path='config.yaml',
        checkpoint_path='checkpoints/best_model.pt'
    )
    
    logger.info("Inference ready. Load your data and make predictions!")
    
    # Example: Make prediction on random data
    example_data = np.random.randn(4, 100, 5)  # 4 samples, 100 timesteps, 5 channels
    predictions = inference.predict_batch(example_data)
    logger.info(f"Example predictions shape: {predictions.shape}")
    logger.info(f"Example predictions: {predictions.flatten()}")


if __name__ == '__main__':
    main()
