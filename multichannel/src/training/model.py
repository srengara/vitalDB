import torch
import torch.nn as nn
from typing import List
import logging

logger = logging.getLogger(__name__)


class EarlyFusionCNN(nn.Module):
    """
    Early Fusion CNN Architecture
    
    All input channels are concatenated at the input level and processed together.
    Suitable for learning inter-channel correlations from the beginning.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary with model parameters
        """
        super(EarlyFusionCNN, self).__init__()
        
        self.config = config
        input_channels = config['model']['input_channels']
        output_units = config['model']['output_units']
        
        # Build CNN layers
        self.conv_layers = nn.ModuleList()
        conv_config = config['model']['conv_layers']
        
        in_channels = input_channels
        for i, layer_config in enumerate(conv_config):
            filters = layer_config['filters']
            kernel = layer_config['kernel_size']
            activation = layer_config['activation']
            dropout = layer_config['dropout']
            
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=filters,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        bias=True
                    ),
                    nn.BatchNorm1d(filters),
                    nn.ReLU() if activation == 'relu' else nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = filters
        
        # Calculate flattened size after conv layers
        dummy_input = torch.randn(1, config['model']['input_channels'], 
                                  config['model']['sequence_length'])
        dummy_output = self._forward_conv(dummy_input)
        self.flattened_size = dummy_output.view(1, -1).size(1)
        
        logger.info(f"Flattened size after conv layers: {self.flattened_size}")
        
        # Build dense layers
        self.dense_layers = nn.ModuleList()
        dense_config = config['model']['dense_layers']
        
        prev_units = self.flattened_size
        for layer_config in dense_config:
            units = layer_config['units']
            activation = layer_config['activation']
            dropout = layer_config['dropout']
            
            self.dense_layers.append(
                nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.ReLU() if activation == 'relu' else nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            prev_units = units
        
        # Output layer
        output_activation = config['model']['output_activation']
        if output_activation == 'sigmoid':
            self.output_layer = nn.Sequential(
                nn.Linear(prev_units, output_units),
                nn.Sigmoid()
            )
        elif output_activation == 'softmax':
            self.output_layer = nn.Sequential(
                nn.Linear(prev_units, output_units),
                nn.Softmax(dim=1)
            )
        else:  # linear
            self.output_layer = nn.Linear(prev_units, output_units)
    
    def _forward_conv(self, x):
        """Forward pass through conv layers"""
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, num_channels)
            
        Returns:
            output: Prediction tensor
        """
        # x shape: (batch_size, sequence_length, num_channels)
        # Conv1d expects: (batch_size, num_channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Forward through conv layers
        x = self._forward_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Forward through dense layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        
        # Output
        x = self.output_layer(x)
        
        return x


class EarlyFusionRNN(nn.Module):
    """
    Early Fusion RNN Architecture (LSTM/GRU)
    
    All channels processed together through recurrent layers.
    Better for temporal dependencies.
    """
    
    def __init__(self, config: dict):
        super(EarlyFusionRNN, self).__init__()
        
        self.config = config
        input_channels = config['model']['input_channels']
        output_units = config['model']['output_units']
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Dense layers
        self.dense1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dense2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output
        output_activation = config['model']['output_activation']
        if output_activation == 'sigmoid':
            self.output_layer = nn.Sequential(
                nn.Linear(64, output_units),
                nn.Sigmoid()
            )
        elif output_activation == 'softmax':
            self.output_layer = nn.Sequential(
                nn.Linear(64, output_units),
                nn.Softmax(dim=1)
            )
        else:
            self.output_layer = nn.Linear(64, output_units)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, num_channels)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        x = h_n[-1]
        
        # Dense layers
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        
        return x


class EarlyFusionHybrid(nn.Module):
    """
    Early Fusion Hybrid Architecture
    
    Combines CNN for feature extraction and LSTM for temporal modeling.
    CNN reduces dimensionality, LSTM captures temporal patterns.
    """
    
    def __init__(self, config: dict):
        super(EarlyFusionHybrid, self).__init__()
        
        self.config = config
        input_channels = config['model']['input_channels']
        output_units = config['model']['output_units']
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Fully connected layers
        self.dense1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dense2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output
        output_activation = config['model']['output_activation']
        if output_activation == 'sigmoid':
            self.output_layer = nn.Sequential(
                nn.Linear(64, output_units),
                nn.Sigmoid()
            )
        elif output_activation == 'softmax':
            self.output_layer = nn.Sequential(
                nn.Linear(64, output_units),
                nn.Softmax(dim=1)
            )
        else:
            self.output_layer = nn.Linear(64, output_units)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, num_channels)
        """
        # Transpose for CNN: (batch, channels, sequence)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = self.cnn(x)
        
        # Transpose back for LSTM: (batch, sequence, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        
        # Dense layers
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        
        return x


class ResidualBlock1D(nn.Module):
    """
    Residual Block for 1D Convolutional Networks.
    
    Implements: x + F(x) where F is a sequence of conv layers.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 dropout: float = 0.3):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            dropout: Dropout rate
        """
        super(ResidualBlock1D, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding, 
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu_final = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (batch_size, in_channels, length)
            
        Returns:
            Output tensor (batch_size, out_channels, length)
        """
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Shortcut path
        shortcut = self.shortcut(x)
        
        # Combine and apply final activation
        out = out + shortcut
        out = self.relu_final(out)
        
        return out


class EarlyFusionResNet1D(nn.Module):
    """
    Early Fusion ResNet 1D Architecture
    
    Multi-channel signals are fused at input and processed through
    residual blocks. Allows very deep networks without vanishing gradients.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary with model parameters
        """
        super(EarlyFusionResNet1D, self).__init__()
        
        self.config = config
        input_channels = config['model']['input_channels']
        output_units = config['model']['output_units']
        
        # Initial convolution block
        self.initial_conv = nn.Sequential(
            nn.Conv1d(
                input_channels, 64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        
        # ResNet structure: [64, 128, 256, 512] channels with multiple blocks each
        channel_configs = [
            (64, 64, 2),    # (in_channels, out_channels, num_blocks)
            (64, 128, 2),
            (128, 256, 2),
            (256, 512, 2)
        ]
        
        conv_config = config['model'].get('conv_layers', 
                                         [{'kernel_size': 3, 'dropout': 0.3},
                                          {'kernel_size': 3, 'dropout': 0.3},
                                          {'kernel_size': 3, 'dropout': 0.3},
                                          {'kernel_size': 3, 'dropout': 0.3}])
        
        for stage_idx, (in_ch, out_ch, num_blocks) in enumerate(channel_configs):
            kernel_size = conv_config[stage_idx]['kernel_size']
            dropout = conv_config[stage_idx]['dropout']
            
            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                block = ResidualBlock1D(
                    in_ch if block_idx == 0 else out_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=dropout
                )
                self.residual_blocks.append(block)
        
        # Calculate flattened size
        dummy_input = torch.randn(
            1, config['model']['input_channels'], 
            config['model']['sequence_length']
        )
        dummy_output = self._forward_resnet(dummy_input)
        self.flattened_size = dummy_output.view(1, -1).size(1)
        
        logger.info(f"Flattened size after ResNet blocks: {self.flattened_size}")
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dense_layers = nn.ModuleList()
        dense_config = config['model']['dense_layers']
        
        prev_units = 512  # Last residual block outputs 512 channels
        for layer_config in dense_config:
            units = layer_config['units']
            activation = layer_config['activation']
            dropout = layer_config['dropout']
            
            self.dense_layers.append(
                nn.Sequential(
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units),
                    nn.ReLU(inplace=True) if activation == 'relu' else nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            )
            prev_units = units
        
        # Output layer
        output_activation = config['model']['output_activation']
        if output_activation == 'sigmoid':
            self.output_layer = nn.Sequential(
                nn.Linear(prev_units, output_units),
                nn.Sigmoid()
            )
        elif output_activation == 'softmax':
            self.output_layer = nn.Sequential(
                nn.Linear(prev_units, output_units),
                nn.Softmax(dim=1)
            )
        else:  # linear
            self.output_layer = nn.Linear(prev_units, output_units)
    
    def _forward_resnet(self, x):
        """Forward pass through ResNet blocks"""
        x = self.initial_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        return x
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, num_channels)
            
        Returns:
            output: Prediction tensor
        """
        # x shape: (batch_size, sequence_length, num_channels)
        # Conv1d expects: (batch_size, num_channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Forward through ResNet blocks
        x = self._forward_resnet(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Forward through dense layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        
        # Output
        x = self.output_layer(x)
        
        return x


def build_model(config: dict) -> nn.Module:
    """
    Build model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_type = config['model']['type']
    
    if model_type == 'early_fusion_cnn':
        model = EarlyFusionCNN(config)
    elif model_type == 'early_fusion_rnn':
        model = EarlyFusionRNN(config)
    elif model_type == 'early_fusion_hybrid':
        model = EarlyFusionHybrid(config)
    elif model_type == 'early_fusion_resnet1d':
        model = EarlyFusionResNet1D(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Built model: {model_type}")
    return model
