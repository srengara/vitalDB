#!/usr/bin/env python
"""
Train ResNet34-1D Glucose Prediction Model

Usage:
    python train_model.py --data_dir ./training_data --epochs 100

For all options:
    python train_model.py --help
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run training script
from src.training.train_glucose_predictor import main

if __name__ == '__main__':
    main()
