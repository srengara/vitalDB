# Multi-Channel Early Fusion Model Training Guide

## Overview

This is a complete training setup for multi-channel signals using **early fusion architecture**. Early fusion concatenates all input channels at the beginning and processes them together, allowing the model to learn inter-channel correlations from the start.

### Your Dataset Channels:
- **Pressure_cuff_signal**: Blood pressure measurement
- **Force_pulse**: Force/pulse information
- **Signal_1**: PPG signal (Green)
- **Signal_2**: PPG signal (infrared)
- **Signal_3**: Additional signal (FSR)

---

## File Structure

```
├── config.yaml              # Configuration file
├── data_loader.py          # Data loading and preprocessing
├── model.py                # Model architectures
├── train.py                # Training script
├── README.md               # This file
├── checkpoints/            # Saved model checkpoints
└── logs/                   # Training logs
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn pyyaml
```

---

## Configuration Guide

Edit `config.yaml` to customize training:

### Dataset Configuration
```yaml
dataset:
  csv_path: "your_data.csv"              # Path to CSV file
  test_split: 0.2                        # 20% test data
  validation_split: 0.1                  # 10% validation data
  normalize: true                        # Normalize data
  normalization_method: "minmax"         # or "standard"
  
  channels:                              # Channels to use (order matters!)
    - Pressure_cuff_signal
    - Force_pulse
    - Signal_1
    - Signal_2
    - Signal_3
```

### Preprocessing Configuration
```yaml
preprocessing:
  window_size: 100          # Sequence length (samples per window)
  stride: 50                # Overlap between windows
  remove_outliers: true     # Remove anomalies
  outlier_threshold: 3      # Z-score threshold
```

**Important**: Adjust `window_size` based on your signal:
- Small (`50-100`): Fast dynamics, high-frequency features
- Large (`200-500`): Slow dynamics, contextual patterns
- Medical signals: typically `100-256`

### Model Configuration

#### Architecture Types:

**1. Early Fusion CNN** (Recommended for signals)
```yaml
model:
  type: "early_fusion_cnn"
  input_channels: 5          # 5 channels
  sequence_length: 100       # Match preprocessing window_size
```
**Best for**: Learning local patterns, fast training
**When to use**: Detecting anomalies, classification tasks

**2. Early Fusion RNN (LSTM)**
```yaml
model:
  type: "early_fusion_rnn"
```
**Best for**: Long-term dependencies, temporal patterns
**When to use**: Forecasting, continuous signals

**3. Early Fusion Hybrid (CNN + LSTM)**
```yaml
model:
  type: "early_fusion_hybrid"
```
**Best for**: Combining feature extraction and temporal modeling
**When to use**: Complex signals with spatial-temporal patterns

#### Customize Layers:
```yaml
conv_layers:
  - filters: 64              # Number of filters
    kernel_size: 5           # Filter size
    activation: relu         # Activation function
    dropout: 0.3             # Dropout rate (0-1)
    
dense_layers:
  - units: 256              # Neurons in layer
    activation: relu
    dropout: 0.5
```

**Layer Guidelines**:
- Start with **filters: [64, 128, 256]** for 3 conv layers
- Use **kernel_size: 5** for 1D signals
- Increase **dropout** (0.3-0.5) if overfitting
- Decrease if underfitting

### Output Configuration

Set based on your task:

**Regression** (predict continuous values):
```yaml
output_units: 1
output_activation: linear      # MSE loss
loss_function: mse
```

**Binary Classification** (2 classes):
```yaml
output_units: 1
output_activation: sigmoid
loss_function: binary_crossentropy
```

**Multi-Class** (3+ classes):
```yaml
output_units: num_classes
output_activation: softmax
loss_function: categorical_crossentropy
```

### Training Configuration

```yaml
training:
  batch_size: 32              # 16-64 typical
  epochs: 100                 # Max training iterations
  learning_rate: 0.001        # 1e-3 to 1e-5
  optimizer: adam             # adam, sgd, rmsprop
  loss_function: mse          # See loss options above
```

**Tips**:
- **Batch size**: Larger (64) = faster, noisier; Smaller (16) = slower, more stable
- **Learning rate**: 
  - Too high (0.01): Loss oscillates
  - Too low (0.00001): Training too slow
  - Start with 0.001, adjust if needed
- **Epochs**: Stop early if validation loss plateaus

### Learning Rate Scheduling

```yaml
training:
  use_scheduler: true
  scheduler_type: reduce_on_plateau
  scheduler_params:
    factor: 0.5              # Multiply LR by 0.5
    patience: 10             # Wait 10 epochs before reducing
    min_lr: 0.00001          # Minimum LR
```

### Early Stopping

```yaml
training:
  use_early_stopping: true
  early_stopping_patience: 15  # Stop if no improvement for 15 epochs
```

---

## Running Training

### Basic Training
```bash
python train.py
```

### What Happens:
1. Loads CSV and creates sliding windows
2. Normalizes data
3. Splits into train/val/test
4. Builds model architecture
5. Trains with validation
6. Saves best model to `checkpoints/best_model.pt`
7. Evaluates on test set

### Expected Output:
```
Epoch 1/100 [10/50] Loss: 0.0234
Epoch 1/100
Train Loss: 0.0201
Val Loss: 0.0189
Val MAE: 0.0156

Epoch 2/100
Train Loss: 0.0198
Val Loss: 0.0185
Val MAE: 0.0152
...

Test Results:
Test Loss: 0.0182
Test MAE: 0.0149
Test MSE: 0.0198
```

---

## Common Configuration Changes

### For Your Dataset

**1. If model underfits** (high train & val loss):
```yaml
conv_layers:
  - filters: 128           # Increase from 64
    kernel_size: 5
    dropout: 0.2           # Decrease dropout

dense_layers:
  - units: 512             # Increase from 256
    dropout: 0.3           # Decrease dropout

training:
  epochs: 150              # Train longer
  learning_rate: 0.0005    # Lower LR for stability
```

**2. If model overfits** (low train, high val loss):
```yaml
preprocessing:
  remove_outliers: true

training:
  l2_regularization: 0.01  # Increase from 0.001
  
conv_layers:
  - dropout: 0.5           # Increase from 0.3
  
dense_layers:
  - dropout: 0.6           # Increase from 0.5
```

**3. For faster training**:
```yaml
preprocessing:
  window_size: 50          # Reduce from 100
  stride: 25               # Reduce from 50

training:
  batch_size: 64           # Increase from 32

model:
  type: "early_fusion_cnn"  # Faster than RNN
```

**4. For better accuracy** (slower):
```yaml
preprocessing:
  window_size: 256         # Increase context
  stride: 128

training:
  batch_size: 16           # Smaller batch
  learning_rate: 0.0001    # Lower for stability
  epochs: 200

model:
  type: "early_fusion_hybrid"  # More complex
```

---

## Early Fusion Explained

**What is Early Fusion?**

Early fusion concatenates all channels **before** processing:

```
Input: [Pressure, Force, Signal_1, Signal_2, Signal_3]
           ↓
    Concatenate (5 channels)
           ↓
    Conv layers process all 5 together
           ↓
    Learn inter-channel patterns
```

**Why Early Fusion?**

✓ Learn relationships between channels from start  
✓ Efficient: Single stream of processing  
✓ Good for correlated signals (vital signs)  
✓ Fewer parameters than late fusion

**vs. Late Fusion** (for comparison):

Late fusion processes channels separately then merges:
```
Pressure → Conv → |
Force    → Conv → |→ Merge → Dense → Output
Signal_1 → Conv → |
...
```

---

## Monitoring Training

### Check Logs
```bash
# View training progress
cat logs/training.log
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Loss not decreasing | Increase learning rate to 0.01, reduce batch size |
| Loss oscillating | Decrease learning rate to 0.0001 |
| Out of memory | Reduce batch_size to 16 |
| Val loss > train loss | Increase dropout, increase l2_regularization |
| Training too slow | Reduce window_size, increase batch_size |

---

## Advanced Customization

### Add Data Augmentation

```yaml
training:
  use_data_augmentation: true
  augmentation_params:
    noise_std: 0.05         # Add Gaussian noise
    time_shift: 10          # Shift signal temporally
    amplitude_scale: 0.15   # Scale amplitude
```

### Custom Loss Function

Edit `train.py`, `build_loss_function()` method to add custom losses.

### Multi-GPU Training

Modify `train.py`:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

## Next Steps

1. **Prepare your data**: Ensure CSV format matches
2. **Update config.yaml**: Set your parameters
3. **Run training**: `python train.py`
4. **Monitor results**: Check logs and metrics
5. **Fine-tune**: Adjust config based on results
6. **Deploy**: Load saved model and make predictions

---

## Loading Trained Model

```python
import torch
from model import build_model

# Load config and build model
with open('config.yaml') as f:
    config = yaml.safe_load(f)

model = build_model(config)

# Load weights
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    output = model(input_tensor)
```

---

## References

- **PyTorch**: https://pytorch.org/
- **Early Fusion**: Common in medical signal processing
- **Normalization**: MinMax good for bounded signals, Standard for unbounded

Good luck with your multi-channel model! 🚀
