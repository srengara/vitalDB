# Training Quick Start Guide

## Prerequisites

1. **Data Format**: Your training data must be in this format:

```
training_data/
├── ppg_windows.csv          # PPG signals
└── glucose_labels.csv       # Glucose labels
```

**ppg_windows.csv**:
```
window_index,sample_index,amplitude
0,0,100.5
0,1,101.2
0,2,102.0
...
```

**glucose_labels.csv**:
```
window_index,glucose_mg_dl
0,120.5
1,115.3
2,130.2
...
```

2. **Environment Setup**:

```bash
# Install dependencies
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib tensorboard

# Verify GPU (if available)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Basic Training Command

```bash
python train_glucose_predictor.py \
  --data_dir ./training_data \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001
```

## Training with Custom Parameters

```bash
python train_glucose_predictor.py \
  --data_dir ./training_data \
  --output_dir ./my_training \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --optimizer adam \
  --scheduler step \
  --train_split 0.7 \
  --val_split 0.15 \
  --early_stopping 20 \
  --save_freq 10
```

## All Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Directory with ppg_windows.csv and glucose_labels.csv |
| `--output_dir` | ./training_outputs | Where to save checkpoints |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--lr` | 0.001 | Learning rate |
| `--optimizer` | adam | Optimizer (adam/sgd) |
| `--weight_decay` | 1e-4 | Weight decay (L2 regularization) |
| `--scheduler` | step | LR scheduler (step/cosine/plateau/none) |
| `--scheduler_step` | 30 | StepLR: reduce LR every N epochs |
| `--scheduler_gamma` | 0.1 | StepLR: multiply LR by this factor |
| `--train_split` | 0.7 | Fraction of data for training |
| `--val_split` | 0.15 | Fraction of data for validation |
| `--save_freq` | 10 | Save checkpoint every N epochs |
| `--plot_freq` | 10 | Plot predictions every N epochs |
| `--early_stopping` | 20 | Stop if no improvement for N epochs |
| `--seed` | 42 | Random seed for reproducibility |
| `--num_workers` | 4 | Number of data loader workers |

## Monitoring Training

### Option 1: Watch Console Output

Training prints progress every 50 batches:
```
Epoch 1/100
--------------------------------------------------------------------------------
  Batch [50/312] - Loss: 0.3452
  Batch [100/312] - Loss: 0.3201
  ...
  Train Loss: 0.3156
  Val Loss: 0.3089
  Val MAE: 12.45 mg/dL
  Val RMSE: 16.32 mg/dL
```

### Option 2: TensorBoard

```bash
# In another terminal
tensorboard --logdir=./training_outputs --port=6006

# Open browser to: http://localhost:6006
```

TensorBoard shows:
- Training/validation loss curves
- MAE and RMSE over time
- Learning rate schedule

## Output Files

After training, you'll find:

```
training_outputs/
└── training_20241125_123456/
    ├── config.json                      # Training configuration
    ├── training_history.csv            # Loss/metrics per epoch
    ├── final_metrics.json              # Final test set results
    ├── best_model.pth                  # Best model checkpoint
    ├── checkpoint_epoch_10.pth         # Periodic checkpoints
    ├── checkpoint_epoch_20.pth
    ├── predictions_epoch_10.png        # Prediction plots
    ├── predictions_test.png            # Final test predictions
    └── tensorboard/                    # TensorBoard logs
```

## Using the Trained Model

```python
import torch
from resnet34_glucose_predictor import ResNet34_1D, GlucosePredictor

# Load checkpoint
checkpoint = torch.load('training_outputs/.../best_model.pth')

# Create model
model = ResNet34_1D(input_channels=1, num_classes=1)
model.load_state_dict(checkpoint['model_state_dict'])

# Create predictor
predictor = GlucosePredictor(model=model, input_length=500)

# Predict on new data
import numpy as np
new_windows = np.random.randn(10, 500)  # 10 windows
predictions = predictor.predict(new_windows)

print(f"Glucose predictions: {predictions}")
```

## Common Issues

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size
```bash
python train_glucose_predictor.py --batch_size 16
```

### Issue: Training too slow

**Solution 1**: Increase batch size (if GPU has memory)
```bash
python train_glucose_predictor.py --batch_size 64
```

**Solution 2**: Use more data workers
```bash
python train_glucose_predictor.py --num_workers 8
```

### Issue: Model not converging

**Solution 1**: Lower learning rate
```bash
python train_glucose_predictor.py --lr 0.0001
```

**Solution 2**: Try different optimizer
```bash
python train_glucose_predictor.py --optimizer sgd --lr 0.01
```

### Issue: Overfitting (train loss << val loss)

**Solution 1**: Increase weight decay
```bash
python train_glucose_predictor.py --weight_decay 1e-3
```

**Solution 2**: Use early stopping
```bash
python train_glucose_predictor.py --early_stopping 10
```

### Issue: Underfitting (both losses high)

**Solution 1**: Increase learning rate
```bash
python train_glucose_predictor.py --lr 0.01
```

**Solution 2**: Train for more epochs
```bash
python train_glucose_predictor.py --epochs 200
```

## Example: Complete Training Run

```bash
# 1. Prepare data
cd /c/IITM/vitalDB
mkdir training_data

# Copy your ppg_windows.csv and glucose_labels.csv to training_data/

# 2. Start training
python train_glucose_predictor.py \
  --data_dir ./training_data \
  --output_dir ./my_models \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --early_stopping 20

# 3. Monitor with TensorBoard (optional, in another terminal)
tensorboard --logdir=./my_models

# 4. After training completes, check results
cat my_models/training_*/final_metrics.json

# 5. Use the trained model
python glucose_from_csv.py \
  --model my_models/training_*/best_model.pth \
  filtered_windows.csv
```

## Expected Results

### Good Performance
- **MAE**: < 15 mg/dL
- **RMSE**: < 20 mg/dL
- **Training time**: 2-10 hours (depends on dataset size and GPU)

### Acceptable Performance
- **MAE**: 15-25 mg/dL
- **RMSE**: 20-30 mg/dL

### Poor Performance (Need More Data or Tuning)
- **MAE**: > 25 mg/dL
- **RMSE**: > 30 mg/dL

## Next Steps

1. **Evaluate**: Check `predictions_test.png` to see how well predictions match actual values
2. **Tune**: Adjust hyperparameters based on performance
3. **Deploy**: Integrate trained model into your application
4. **Monitor**: Track performance on new data

## Support

For issues or questions:
- Check training logs in output directory
- Review TensorBoard metrics
- Verify data format matches expected structure
- Try training on a small subset first to debug quickly
