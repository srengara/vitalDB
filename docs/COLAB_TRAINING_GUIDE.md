# Training on Google Colab - Complete Guide

## Why Use Google Colab?

- ✅ **Free GPU access** (Tesla T4 or better)
- ✅ **Faster training** (10-50x faster than CPU)
- ✅ **No local setup needed**
- ✅ **Pre-installed libraries** (PyTorch, pandas, etc.)

## Step-by-Step Guide

### Option 1: Upload Directly to Colab (Simple, for small datasets)

#### Step 1: Create a Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **"New Notebook"**
3. Save it as `VitalDB_Glucose_Training.ipynb`

#### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: `GPU`
3. Click **Save**

#### Step 3: Upload Your Code and Data

Add this cell to your notebook:

```python
# Cell 1: Upload training data and code
from google.colab import files
import zipfile
import os

# Create directories
!mkdir -p /content/vitalDB
!mkdir -p /content/vitalDB/training_data_combined

# Upload your files
print("Please upload the following files:")
print("1. training_data_combined.zip (containing ppg_windows.csv and glucose_labels.csv)")
print("2. vitalDB_code.zip (containing src/ folder with all Python files)")

uploaded = files.upload()

# Extract uploaded files
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/vitalDB')
        print(f"Extracted: {filename}")
```

**Before running this cell, prepare on your local machine:**

```bash
# On your local machine (Windows PowerShell):

# 1. Zip your training data
cd C:\IITM\vitalDB
Compress-Archive -Path .\training_data_combined\* -DestinationPath training_data_combined.zip

# 2. Zip your source code
Compress-Archive -Path .\src\* -DestinationPath vitalDB_code.zip
```

#### Step 4: Install Dependencies and Run Training

```python
# Cell 2: Install dependencies
!pip install torch torchvision scikit-learn tensorboard matplotlib

# Cell 3: Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

```python
# Cell 4: Import training module
import sys
sys.path.insert(0, '/content/vitalDB')

from src.training.train_glucose_predictor import train, parse_args
```

```python
# Cell 5: Run training
import argparse

# Create arguments
args = argparse.Namespace(
    data_dir='/content/vitalDB/training_data_combined',
    output_dir='/content/vitalDB/output',
    train_split=0.7,
    val_split=0.15,
    epochs=100,
    batch_size=32,
    lr=0.001,
    weight_decay=1e-4,
    optimizer='adam',
    scheduler='plateau',
    scheduler_step=30,
    scheduler_gamma=0.1,
    early_stopping=20,
    save_freq=10,
    plot_freq=10,
    num_workers=2,
    seed=42
)

# Run training
train(args)
```

```python
# Cell 6: Download trained model
from google.colab import files

# Download best model
files.download('/content/vitalDB/output/best_model.pth')

# Download training history
files.download('/content/vitalDB/output/training_history.csv')
```

---

### Option 2: Use Google Drive (Recommended for larger datasets)

This method is better because:
- No need to re-upload data each time
- Data persists across sessions
- Faster access

#### Step 1: Upload Data to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder: `VitalDB_Training`
3. Upload these files/folders:
   - `training_data_combined/` (folder with CSV files)
   - `src/` (folder with all Python code)

#### Step 2: Create Colab Notebook

Create a new notebook with these cells:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify your files are there
!ls /content/drive/MyDrive/VitalDB_Training
```

```python
# Cell 2: Enable GPU and install dependencies
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

!pip install tensorboard scikit-learn
```

```python
# Cell 3: Set up paths
import sys
import os

# Add source code to Python path
sys.path.insert(0, '/content/drive/MyDrive/VitalDB_Training')

# Define paths
DATA_DIR = '/content/drive/MyDrive/VitalDB_Training/training_data_combined'
OUTPUT_DIR = '/content/drive/MyDrive/VitalDB_Training/output'
CODE_DIR = '/content/drive/MyDrive/VitalDB_Training/src'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Code directory: {CODE_DIR}")
```

```python
# Cell 4: Import training module
from src.training.train_glucose_predictor import train
import argparse
```

```python
# Cell 5: Configure and run training
args = argparse.Namespace(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,

    # Data split
    train_split=0.7,
    val_split=0.15,

    # Training hyperparameters
    epochs=100,
    batch_size=32,
    lr=0.001,
    weight_decay=1e-4,
    optimizer='adam',

    # Learning rate scheduler
    scheduler='plateau',
    scheduler_step=30,
    scheduler_gamma=0.1,

    # Regularization and early stopping
    early_stopping=20,

    # Logging
    save_freq=10,
    plot_freq=10,

    # System
    num_workers=2,
    seed=42
)

# Run training
print("Starting training...")
train(args)
```

```python
# Cell 6: Monitor training with TensorBoard (optional)
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/VitalDB_Training/output/tensorboard
```

```python
# Cell 7: Evaluate trained model
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
history = pd.read_csv(f'{OUTPUT_DIR}/training_history.csv')

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
axes[0, 0].plot(history['train_loss'], label='Train Loss')
axes[0, 0].plot(history['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# MAE curve
axes[0, 1].plot(history['val_mae'], label='Val MAE', color='orange')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE (mg/dL)')
axes[0, 1].set_title('Validation MAE')
axes[0, 1].legend()
axes[0, 1].grid(True)

# RMSE curve
axes[1, 0].plot(history['val_rmse'], label='Val RMSE', color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('RMSE (mg/dL)')
axes[1, 0].set_title('Validation RMSE')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Learning rate
axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='red')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=150)
plt.show()

# Print best metrics
best_epoch = history['val_mae'].idxmin()
print(f"\nBest Model Performance:")
print(f"  Epoch: {best_epoch + 1}")
print(f"  Val MAE: {history.loc[best_epoch, 'val_mae']:.2f} mg/dL")
print(f"  Val RMSE: {history.loc[best_epoch, 'val_rmse']:.2f} mg/dL")
print(f"  Val Loss: {history.loc[best_epoch, 'val_loss']:.4f}")
```

---

### Option 3: Clone from GitHub (Best for collaboration)

If you push your code to GitHub, you can clone it directly:

```python
# Cell 1: Clone repository
!git clone https://github.com/yourusername/vitalDB.git
%cd vitalDB
```

```python
# Cell 2: Upload just the training data
from google.colab import files
uploaded = files.upload()  # Upload training_data_combined.zip
!unzip training_data_combined.zip -d ./training_data_combined
```

```python
# Cell 3: Install and run
!pip install -r requirements.txt  # If you have a requirements.txt

import argparse
from src.training.train_glucose_predictor import train

args = argparse.Namespace(
    data_dir='./training_data_combined',
    output_dir='./output',
    train_split=0.7,
    val_split=0.15,
    epochs=100,
    batch_size=32,
    lr=0.001,
    weight_decay=1e-4,
    optimizer='adam',
    scheduler='plateau',
    scheduler_step=30,
    scheduler_gamma=0.1,
    early_stopping=20,
    save_freq=10,
    plot_freq=10,
    num_workers=2,
    seed=42
)

train(args)
```

---

## Complete Ready-to-Use Colab Notebook

Here's a complete notebook you can copy-paste:

```python
# ============================================================================
# VitalDB Glucose Prediction Training - Google Colab
# ============================================================================

# ---- CELL 1: Setup ----
from google.colab import drive
import torch
import os

# Mount Google Drive
drive.mount('/content/drive')

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ---- CELL 2: Install Dependencies ----
!pip install -q tensorboard scikit-learn matplotlib

# ---- CELL 3: Set Paths ----
import sys

# TODO: Update these paths to match your Google Drive structure
DRIVE_ROOT = '/content/drive/MyDrive/VitalDB_Training'
DATA_DIR = f'{DRIVE_ROOT}/training_data_combined'
OUTPUT_DIR = f'{DRIVE_ROOT}/output'

# Add code to path
sys.path.insert(0, DRIVE_ROOT)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✓ Data directory: {DATA_DIR}")
print(f"✓ Output directory: {OUTPUT_DIR}")

# Verify files exist
!ls {DATA_DIR}

# ---- CELL 4: Import and Configure ----
from src.training.train_glucose_predictor import train
import argparse

# Training configuration
args = argparse.Namespace(
    # Data
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,

    # Data split
    train_split=0.7,
    val_split=0.15,

    # Training hyperparameters
    epochs=100,
    batch_size=32,
    lr=0.001,
    weight_decay=1e-4,

    # Optimization
    optimizer='adam',
    scheduler='plateau',
    scheduler_step=30,
    scheduler_gamma=0.1,

    # Regularization
    early_stopping=20,

    # Saving and logging
    save_freq=10,
    plot_freq=10,

    # System
    num_workers=2,
    seed=42
)

print("✓ Configuration ready")

# ---- CELL 5: Train Model ----
print("=" * 70)
print("Starting Training on GPU")
print("=" * 70)

train(args)

print("\n✓ Training complete!")

# ---- CELL 6: Visualize Results ----
import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv(f'{OUTPUT_DIR}/training_history.csv')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Progress - Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MAE
axes[0, 1].plot(history['val_mae'], color='orange', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE (mg/dL)')
axes[0, 1].set_title('Validation MAE')
axes[0, 1].grid(True, alpha=0.3)

# RMSE
axes[1, 0].plot(history['val_rmse'], color='green', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('RMSE (mg/dL)')
axes[1, 0].set_title('Validation RMSE')
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(history['learning_rate'], color='red', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Print best results
best_epoch = history['val_mae'].idxmin()
print(f"\n{'='*70}")
print(f"BEST MODEL PERFORMANCE")
print(f"{'='*70}")
print(f"Epoch: {best_epoch + 1}/{len(history)}")
print(f"Val MAE:  {history.loc[best_epoch, 'val_mae']:.2f} mg/dL")
print(f"Val RMSE: {history.loc[best_epoch, 'val_rmse']:.2f} mg/dL")
print(f"Val Loss: {history.loc[best_epoch, 'val_loss']:.4f}")
print(f"{'='*70}")

# ---- CELL 7: Download Trained Model (Optional) ----
from google.colab import files

print("Downloading trained model...")
files.download(f'{OUTPUT_DIR}/best_model.pth')
files.download(f'{OUTPUT_DIR}/training_history.csv')
files.download(f'{OUTPUT_DIR}/training_curves.png')
print("✓ Downloads complete!")
```

---

## Troubleshooting

### Issue: "No module named 'src'"

**Solution**: Make sure you've added the correct path to `sys.path`:

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/VitalDB_Training')
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size:

```python
args.batch_size = 16  # Instead of 32
```

### Issue: "Files not found"

**Solution**: Verify Google Drive mount and paths:

```python
!ls /content/drive/MyDrive/
!ls /content/drive/MyDrive/VitalDB_Training/
```

### Issue: Colab disconnects during training

**Solution**: Keep the tab open and run this cell to prevent disconnection:

```python
# Keep Colab alive
import IPython
from google.colab import output

display(IPython.display.Javascript('''
 function ClickConnect(){
   btn = document.querySelector("colab-connect-button")
   if (btn != null){
     console.log("Click colab-connect-button");
     btn.click()
     }

   btn = document.getElementById('ok')
   if (btn != null){
     console.log("Click reconnect");
     btn.click()
     }
  }

setInterval(ClickConnect,60000)
'''))
```

---

## Expected Training Time

With GPU (Tesla T4):
- **Small dataset** (1,000 windows): ~5-10 minutes
- **Medium dataset** (5,000 windows): ~15-30 minutes
- **Large dataset** (20,000 windows): ~1-2 hours

Without GPU (CPU only):
- 10-50x slower

---

## Download Your Trained Model

After training completes, your model will be saved in Google Drive at:
```
/content/drive/MyDrive/VitalDB_Training/output/best_model.pth
```

You can access it anytime from Google Drive or download it to your local machine!
