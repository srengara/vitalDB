# Glucose Prediction Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Complete Pipeline Flow](#complete-pipeline-flow)
3. [Module Architecture](#module-architecture)
4. [Class Hierarchy](#class-hierarchy)
5. [Function Reference](#function-reference)
6. [Data Flow](#data-flow)
7. [Interface Specifications](#interface-specifications)
8. [Neural Network Architecture](#neural-network-architecture)
9. [Step-by-Step Execution](#step-by-step-execution)

---

## Overview

### Purpose
The glucose prediction system uses **deep learning (ResNet34-1D)** to predict blood glucose levels from **PPG (Photoplethysmography) signals** extracted from the VitalDB dataset.

### High-Level Flow
```
Raw PPG Signal → Preprocessing → Peak Detection → Window Extraction →
Window Filtering → ResNet34-1D → Glucose Prediction (mg/dL)
```

### Key Components
1. **PPG Extractor** - Extracts raw PPG data from VitalDB
2. **PPG Segmenter** - Preprocesses signal (filtering, smoothing)
3. **Peak Detector** - Detects heartbeat peaks in PPG signal
4. **Window Extractor** - Extracts signal segments around peaks
5. **Template Filter** - Filters windows by similarity to average template
6. **ResNet34-1D** - Neural network that predicts glucose from windows

---

## Complete Pipeline Flow

### Visual Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GLUCOSE PREDICTION PIPELINE                      │
└─────────────────────────────────────────────────────────────────────┘

[1] VitalDB Database
     │
     │ extract_ppg_raw() / extract_ppg_cleansed()
     ▼
[2] Raw PPG Signal
     │ Time series: [time[], ppg[]]
     │ Example: 15000 samples at 500 Hz = 30 seconds
     │
     │ preprocess_signal()
     ▼
[3] Preprocessed Signal
     │ - DC component removed (mean subtracted)
     │ - Bandpass filtered (0.5-10 Hz)
     │ - Savitzky-Golay smoothed (window=11, order=3)
     │
     │ detect_peaks()
     ▼
[4] Peak Locations
     │ List of indices: [70, 874, 1277, 2078, ...]
     │ Example: 22 peaks detected
     │
     │ extract_windows()
     ▼
[5] Raw Windows
     │ List of signal segments centered on peaks
     │ Each window: ~500 samples (1 second at 500 Hz)
     │ Example: 22 windows extracted
     │
     │ compute_template()
     ▼
[6] Template (Average Beat)
     │ Average of all windows
     │ Shape: (500,) - represents ideal heartbeat
     │
     │ filter_windows_by_similarity()
     ▼
[7] Filtered Windows
     │ Windows with cosine similarity > 0.85 to template
     │ Example: 21 windows (95.5% pass rate)
     │
     │ preprocess_windows()
     ▼
[8] Normalized Tensor
     │ Shape: (21, 1, 500)
     │ - Batch size: 21
     │ - Channels: 1 (single PPG channel)
     │ - Sequence length: 500 samples
     │
     │ ResNet34_1D.forward()
     ▼
[9] Glucose Predictions
     │ Shape: (21, 1)
     │ Values in mg/dL
     │ Example: [120.5, 118.3, 122.1, ...]
     │
     └─> Output: Mean, Std, Min, Max glucose statistics
```

---

## Module Architecture

### File Structure and Dependencies

```
resnet34_glucose_predictor.py
├── ResidualBlock1D (class)
│   └── Neural network building block
├── ResNet34_1D (class)
│   └── Main neural network architecture
└── GlucosePredictor (class)
    └── High-level interface for users

Dependencies:
├── peak_detection.py
│   ├── detect_peaks()
│   ├── extract_windows()
│   ├── compute_template()
│   ├── cosine_similarity()
│   ├── filter_windows_by_similarity()
│   └── ppg_peak_detection_pipeline_with_template()
├── ppg_segmentation.py
│   └── PPGSegmenter.preprocess_signal()
└── ppg_extractor.py
    └── PPGExtractor.extract_ppg_raw()
```

---

## Class Hierarchy

### 1. ResidualBlock1D

**Purpose**: Building block for ResNet architecture with skip connections

**Inheritance**: `nn.Module` (PyTorch)

**Architecture**:
```
Input (in_channels, seq_len)
    │
    ├─────────────────────────┐
    │                         │
    │ [Main Path]             │ [Skip Connection]
    │                         │
    ▼                         │
Conv1D (kernel=3, stride=s)  │
    ▼                         │
BatchNorm1D                   │
    ▼                         │
ReLU                          │
    ▼                         │
Conv1D (kernel=3, stride=1)  │
    ▼                         │
BatchNorm1D                   │
    │                         │
    │                         ▼
    └─────────> ADD <─── Downsample (if needed)
                 ▼
               ReLU
                 ▼
    Output (out_channels, seq_len')
```

**Key Attributes**:
- `conv1`: First 1D convolution layer
- `bn1`: First batch normalization
- `conv2`: Second 1D convolution layer
- `bn2`: Second batch normalization
- `downsample`: Optional 1x1 conv for dimension matching
- `relu`: ReLU activation function

**Methods**:
```python
__init__(in_channels, out_channels, kernel_size=3, stride=1, downsample=None)
forward(x) -> torch.Tensor
```

---

### 2. ResNet34_1D

**Purpose**: Complete 34-layer residual network for glucose prediction

**Inheritance**: `nn.Module` (PyTorch)

**Layer Structure**:
```
Layer Name       | Blocks | Input Ch | Output Ch | Stride | Output Shape (L=100)
-----------------|--------|----------|-----------|--------|---------------------
conv1            | 1      | 1        | 64        | 2      | (N, 64, 50)
maxpool          | -      | 64       | 64        | 2      | (N, 64, 25)
layer1           | 3      | 64       | 64        | 1      | (N, 64, 25)
layer2           | 4      | 64       | 128       | 2      | (N, 128, 13)
layer3           | 6      | 128      | 256       | 2      | (N, 256, 7)
layer4           | 3      | 256      | 512       | 2      | (N, 512, 4)
avgpool          | -      | 512      | 512       | -      | (N, 512, 1)
flatten          | -      | 512      | 512       | -      | (N, 512)
dropout          | -      | 512      | 512       | -      | (N, 512)
fc               | 1      | 512      | 1         | -      | (N, 1)
```

**Key Attributes**:
- `input_length`: Expected sequence length (e.g., 100 samples)
- `num_classes`: Output dimension (1 for glucose)
- `conv1`: Initial 7x1 convolution
- `bn1`: Initial batch normalization
- `layer1, layer2, layer3, layer4`: ResNet layers
- `avgpool`: Adaptive average pooling
- `dropout`: Dropout layer (p=0.5)
- `fc`: Fully connected output layer

**Methods**:
```python
__init__(input_length=100, num_classes=1, dropout_rate=0.5)
_make_layer(in_channels, out_channels, num_blocks, stride) -> nn.Sequential
_initialize_weights() -> None
forward(x) -> torch.Tensor
```

**Parameter Count**: ~7.2 million trainable parameters

---

### 3. GlucosePredictor

**Purpose**: High-level wrapper providing easy-to-use interface

**Key Attributes**:
- `model`: ResNet34_1D instance
- `device`: torch.device (CPU or CUDA)
- `input_length`: Expected window length

**Methods Overview**:

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `__init__()` | Initialize predictor | input_length, device, model_path | GlucosePredictor |
| `preprocess_windows()` | Normalize and pad windows | List[np.ndarray] | torch.Tensor |
| `predict()` | Predict glucose values | List[np.ndarray] | np.ndarray |
| `predict_single()` | Predict single window | np.ndarray | float |
| `predict_with_stats()` | Predict with statistics | List[np.ndarray] | Dict |
| `train_step()` | Single training iteration | windows, targets, optimizer, criterion | float |
| `evaluate()` | Evaluate on test set | windows, targets, criterion | Dict |
| `save_model()` | Save model weights | save_path | None |
| `load_model()` | Load model weights | load_path | None |
| `get_model_summary()` | Get architecture info | - | str |

---

## Function Reference

### A. Peak Detection Functions (from peak_detection.py)

#### 1. `detect_peaks(ppg_signal, height_threshold, distance_threshold, fs)`

**Purpose**: Detect systolic peaks (heartbeats) in PPG signal

**Algorithm**:
```
FOR each sample i in ppg_signal:
    IF ppg_signal[i-1] < ppg_signal[i] > ppg_signal[i+1]:  # Local maximum
        IF ppg_signal[i] > height_threshold:                # Above threshold
            IF (no peaks yet) OR (i - last_peak > distance_threshold):
                ADD i to peaks list
RETURN peaks
```

**Parameters**:
- `ppg_signal` (np.ndarray): Input PPG signal, shape (N,)
- `height_threshold` (float): Minimum peak amplitude (e.g., mean + 0.3*std)
- `distance_threshold` (float): Minimum samples between peaks (e.g., 0.8*fs)
- `fs` (float): Sampling frequency in Hz

**Returns**:
- `peaks` (List[int]): List of peak indices

**Example**:
```python
signal = np.array([10, 20, 30, 25, 15, 10, 25, 35, 30, 20])
peaks = detect_peaks(signal, height_threshold=28, distance_threshold=3, fs=100)
# Returns: [2, 7]  (indices where signal peaks)
```

---

#### 2. `extract_windows(ppg_signal, peaks, window_size, similarity_threshold)`

**Purpose**: Extract signal segments centered on each peak

**Algorithm**:
```
FOR each peak in peaks:
    window_start = max(0, peak - window_size // 3)
    window_end = min(len(signal), peak + window_size * 2 // 3)
    window = ppg_signal[window_start:window_end]

    # Only keep windows with exactly one peak
    IF count_peaks(window, height_threshold) == 1:
        ADD window to windows list

RETURN windows
```

**Parameters**:
- `ppg_signal` (np.ndarray): Preprocessed signal
- `peaks` (List[int]): Peak locations from detect_peaks()
- `window_size` (int): Window length in samples (e.g., fs * 1.0 = 500 for 1 sec)
- `similarity_threshold` (float): Threshold for peak validation (0.85)

**Returns**:
- `windows` (List[np.ndarray]): List of extracted windows

**Window Layout**:
```
Peak at index 1000, window_size=500:

window_start = 1000 - 500/3 = 833
window_end = 1000 + 500*2/3 = 1333

Signal: [... 833 ..................... 1000 (peak) ..................... 1333 ...]
Window:      [━━━━━━━━━━━━━━━━━━━━━━━━━━▲━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
             ← ~167 samples →           ← ~333 samples →
```

---

#### 3. `compute_template(windows)`

**Purpose**: Compute average heartbeat template from all windows

**Algorithm**:
```
# Find most common window length
lengths = [len(w) for w in windows]
most_common_length = mode(lengths)

# Filter to uniform length
uniform_windows = [w for w in windows if len(w) == most_common_length]

# Stack and average
stacked = np.stack(uniform_windows, axis=0)  # Shape: (num_windows, length)
template = np.mean(stacked, axis=0)          # Shape: (length,)

RETURN template
```

**Parameters**:
- `windows` (List[np.ndarray]): Extracted windows from extract_windows()

**Returns**:
- `template` (np.ndarray): Average beat template, shape (window_length,)

**Visual Example**:
```
Window 1: [2, 3, 5, 3, 2]
Window 2: [1, 4, 6, 4, 1]
Window 3: [2, 3, 5, 3, 2]
           ↓  ↓  ↓  ↓  ↓
Template: [1.67, 3.33, 5.33, 3.33, 1.67]  (element-wise mean)
```

---

#### 4. `cosine_similarity(window, template)`

**Purpose**: Measure shape similarity between window and template

**Algorithm**:
```
dot_product = sum(window[i] × template[i] for all i)
magnitude_window = sqrt(sum(window[i]² for all i))
magnitude_template = sqrt(sum(template[i]² for all i))

similarity = dot_product / (magnitude_window × magnitude_template)

RETURN similarity  # Range: [-1, 1], typically [0, 1] for positive signals
```

**Parameters**:
- `window` (np.ndarray): Signal window
- `template` (np.ndarray): Template to compare against

**Returns**:
- `similarity` (float): Cosine similarity value

**Interpretation**:
- `1.0`: Perfect match (identical shape)
- `0.9-0.99`: Very similar
- `0.85-0.9`: Similar (typical threshold)
- `< 0.85`: Different shape (filtered out)

**Example**:
```python
window = np.array([1, 2, 3, 4, 5])
template = np.array([1, 2, 3, 4, 5])
sim = cosine_similarity(window, template)
# Returns: 1.0 (identical)

window2 = np.array([5, 4, 3, 2, 1])
sim2 = cosine_similarity(window2, template)
# Returns: 0.636 (opposite direction)
```

---

#### 5. `filter_windows_by_similarity(windows, template, similarity_threshold)`

**Purpose**: Keep only windows that match template shape

**Algorithm**:
```
filtered_windows = []

FOR each window in windows:
    similarity = cosine_similarity(window, template)

    IF similarity >= similarity_threshold:
        ADD window to filtered_windows

RETURN filtered_windows
```

**Parameters**:
- `windows` (List[np.ndarray]): All extracted windows
- `template` (np.ndarray): Template from compute_template()
- `similarity_threshold` (float): Minimum similarity (default: 0.85)

**Returns**:
- `filtered_windows` (List[np.ndarray]): High-quality windows

**Filtering Effect**:
```
Input: 22 windows
  ├─ Window 1: similarity = 0.92 ✓ Keep
  ├─ Window 2: similarity = 0.88 ✓ Keep
  ├─ Window 3: similarity = 0.76 ✗ Remove (artifact)
  ├─ ...
  └─ Window 22: similarity = 0.91 ✓ Keep

Output: 21 windows (95.5% pass rate)
```

---

#### 6. `ppg_peak_detection_pipeline_with_template(...)`

**Purpose**: Complete pipeline combining all peak detection functions

**Algorithm**:
```
# Step 1: Detect peaks
peaks = detect_peaks(ppg_signal, height_threshold, distance_threshold)

# Step 2: Extract windows
all_windows = extract_windows(ppg_signal, peaks, window_size)

# Step 3: Compute template
template = compute_template(all_windows)

# Step 4: Filter by similarity
filtered_windows = filter_windows_by_similarity(all_windows, template, threshold)

RETURN peaks, filtered_windows, template, all_windows
```

**Parameters**:
- `ppg_signal` (np.ndarray): Preprocessed PPG signal
- `fs` (float): Sampling frequency (Hz)
- `window_duration` (float): Window length in seconds
- `height_threshold` (float): Peak height threshold
- `distance_threshold` (float): Minimum peak distance
- `similarity_threshold` (float): Template matching threshold

**Returns**:
- `peaks` (List[int]): All detected peak indices
- `filtered_windows` (List[np.ndarray]): High-quality windows
- `template` (np.ndarray): Average beat template
- `all_windows` (List[np.ndarray]): All extracted windows

---

### B. Preprocessing Functions

#### 7. `PPGSegmenter.preprocess_signal(signal)`

**Purpose**: Clean and filter PPG signal for peak detection

**Algorithm**:
```
# Step 1: Remove DC component (mean)
signal_mean = mean(signal)
signal_dc_removed = signal - signal_mean

# Step 2: Bandpass filter (0.5-10 Hz)
# Removes baseline drift and high-frequency noise
signal_filtered = butterworth_bandpass(signal_dc_removed, low=0.5, high=10, fs=fs)

# Step 3: Savitzky-Golay smoothing
# Polynomial smoothing (window=11, order=3)
signal_smoothed = savgol_filter(signal_filtered, window_length=11, polyorder=3)

RETURN signal_smoothed
```

**Why Each Step**:
1. **DC Removal**: Centers signal around zero
2. **Bandpass Filter**: Keeps heart rate frequencies (30-600 BPM = 0.5-10 Hz)
3. **Smoothing**: Reduces noise while preserving peaks

---

### C. ResNet34-1D Methods

#### 8. `GlucosePredictor.preprocess_windows(windows, target_length)`

**Purpose**: Convert PPG windows to neural network input format

**Algorithm**:
```
processed_windows = []

FOR each window in windows:
    # Step 1: Normalize (zero mean, unit variance)
    mean = np.mean(window)
    std = np.std(window)
    normalized = (window - mean) / (std + 1e-8)  # +epsilon prevents division by zero

    # Step 2: Pad or truncate to target length
    IF len(normalized) < target_length:
        # Pad with zeros
        padding = target_length - len(normalized)
        padded = np.pad(normalized, (0, padding), constant_values=0)
    ELSE:
        # Truncate
        padded = normalized[:target_length]

    ADD padded to processed_windows

# Step 3: Convert to tensor
array = np.array(processed_windows)              # Shape: (N, L)
array = np.expand_dims(array, axis=1)            # Shape: (N, 1, L) - add channel dim
tensor = torch.from_numpy(array).float()         # Convert to PyTorch tensor

RETURN tensor
```

**Parameters**:
- `windows` (List[np.ndarray]): Filtered windows from peak detection
- `target_length` (int): Desired sequence length (default: input_length)

**Returns**:
- `tensor` (torch.Tensor): Shape (batch_size, 1, target_length)

**Example**:
```python
windows = [
    np.array([100, 110, 120, 110, 100]),  # Window 1 (length=5)
    np.array([95, 105, 115, 105, 95])     # Window 2 (length=5)
]

tensor = preprocess_windows(windows, target_length=8)

# Output tensor shape: (2, 1, 8)
# Window 1 normalized and padded: [-1.26, -0.63, 0, -0.63, -1.26, 0, 0, 0]
# Window 2 normalized and padded: [-1.26, -0.63, 0, -0.63, -1.26, 0, 0, 0]
```

---

#### 9. `ResNet34_1D.forward(x)`

**Purpose**: Neural network forward pass to predict glucose

**Algorithm** (Detailed Layer-by-Layer):

```python
def forward(x):
    # Input: (batch_size, 1, sequence_length)
    # Example: (21, 1, 500)

    # ═══════════════════════════════════════════════════════════
    # STAGE 1: Initial Convolution
    # ═══════════════════════════════════════════════════════════
    x = self.conv1(x)        # (21, 1, 500) → (21, 64, 250)
                             # 7x1 conv, stride=2 reduces length by half
    x = self.bn1(x)          # Batch normalization
    x = self.relu(x)         # ReLU activation
    x = self.maxpool(x)      # (21, 64, 250) → (21, 64, 125)
                             # 3x1 maxpool, stride=2

    # ═══════════════════════════════════════════════════════════
    # STAGE 2: Layer 1 - 3 Residual Blocks (64 channels)
    # ═══════════════════════════════════════════════════════════
    x = self.layer1(x)       # (21, 64, 125) → (21, 64, 125)
    # Each block:
    #   Block 1: Conv(64→64) → BN → ReLU → Conv(64→64) → BN → +skip → ReLU
    #   Block 2: Same structure
    #   Block 3: Same structure

    # ═══════════════════════════════════════════════════════════
    # STAGE 3: Layer 2 - 4 Residual Blocks (128 channels)
    # ═══════════════════════════════════════════════════════════
    x = self.layer2(x)       # (21, 64, 125) → (21, 128, 63)
    # First block downsamples (stride=2):
    #   Block 1: Conv(64→128, s=2) → BN → ReLU → Conv(128→128) → BN
    #            + 1x1Conv(64→128, s=2) [skip] → ReLU
    # Remaining 3 blocks maintain size:
    #   Blocks 2-4: Conv(128→128) → BN → ReLU → Conv(128→128) → BN → +skip → ReLU

    # ═══════════════════════════════════════════════════════════
    # STAGE 4: Layer 3 - 6 Residual Blocks (256 channels)
    # ═══════════════════════════════════════════════════════════
    x = self.layer3(x)       # (21, 128, 63) → (21, 256, 32)
    # First block downsamples (stride=2):
    #   Block 1: Conv(128→256, s=2) → ... + 1x1Conv(128→256, s=2) → ReLU
    # Remaining 5 blocks maintain size

    # ═══════════════════════════════════════════════════════════
    # STAGE 5: Layer 4 - 3 Residual Blocks (512 channels)
    # ═══════════════════════════════════════════════════════════
    x = self.layer4(x)       # (21, 256, 32) → (21, 512, 16)
    # First block downsamples (stride=2):
    #   Block 1: Conv(256→512, s=2) → ... + 1x1Conv(256→512, s=2) → ReLU
    # Remaining 2 blocks maintain size

    # ═══════════════════════════════════════════════════════════
    # STAGE 6: Global Pooling and Classification
    # ═══════════════════════════════════════════════════════════
    x = self.avgpool(x)      # (21, 512, 16) → (21, 512, 1)
                             # Adaptive average pool reduces to single value per channel
    x = torch.flatten(x, 1)  # (21, 512, 1) → (21, 512)
                             # Flatten spatial dimension
    x = self.dropout(x)      # (21, 512) → (21, 512)
                             # Dropout with p=0.5 for regularization
    glucose = self.fc(x)     # (21, 512) → (21, 1)
                             # Fully connected layer: 512 features → 1 glucose value

    return glucose           # (21, 1) - glucose predictions in mg/dL
```

**Parameters**:
- `x` (torch.Tensor): Input tensor, shape (batch_size, 1, sequence_length)

**Returns**:
- `glucose` (torch.Tensor): Predicted glucose, shape (batch_size, 1)

**Feature Extraction Intuition**:
- **Early layers (layer1)**: Learn basic waveform patterns (peaks, valleys)
- **Middle layers (layer2, layer3)**: Learn heartbeat morphology (P-wave, QRS, T-wave analogs)
- **Deep layers (layer4)**: Learn high-level features correlating with glucose
- **FC layer**: Combines features into final glucose prediction

---

#### 10. `GlucosePredictor.predict(windows, batch_size)`

**Purpose**: Predict glucose values from filtered windows

**Algorithm**:
```
# Step 1: Preprocess windows to tensor
tensor = self.preprocess_windows(windows)  # Shape: (N, 1, L)

# Step 2: Set model to evaluation mode
self.model.eval()  # Disables dropout, uses running stats for batch norm

# Step 3: Process in batches
predictions = []

FOR i in range(0, N, batch_size):
    # Get batch
    batch = tensor[i:i+batch_size].to(self.device)

    # Disable gradient computation (faster inference)
    WITH torch.no_grad():
        # Forward pass through ResNet34
        glucose_batch = self.model.forward(batch)

        # Move to CPU and store
        predictions.append(glucose_batch.cpu().numpy())

# Step 4: Concatenate all batches
glucose_values = np.concatenate(predictions, axis=0).squeeze()

RETURN glucose_values  # Shape: (N,) - one value per window
```

**Parameters**:
- `windows` (List[np.ndarray]): Filtered PPG windows
- `batch_size` (int): Number of windows to process together (default: 32)

**Returns**:
- `glucose_values` (np.ndarray): Predicted glucose in mg/dL, shape (num_windows,)

**Why Batching**:
- **Memory efficiency**: Process 32 windows at once instead of all 1000+
- **GPU utilization**: GPUs are optimized for batch processing
- **Speed**: Faster than processing one window at a time

---

#### 11. `GlucosePredictor.predict_with_stats(windows, batch_size)`

**Purpose**: Predict glucose and compute summary statistics

**Algorithm**:
```
# Get predictions
predictions = self.predict(windows, batch_size)

# Compute statistics
mean_glucose = np.mean(predictions)
std_glucose = np.std(predictions)
min_glucose = np.min(predictions)
max_glucose = np.max(predictions)
num_windows = len(predictions)

# Return structured result
RETURN {
    'predictions': predictions,
    'mean_glucose': mean_glucose,
    'std_glucose': std_glucose,
    'min_glucose': min_glucose,
    'max_glucose': max_glucose,
    'num_windows': num_windows
}
```

**Returns**: Dictionary with keys:
- `predictions`: Individual glucose values per window
- `mean_glucose`: Average glucose across all windows
- `std_glucose`: Standard deviation (variability)
- `min_glucose`: Minimum glucose value
- `max_glucose`: Maximum glucose value
- `num_windows`: Total number of windows processed

---

## Data Flow

### Complete Data Transformation Journey

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Raw PPG Signal from VitalDB                                │
├─────────────────────────────────────────────────────────────────────┤
│ Type: np.ndarray                                                    │
│ Shape: (15000,)                                                     │
│ Values: [185.2, 185.4, 185.1, ..., 210.3]                         │
│ Units: Arbitrary PPG amplitude                                      │
│ Range: ~150-250                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ preprocess_signal()
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Preprocessed Signal                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Type: np.ndarray                                                    │
│ Shape: (15000,)                                                     │
│ Values: [-12.3, -8.5, -2.1, ..., 15.7]                            │
│ Units: Normalized amplitude (DC removed)                           │
│ Range: ~-20 to +20                                                  │
│ Transformations:                                                    │
│   - DC removed: signal - mean                                       │
│   - Bandpass filtered: 0.5-10 Hz                                   │
│   - Savitzky-Golay smoothed                                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ detect_peaks()
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Peak Indices                                               │
├─────────────────────────────────────────────────────────────────────┤
│ Type: List[int]                                                     │
│ Shape: (22,)                                                        │
│ Values: [70, 874, 1277, 2078, 2876, ..., 14476]                   │
│ Units: Sample indices                                               │
│ Meaning: Locations of systolic peaks (heartbeats)                  │
│ Frequency: ~75 BPM (22 peaks in 30 seconds ≈ 44 peaks/min × 2)    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ extract_windows()
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Extracted Windows (All)                                    │
├─────────────────────────────────────────────────────────────────────┤
│ Type: List[np.ndarray]                                              │
│ Length: 22 windows                                                  │
│ Window Shape: (500,) each                                           │
│ Window Structure:                                                   │
│   - 167 samples before peak                                         │
│   - Peak at center                                                  │
│   - 333 samples after peak                                          │
│                                                                      │
│ Example Window 0:                                                   │
│   np.array([-5.2, -4.8, ..., 15.3 (peak), ..., -6.1])             │
│   Shape: (500,)                                                     │
│   Duration: 1 second at 500 Hz                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ compute_template()
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Template (Average Beat)                                    │
├─────────────────────────────────────────────────────────────────────┤
│ Type: np.ndarray                                                    │
│ Shape: (500,)                                                       │
│ Values: [-4.2, -3.8, ..., 14.1 (avg peak), ..., -5.3]             │
│ Computation: Element-wise mean of all 22 windows                   │
│ Purpose: Represents "ideal" heartbeat waveform                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ filter_windows_by_similarity()
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Filtered Windows (High Quality)                            │
├─────────────────────────────────────────────────────────────────────┤
│ Type: List[np.ndarray]                                              │
│ Length: 21 windows (95.5% retention)                                │
│ Window Shape: (500,) each                                           │
│ Filter Criterion: cosine_similarity(window, template) >= 0.85      │
│                                                                      │
│ Removed: 1 window (similarity = 0.76, likely motion artifact)      │
│                                                                      │
│ Similarity Scores:                                                  │
│   Window 0: 0.92 ✓                                                 │
│   Window 1: 0.88 ✓                                                 │
│   Window 2: 0.76 ✗ (removed)                                       │
│   ...                                                               │
│   Window 21: 0.91 ✓                                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ preprocess_windows()
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: Normalized Tensor (Neural Network Input)                   │
├─────────────────────────────────────────────────────────────────────┤
│ Type: torch.Tensor (dtype=float32)                                  │
│ Shape: (21, 1, 500)                                                 │
│   - Batch size: 21                                                  │
│   - Channels: 1                                                     │
│   - Sequence length: 500                                            │
│                                                                      │
│ Normalization: Each window → zero mean, unit variance              │
│   Before: [-5.2, -4.8, ..., 15.3, ..., -6.1]                      │
│   After:  [-0.41, -0.38, ..., 1.21, ..., -0.48]                   │
│                                                                      │
│ Example Tensor[0]:                                                  │
│   Shape: (1, 500)                                                   │
│   Mean: ~0.0                                                        │
│   Std: ~1.0                                                         │
│   Range: ~-3.0 to +3.0 (standardized)                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ ResNet34_1D.forward()
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 8: Glucose Predictions                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Type: np.ndarray (dtype=float32)                                    │
│ Shape: (21,)                                                        │
│ Values: [-9.03, -8.95, -9.01, ..., -8.90]                         │
│ Units: mg/dL (milligrams per deciliter)                            │
│                                                                      │
│ NOTE: Current values are negative/unrealistic because model is     │
│       UNTRAINED. After training, expect values like:                │
│       [118.3, 120.5, 119.2, ..., 121.7]                           │
│                                                                      │
│ Normal glucose range: 70-140 mg/dL                                  │
│ Diabetic range: >180 mg/dL                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ compute statistics
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 9: Glucose Statistics                                         │
├─────────────────────────────────────────────────────────────────────┤
│ Type: Dictionary                                                    │
│ Contents:                                                           │
│   {                                                                 │
│     'predictions': array([-9.03, -8.95, ..., -8.90]),             │
│     'mean_glucose': -8.92,                                         │
│     'std_glucose': 0.27,                                           │
│     'min_glucose': -9.05,                                          │
│     'max_glucose': -7.76,                                          │
│     'num_windows': 21                                               │
│   }                                                                 │
│                                                                      │
│ Clinical Interpretation (after training):                           │
│   - Mean: Average glucose level over measurement period            │
│   - Std: Glucose variability (low std = stable glucose)           │
│   - Min/Max: Range of glucose fluctuation                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Interface Specifications

### API Contract for Each Function

#### Interface 1: `detect_peaks()`

```python
def detect_peaks(
    ppg_signal: np.ndarray,           # Required: Input signal
    height_threshold: float = 20,     # Optional: Peak height cutoff
    distance_threshold: float = None, # Optional: Min peak spacing
    fs: float = 100                   # Optional: Sampling rate
) -> List[int]:                       # Returns: Peak indices
    """
    Detect systolic peaks in PPG signal.

    Pre-conditions:
    - ppg_signal must be 1D numpy array
    - ppg_signal must have at least 3 samples
    - height_threshold > 0
    - fs > 0

    Post-conditions:
    - Returns list of integers
    - All returned indices are valid (0 <= idx < len(ppg_signal))
    - Returned indices are sorted in ascending order
    - Distance between consecutive peaks >= distance_threshold

    Side effects: None (pure function)

    Exceptions:
    - None (degrades gracefully, returns empty list if no peaks)
    """
```

**Example Usage**:
```python
# Basic usage
peaks = detect_peaks(signal)

# Custom thresholds
peaks = detect_peaks(
    signal,
    height_threshold=signal.mean() + 0.3 * signal.std(),
    distance_threshold=0.8 * sampling_rate,
    fs=500
)
```

---

#### Interface 2: `ppg_peak_detection_pipeline_with_template()`

```python
def ppg_peak_detection_pipeline_with_template(
    ppg_signal: np.ndarray,           # Required: Preprocessed signal
    fs: float = 100,                  # Optional: Sampling rate (Hz)
    window_duration: float = 1.0,     # Optional: Window length (seconds)
    height_threshold: float = 20,     # Optional: Peak height threshold
    distance_threshold: float = None, # Optional: Min peak distance
    similarity_threshold: float = 0.85 # Optional: Template matching threshold
) -> Tuple[List[int], List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Complete peak detection and window filtering pipeline.

    Pre-conditions:
    - ppg_signal is 1D numpy array
    - fs > 0
    - window_duration > 0
    - 0 <= similarity_threshold <= 1

    Post-conditions:
    - peaks: Sorted list of peak indices
    - filtered_windows: List of arrays, all same length
    - template: 1D array, same length as filtered windows
    - all_windows: List of arrays (may have varying lengths)

    Returns:
    - peaks: All detected peak locations
    - filtered_windows: High-quality windows (similarity >= threshold)
    - template: Average beat template
    - all_windows: All extracted windows (before filtering)

    Side effects: None
    """
```

**Example Usage**:
```python
peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
    ppg_signal=preprocessed_signal,
    fs=500,
    window_duration=1.0,
    height_threshold=signal.mean() + 0.3 * signal.std(),
    distance_threshold=0.8 * 500,
    similarity_threshold=0.85
)

print(f"Detected {len(peaks)} peaks")
print(f"Extracted {len(all_windows)} windows")
print(f"Filtered to {len(filtered_windows)} high-quality windows")
print(f"Template shape: {template.shape}")
```

---

#### Interface 3: `GlucosePredictor.__init__()`

```python
def __init__(
    input_length: int = 100,          # Optional: Expected window length
    device: str = None,               # Optional: 'cpu', 'cuda', or None (auto)
    model_path: str = None            # Optional: Path to pre-trained weights
) -> GlucosePredictor:
    """
    Initialize glucose predictor.

    Parameters:
    - input_length: Expected PPG window length in samples
    - device: Computation device ('cpu' or 'cuda'). Auto-selects if None.
    - model_path: Path to .pth file with trained weights

    Pre-conditions:
    - input_length > 0
    - If model_path provided, file must exist

    Post-conditions:
    - self.model is initialized ResNet34_1D
    - self.device is set to torch.device
    - If model_path provided, weights are loaded

    Side effects:
    - Allocates ~44 MB memory for model
    - Prints device and model loading status

    Exceptions:
    - FileNotFoundError: If model_path doesn't exist
    - RuntimeError: If CUDA requested but not available
    """
```

**Example Usage**:
```python
# Auto-detect device, untrained model
predictor = GlucosePredictor(input_length=500)

# Explicit CPU, trained model
predictor = GlucosePredictor(
    input_length=500,
    device='cpu',
    model_path='trained_glucose_model.pth'
)

# Use GPU if available
predictor = GlucosePredictor(input_length=500, device='cuda')
```

---

#### Interface 4: `GlucosePredictor.predict()`

```python
def predict(
    windows: List[np.ndarray],        # Required: PPG windows
    batch_size: int = 32              # Optional: Batch size for inference
) -> np.ndarray:
    """
    Predict glucose from PPG windows.

    Parameters:
    - windows: List of PPG signal windows (filtered_windows from pipeline)
    - batch_size: Number of windows to process simultaneously

    Pre-conditions:
    - windows is non-empty list
    - Each window is 1D numpy array
    - batch_size > 0

    Post-conditions:
    - Returns 1D array of predictions
    - len(output) == len(windows)
    - All values are finite (not NaN or inf)

    Returns:
    - glucose_values: Predicted glucose in mg/dL, shape (num_windows,)

    Side effects:
    - Sets model to eval mode
    - Allocates GPU/CPU memory proportional to batch_size

    Exceptions:
    - ValueError: If windows is empty
    - RuntimeError: If GPU out of memory (reduce batch_size)
    """
```

**Example Usage**:
```python
# Basic prediction
glucose_values = predictor.predict(filtered_windows)
print(f"Predicted glucose: {glucose_values}")
# Output: array([118.3, 120.5, 119.2, ...])

# Large dataset, smaller batches
glucose_values = predictor.predict(filtered_windows, batch_size=16)

# Single window prediction
glucose = predictor.predict([single_window])[0]
```

---

#### Interface 5: `GlucosePredictor.predict_with_stats()`

```python
def predict_with_stats(
    windows: List[np.ndarray],        # Required: PPG windows
    batch_size: int = 32              # Optional: Batch size
) -> Dict[str, any]:
    """
    Predict glucose with summary statistics.

    Parameters:
    - windows: List of PPG signal windows
    - batch_size: Batch size for inference

    Returns:
    Dictionary with keys:
    - 'predictions': np.ndarray of individual predictions
    - 'mean_glucose': float, average glucose (mg/dL)
    - 'std_glucose': float, standard deviation (mg/dL)
    - 'min_glucose': float, minimum value (mg/dL)
    - 'max_glucose': float, maximum value (mg/dL)
    - 'num_windows': int, number of windows processed

    Pre-conditions: Same as predict()
    Post-conditions: All statistics consistent with predictions

    Side effects: Same as predict()
    """
```

**Example Usage**:
```python
results = predictor.predict_with_stats(filtered_windows)

print(f"Mean Glucose: {results['mean_glucose']:.2f} mg/dL")
print(f"Std Glucose:  {results['std_glucose']:.2f} mg/dL")
print(f"Range: {results['min_glucose']:.2f} - {results['max_glucose']:.2f} mg/dL")
print(f"Processed {results['num_windows']} windows")

# Access individual predictions
for i, pred in enumerate(results['predictions']):
    print(f"Window {i}: {pred:.2f} mg/dL")
```

---

## Neural Network Architecture

### ResNet34-1D Detailed Layer Breakdown

```
┌──────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                  │
│ Shape: (batch_size, 1, 500)                                     │
│ Example: (21, 1, 500)                                            │
│   - 21 PPG windows                                               │
│   - 1 channel (univariate signal)                                │
│   - 500 samples (1 second at 500 Hz)                             │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                 INITIAL CONVOLUTION BLOCK                        │
├──────────────────────────────────────────────────────────────────┤
│ Conv1D(in=1, out=64, kernel=7, stride=2, padding=3)             │
│   Output: (21, 64, 250)                                          │
│   Parameters: 1×64×7 = 448                                       │
│                                                                  │
│ BatchNorm1D(64)                                                  │
│   Output: (21, 64, 250)                                          │
│   Parameters: 64×2 = 128 (gamma, beta)                          │
│                                                                  │
│ ReLU(inplace=True)                                               │
│   Output: (21, 64, 250)                                          │
│                                                                  │
│ MaxPool1D(kernel=3, stride=2, padding=1)                        │
│   Output: (21, 64, 125)                                          │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                    LAYER 1 (3 Blocks)                            │
│                   Channels: 64 → 64                              │
├──────────────────────────────────────────────────────────────────┤
│ ResidualBlock #1 (64→64, stride=1):                             │
│   Main path:                                                     │
│     Conv1D(64→64, k=3, s=1, p=1) → (21, 64, 125)               │
│     BN(64) → ReLU                                                │
│     Conv1D(64→64, k=3, s=1, p=1) → (21, 64, 125)               │
│     BN(64)                                                       │
│   Skip: identity (no downsample needed)                          │
│   Add + ReLU → (21, 64, 125)                                    │
│   Parameters: 64×64×3×2 + 64×2×2 = 24,832                       │
│                                                                  │
│ ResidualBlock #2 (64→64, stride=1):                             │
│   Same structure as #1                                           │
│   Output: (21, 64, 125)                                          │
│   Parameters: 24,832                                             │
│                                                                  │
│ ResidualBlock #3 (64→64, stride=1):                             │
│   Same structure as #1                                           │
│   Output: (21, 64, 125)                                          │
│   Parameters: 24,832                                             │
│                                                                  │
│ Layer 1 Total Parameters: 74,496                                 │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                    LAYER 2 (4 Blocks)                            │
│                   Channels: 64 → 128                             │
├──────────────────────────────────────────────────────────────────┤
│ ResidualBlock #1 (64→128, stride=2): [DOWNSAMPLE]               │
│   Main path:                                                     │
│     Conv1D(64→128, k=3, s=2, p=1) → (21, 128, 63)              │
│     BN(128) → ReLU                                               │
│     Conv1D(128→128, k=3, s=1, p=1) → (21, 128, 63)             │
│     BN(128)                                                      │
│   Skip: Conv1D(64→128, k=1, s=2) + BN(128) → (21, 128, 63)     │
│   Add + ReLU → (21, 128, 63)                                    │
│   Parameters: 64×128×3 + 128×128×3 + 64×128×1 + 128×4 = 82,432 │
│                                                                  │
│ ResidualBlock #2 (128→128, stride=1):                           │
│   Output: (21, 128, 63)                                          │
│   Parameters: 128×128×3×2 + 128×2×2 = 98,816                    │
│                                                                  │
│ ResidualBlock #3 (128→128, stride=1):                           │
│   Output: (21, 128, 63)                                          │
│   Parameters: 98,816                                             │
│                                                                  │
│ ResidualBlock #4 (128→128, stride=1):                           │
│   Output: (21, 128, 63)                                          │
│   Parameters: 98,816                                             │
│                                                                  │
│ Layer 2 Total Parameters: 378,880                                │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                    LAYER 3 (6 Blocks)                            │
│                   Channels: 128 → 256                            │
├──────────────────────────────────────────────────────────────────┤
│ ResidualBlock #1 (128→256, stride=2): [DOWNSAMPLE]              │
│   Main path:                                                     │
│     Conv1D(128→256, k=3, s=2, p=1) → (21, 256, 32)              │
│     BN(256) → ReLU                                               │
│     Conv1D(256→256, k=3, s=1, p=1) → (21, 256, 32)              │
│     BN(256)                                                      │
│   Skip: Conv1D(128→256, k=1, s=2) + BN(256) → (21, 256, 32)     │
│   Add + ReLU → (21, 256, 32)                                     │
│   Parameters: 128×256×3 + 256×256×3 + 128×256×1 + 256×4 = 328,192│
│                                                                  │
│ ResidualBlocks #2-6 (256→256, stride=1): ×5                     │
│   Output: (21, 256, 32)                                          │
│   Parameters per block: 256×256×3×2 + 256×2×2 = 394,752         │
│   Total: 5 × 394,752 = 1,973,760                                 │
│                                                                  │
│ Layer 3 Total Parameters: 2,301,952                              │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                    LAYER 4 (3 Blocks)                            │
│                   Channels: 256 → 512                            │
├──────────────────────────────────────────────────────────────────┤
│ ResidualBlock #1 (256→512, stride=2): [DOWNSAMPLE]              │
│   Main path:                                                     │
│     Conv1D(256→512, k=3, s=2, p=1) → (21, 512, 16)              │
│     BN(512) → ReLU                                               │
│     Conv1D(512→512, k=3, s=1, p=1) → (21, 512, 16)              │
│     BN(512)                                                      │
│   Skip: Conv1D(256→512, k=1, s=2) + BN(512) → (21, 512, 16)     │
│   Add + ReLU → (21, 512, 16)                                     │
│   Parameters: 256×512×3 + 512×512×3 + 256×512×1 + 512×4 = 1,312,256│
│                                                                  │
│ ResidualBlocks #2-3 (512→512, stride=1): ×2                     │
│   Output: (21, 512, 16)                                          │
│   Parameters per block: 512×512×3×2 + 512×2×2 = 1,576,960       │
│   Total: 2 × 1,576,960 = 3,153,920                               │
│                                                                  │
│ Layer 4 Total Parameters: 4,466,176                              │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│              GLOBAL POOLING & CLASSIFICATION                     │
├──────────────────────────────────────────────────────────────────┤
│ AdaptiveAvgPool1D(output_size=1)                                │
│   Input: (21, 512, 16)                                           │
│   Output: (21, 512, 1)                                           │
│   Operation: Average each of 512 channels over 16 time steps    │
│   Parameters: 0                                                  │
│                                                                  │
│ Flatten(start_dim=1)                                             │
│   Input: (21, 512, 1)                                            │
│   Output: (21, 512)                                              │
│                                                                  │
│ Dropout(p=0.5)                                                   │
│   Output: (21, 512)                                              │
│   During training: Randomly zero 50% of values                  │
│   During inference: No-op (all values pass through)             │
│   Parameters: 0                                                  │
│                                                                  │
│ Linear(in=512, out=1)                                            │
│   Input: (21, 512)                                               │
│   Output: (21, 1)                                                │
│   Operation: y = W×x + b                                         │
│   Parameters: 512×1 + 1 = 513                                    │
└──────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                 │
│ Shape: (batch_size, 1)                                          │
│ Example: (21, 1)                                                 │
│ Values: Glucose predictions in mg/dL                            │
│ Typical range (after training): 70-180 mg/dL                    │
└──────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════╗
║                    TOTAL PARAMETERS                              ║
╠══════════════════════════════════════════════════════════════════╣
║ Initial Conv Block:              576                             ║
║ Layer 1 (3 blocks):           74,496                             ║
║ Layer 2 (4 blocks):          378,880                             ║
║ Layer 3 (6 blocks):        2,301,952                             ║
║ Layer 4 (3 blocks):        4,466,176                             ║
║ Final FC:                        513                             ║
║────────────────────────────────────────────────────────────────  ║
║ TOTAL:                     7,222,593 (~7.2M parameters)          ║
╚══════════════════════════════════════════════════════════════════╝
```

### Why This Architecture Works

1. **Residual Connections**: Allow gradient flow through 34 layers without vanishing
2. **Progressive Downsampling**: Captures patterns at multiple time scales
3. **Channel Expansion**: 1→64→128→256→512 learns increasingly complex features
4. **Global Pooling**: Summarizes entire sequence into fixed-size representation
5. **Dropout**: Prevents overfitting on limited medical data

---

## Step-by-Step Execution

### Complete Walkthrough with Example Data

```python
# ═══════════════════════════════════════════════════════════════════
# STEP 0: Imports
# ═══════════════════════════════════════════════════════════════════
from ppg_extractor import PPGExtractor
from ppg_segmentation import PPGSegmenter
from peak_detection import ppg_peak_detection_pipeline_with_template
from resnet34_glucose_predictor import GlucosePredictor
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Extract Raw PPG Data
# ═══════════════════════════════════════════════════════════════════
extractor = PPGExtractor()
result = extractor.extract_ppg_raw(
    case_id=2,
    track_name='SNUADC/ART',
    output_dir='./output'
)

# Load data
df = pd.read_csv(result['csv_file'])
time = df['time'].values      # Shape: (15000,)
signal = df['ppg'].values      # Shape: (15000,)
sampling_rate = 500            # Hz

print(f"Loaded {len(signal)} samples at {sampling_rate} Hz")
# Output: Loaded 15000 samples at 500 Hz

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Preprocess Signal
# ═══════════════════════════════════════════════════════════════════
segmenter = PPGSegmenter(sampling_rate=sampling_rate)
preprocessed_signal = segmenter.preprocess_signal(signal)

print(f"Preprocessed signal range: {preprocessed_signal.min():.2f} to {preprocessed_signal.max():.2f}")
# Output: Preprocessed signal range: -18.23 to 19.45

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Calculate Adaptive Thresholds
# ═══════════════════════════════════════════════════════════════════
signal_mean = np.mean(preprocessed_signal)
signal_std = np.std(preprocessed_signal)
height_threshold = signal_mean + 0.3 * signal_std
distance_threshold = 0.8 * sampling_rate

print(f"Height threshold: {height_threshold:.2f}")
print(f"Distance threshold: {distance_threshold:.0f} samples")
# Output:
# Height threshold: 3.45
# Distance threshold: 400 samples

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Detect Peaks and Extract Windows
# ═══════════════════════════════════════════════════════════════════
peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
    ppg_signal=preprocessed_signal,
    fs=sampling_rate,
    window_duration=1.0,
    height_threshold=height_threshold,
    distance_threshold=distance_threshold,
    similarity_threshold=0.85
)

print(f"\nPeak Detection Results:")
print(f"  Peaks detected: {len(peaks)}")
print(f"  Windows extracted: {len(all_windows)}")
print(f"  Windows after filtering: {len(filtered_windows)}")
print(f"  Filtering rate: {len(filtered_windows)/len(all_windows)*100:.1f}%")
print(f"  Template shape: {template.shape}")
# Output:
# Peak Detection Results:
#   Peaks detected: 22
#   Windows extracted: 22
#   Windows after filtering: 21
#   Filtering rate: 95.5%
#   Template shape: (500,)

# Inspect first filtered window
print(f"\nFirst filtered window:")
print(f"  Shape: {filtered_windows[0].shape}")
print(f"  Mean: {filtered_windows[0].mean():.2f}")
print(f"  Std: {filtered_windows[0].std():.2f}")
print(f"  Min: {filtered_windows[0].min():.2f}")
print(f"  Max: {filtered_windows[0].max():.2f}")
# Output:
# First filtered window:
#   Shape: (500,)
#   Mean: -0.12
#   Std: 8.34
#   Min: -14.52
#   Max: 15.33

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Initialize Glucose Predictor
# ═══════════════════════════════════════════════════════════════════
window_length = len(filtered_windows[0])
predictor = GlucosePredictor(
    input_length=window_length,
    device='cpu'
)

print(f"\n{predictor.get_model_summary()}")
# Output:
# ResNet34-1D Glucose Predictor
# ==============================
# Input Length: 500 samples
# Total Parameters: 7,218,753
# ...

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Preprocess Windows for Neural Network
# ═══════════════════════════════════════════════════════════════════
# This happens internally in predict(), but let's see what it does
preprocessed_tensor = predictor.preprocess_windows(filtered_windows)

print(f"\nPreprocessed tensor for neural network:")
print(f"  Shape: {preprocessed_tensor.shape}")
print(f"  Dtype: {preprocessed_tensor.dtype}")
print(f"  Mean: {preprocessed_tensor.mean():.4f}")
print(f"  Std: {preprocessed_tensor.std():.4f}")
print(f"  Min: {preprocessed_tensor.min():.4f}")
print(f"  Max: {preprocessed_tensor.max():.4f}")
# Output:
# Preprocessed tensor for neural network:
#   Shape: torch.Size([21, 1, 500])
#   Dtype: torch.float32
#   Mean: -0.0012
#   Std: 0.9987
#   Min: -3.1245
#   Max: 2.9876

# ═══════════════════════════════════════════════════════════════════
# STEP 7: Predict Glucose
# ═══════════════════════════════════════════════════════════════════
glucose_results = predictor.predict_with_stats(filtered_windows, batch_size=32)

print(f"\n{'='*70}")
print("GLUCOSE PREDICTION RESULTS")
print(f"{'='*70}")
print(f"\nStatistics:")
print(f"  Mean Glucose:  {glucose_results['mean_glucose']:.2f} mg/dL")
print(f"  Std Glucose:   {glucose_results['std_glucose']:.2f} mg/dL")
print(f"  Min Glucose:   {glucose_results['min_glucose']:.2f} mg/dL")
print(f"  Max Glucose:   {glucose_results['max_glucose']:.2f} mg/dL")
print(f"  Num Windows:   {glucose_results['num_windows']}")

print(f"\nFirst 5 predictions:")
for i in range(5):
    pred = glucose_results['predictions'][i]
    peak_idx = peaks[i]
    time_sec = time[peak_idx]
    print(f"  Window {i}: {pred:.2f} mg/dL (peak at {time_sec:.2f}s)")

# Output:
# ======================================================================
# GLUCOSE PREDICTION RESULTS
# ======================================================================
#
# Statistics:
#   Mean Glucose:  -8.92 mg/dL
#   Std Glucose:   0.27 mg/dL
#   Min Glucose:   -9.05 mg/dL
#   Max Glucose:   -7.76 mg/dL
#   Num Windows:   21
#
# First 5 predictions:
#   Window 0: -9.03 mg/dL (peak at 0.14s)
#   Window 1: -8.95 mg/dL (peak at 1.75s)
#   Window 2: -9.01 mg/dL (peak at 2.55s)
#   Window 3: -9.00 mg/dL (peak at 4.16s)
#   Window 4: -8.95 mg/dL (peak at 5.75s)

# ═══════════════════════════════════════════════════════════════════
# STEP 8: Save Results
# ═══════════════════════════════════════════════════════════════════
glucose_df = pd.DataFrame({
    'window_index': range(len(glucose_results['predictions'])),
    'peak_index': peaks[:len(glucose_results['predictions'])],
    'time_seconds': [time[peak] for peak in peaks[:len(glucose_results['predictions'])]],
    'glucose_mg_dl': glucose_results['predictions']
})

glucose_df.to_csv('./output/glucose_predictions.csv', index=False)
print(f"\nSaved predictions to glucose_predictions.csv")
```

---

## Summary

### Key Takeaways

1. **Pipeline is Sequential**: Each step depends on previous outputs
2. **Data Transforms Progressively**: Raw signal → Peaks → Windows → Tensor → Glucose
3. **Quality Control at Each Stage**: Preprocessing, peak validation, window filtering
4. **ResNet34 Learns Features**: Deep network extracts glucose-predictive patterns from PPG
5. **Current Model is Untrained**: Random predictions until trained on labeled data

### Next Steps

To make this system production-ready:

1. **Collect Training Data**: PPG windows paired with true glucose measurements
2. **Train the Model**: Use `train_step()` method with labeled data
3. **Validate Performance**: Achieve MAE < 15 mg/dL, RMSE < 20 mg/dL
4. **Clinical Testing**: Validate on real patients
5. **Deploy**: Integrate trained model into web app or production system

---

**Document Version**: 1.0
**Last Updated**: 2025-01-19
**Author**: VitalDB PPG Analysis Project
