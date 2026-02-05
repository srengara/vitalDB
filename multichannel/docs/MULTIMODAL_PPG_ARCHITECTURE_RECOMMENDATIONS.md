# Multi-Modal PPG Architecture Recommendations

**Date:** 2025-12-22
**Task:** Train model for augmented PPG with 4 waveforms (FSR, GREEN, RED, IR)

---

## Current Architecture Analysis

### Current ResNet34-1D Design
- **Input:** `(batch_size, 1, sequence_length)` - Single channel PPG
- **First Conv Layer:** `in_channels=1` → 64 channels
- **Architecture:** 16 residual blocks (3→4→6→3 pattern)
- **Output:** Single glucose value (mg/dL)

**Current Limitation:** Designed for single-channel input only.

---

## Your Multi-Modal Data Format

```
Case_id | systolic | diastolic | fsr_ppg_waveform | green_ppg_waveform | red_ppg_waveform | ir_ppg_waveform | glucose_label
```

**4 PPG Waveforms:**
1. FSR (Force Sensitive Resistor)
2. GREEN wavelength PPG
3. RED wavelength PPG
4. IR (Infrared) wavelength PPG

**Each waveform:** Likely 100 samples (same as current vanilla PPG)

---

## Architecture Options for Multi-Modal PPG

### **Option 1: Early Fusion (Multi-Channel Input) - RECOMMENDED ✓**

**Concept:** Stack all 4 waveforms as separate input channels, like RGB channels in image processing.

**Input Shape:** `(batch_size, 4, sequence_length)`
- Channel 0: FSR waveform
- Channel 1: GREEN waveform
- Channel 2: RED waveform
- Channel 3: IR waveform

**Architecture Changes:**
```python
class MultiModalResNet34_1D(nn.Module):
    def __init__(self, input_length=100, num_channels=4, dropout_rate=0.5):
        super().__init__()

        # ONLY CHANGE: Modify first conv layer to accept 4 input channels
        self.conv1 = nn.Conv1d(
            in_channels=4,  # ← Changed from 1 to 4
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Rest of architecture IDENTICAL to current ResNet34-1D
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # ... (same as before)
```

**Pros:**
- ✅ **Minimal code changes** - Only first conv layer modified
- ✅ **Shared feature learning** - All waveforms processed together
- ✅ **Automatic cross-modal interaction** - Conv filters learn correlations between channels
- ✅ **Proven approach** - Standard for multi-channel signals (RGB images, multi-lead ECG)
- ✅ **Same preprocessing pipeline** - Each waveform normalized independently

**Cons:**
- ❌ Assumes all waveforms have similar importance (can be mitigated with attention)

**When to Use:**
- **Default choice for multi-modal PPG**
- When waveforms are time-aligned and same length
- When you want the model to learn inter-channel relationships automatically

---

### **Option 2: Late Fusion (Separate Encoders)**

**Concept:** Each waveform processed by separate ResNet encoder, then concatenate features before final FC layer.

**Architecture:**
```python
class LateFusionResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Separate encoder for each waveform
        self.fsr_encoder = ResNet34_1D_Encoder()    # → 512-dim features
        self.green_encoder = ResNet34_1D_Encoder()  # → 512-dim features
        self.red_encoder = ResNet34_1D_Encoder()    # → 512-dim features
        self.ir_encoder = ResNet34_1D_Encoder()     # → 512-dim features

        # Fusion layer
        self.fc_fusion = nn.Linear(512 * 4, 512)
        self.fc_output = nn.Linear(512, 1)  # Glucose prediction

    def forward(self, fsr, green, red, ir):
        # Process each waveform separately
        fsr_features = self.fsr_encoder(fsr.unsqueeze(1))
        green_features = self.green_encoder(green.unsqueeze(1))
        red_features = self.red_encoder(red.unsqueeze(1))
        ir_features = self.ir_encoder(ir.unsqueeze(1))

        # Concatenate features
        combined = torch.cat([fsr_features, green_features, red_features, ir_features], dim=1)

        # Fusion and prediction
        fused = F.relu(self.fc_fusion(combined))
        glucose = self.fc_output(fused)
        return glucose
```

**Pros:**
- ✅ **Separate feature learning** - Each modality has dedicated parameters
- ✅ **Easy to debug** - Can inspect features from each waveform
- ✅ **Flexible** - Can add/remove modalities easily
- ✅ **Pre-training** - Can initialize encoders with vanilla PPG weights

**Cons:**
- ❌ **4x parameters** - Much larger model (4 separate ResNet34s)
- ❌ **4x training time** - More computation
- ❌ **Late interaction** - Cross-modal patterns learned only at final layers
- ❌ **More complex training** - Higher risk of overfitting

**When to Use:**
- When you have pre-trained models for each modality
- When modalities have very different characteristics
- When you need interpretability (feature importance per modality)

---

### **Option 3: Hierarchical Fusion (Hybrid)**

**Concept:** Combine early + late fusion. Process pairs of related waveforms together, then fuse.

**Architecture:**
```python
class HierarchicalFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # Group related waveforms
        self.optical_encoder = ResNet34_1D(num_channels=3)  # GREEN + RED + IR
        self.mechanical_encoder = ResNet34_1D(num_channels=1)  # FSR

        # Fusion
        self.fc_fusion = nn.Linear(512 * 2, 512)
        self.fc_output = nn.Linear(512, 1)

    def forward(self, fsr, green, red, ir):
        # Stack optical waveforms
        optical = torch.stack([green, red, ir], dim=1)  # (batch, 3, seq_len)

        # Process groups
        optical_features = self.optical_encoder(optical)
        mechanical_features = self.mechanical_encoder(fsr.unsqueeze(1))

        # Fuse
        combined = torch.cat([optical_features, mechanical_features], dim=1)
        fused = F.relu(self.fc_fusion(combined))
        return self.fc_output(fused)
```

**Pros:**
- ✅ **Balanced** - Captures both low-level and high-level interactions
- ✅ **Domain knowledge** - Groups modalities by physics (optical vs mechanical)
- ✅ **Moderate parameters** - Only 2 encoders instead of 4

**Cons:**
- ❌ **Design complexity** - Requires domain knowledge to group modalities
- ❌ **Still 2x parameters** - Larger than early fusion

**When to Use:**
- When you understand physical relationships between modalities
- When some waveforms are more related than others

---

### **Option 4: Attention-Based Fusion**

**Concept:** Early fusion + attention mechanism to weight modality importance dynamically.

**Architecture:**
```python
class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # Multi-channel encoder
        self.encoder = MultiModalResNet34_1D(num_channels=4)  # Early fusion

        # Channel attention (learn importance weights)
        self.channel_attention = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, multi_channel_input):  # (batch, 4, seq_len)
        # Apply channel attention BEFORE encoder
        # Compute attention weights from pooled signals
        pooled = multi_channel_input.mean(dim=2)  # (batch, 4)
        attention_weights = self.channel_attention(pooled)  # (batch, 4)

        # Reweight channels
        weighted_input = multi_channel_input * attention_weights.unsqueeze(2)

        # Process with encoder
        glucose = self.encoder(weighted_input)
        return glucose
```

**Pros:**
- ✅ **Adaptive** - Learns which modalities matter for each sample
- ✅ **Interpretable** - Attention weights show modality importance
- ✅ **Handles missing modalities** - Can zero-weight unavailable channels

**Cons:**
- ❌ **More complex** - Additional attention mechanism to tune
- ❌ **Risk of collapse** - Attention might ignore some channels entirely

**When to Use:**
- When modality importance varies across samples
- When you want interpretability
- When some waveforms might be noisy/missing

---

## Recommended Approach: Early Fusion (Option 1)

### Why Early Fusion is Best for Your Use Case

1. **Minimal Changes:** Only change `in_channels=1` → `in_channels=4` in first conv layer
2. **Proven for Multi-Channel Signals:** Standard approach for multi-lead ECG, RGB images
3. **Automatic Feature Learning:** Conv1D filters learn cross-modal patterns (e.g., "GREEN peak + IR trough" → high glucose)
4. **Same Pipeline:** Reuse your existing data preprocessing, training loop, evaluation
5. **Efficient:** No parameter explosion like late fusion

### Implementation Steps

**Step 1: Modify Model Architecture**
```python
# In resnet34_glucose_predictor.py
class MultiModalResNet34_1D(nn.Module):
    def __init__(
        self,
        input_length: int = 100,
        num_channels: int = 4,  # NEW: Number of input waveforms
        num_classes: int = 1,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        # ONLY CHANGE: Multi-channel input
        self.conv1 = nn.Conv1d(
            in_channels=num_channels,  # 4 for FSR+GREEN+RED+IR
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Everything else IDENTICAL
        self.bn1 = nn.BatchNorm1d(64)
        # ... (rest of ResNet34 architecture unchanged)
```

**Step 2: Data Loading**
```python
# In dataset preparation
def load_multimodal_sample(case_id):
    """
    Load all 4 waveforms for a single case.

    Returns:
        waveforms: np.ndarray of shape (4, sequence_length)
        glucose: float
    """
    fsr_waveform = load_ppg_waveform(case_id, 'fsr')      # (100,)
    green_waveform = load_ppg_waveform(case_id, 'green')  # (100,)
    red_waveform = load_ppg_waveform(case_id, 'red')      # (100,)
    ir_waveform = load_ppg_waveform(case_id, 'ir')        # (100,)

    # Stack as channels (like RGB image)
    waveforms = np.stack([fsr_waveform, green_waveform, red_waveform, ir_waveform], axis=0)
    # Shape: (4, 100)

    glucose = load_glucose_label(case_id)

    return waveforms, glucose
```

**Step 3: Normalization**
```python
# Normalize each channel independently (like image normalization)
def normalize_multimodal_ppg(waveforms):
    """
    Normalize each channel using its own statistics.

    Args:
        waveforms: (4, seq_len) array

    Returns:
        normalized: (4, seq_len) array
    """
    normalized = np.zeros_like(waveforms)

    for i in range(4):  # For each channel
        mean = waveforms[i].mean()
        std = waveforms[i].std() + 1e-8
        normalized[i] = (waveforms[i] - mean) / std

    return normalized
```

**Step 4: Training (Minimal Changes)**
```python
# Training loop - almost identical to vanilla PPG
model = MultiModalResNet34_1D(num_channels=4)

for epoch in range(num_epochs):
    for batch_waveforms, batch_labels in train_loader:
        # batch_waveforms: (batch_size, 4, 100)
        # batch_labels: (batch_size,)

        predictions = model(batch_waveforms)
        loss = criterion(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Expected Performance Improvements

### Why Multi-Modal Should Outperform Vanilla PPG

1. **Complementary Information:**
   - **GREEN PPG:** Superficial vessels, strong in well-perfused tissue
   - **RED PPG:** Moderate penetration depth
   - **IR PPG:** Deepest penetration, less affected by skin tone
   - **FSR:** Mechanical pressure, captures pulsatile volume changes

2. **Robustness to Artifacts:**
   - Motion artifacts affect each wavelength differently
   - Model can learn to trust the cleanest signal per sample

3. **Redundancy:**
   - If one channel is noisy, others provide backup information

4. **Glucose-Specific Patterns:**
   - Different wavelengths may be sensitive to blood glucose absorption at different ranges
   - Multi-spectral analysis is standard in non-invasive glucose monitoring research

### Performance Targets (Based on Milestone Plan)

| Milestone | Target Accuracy | Dataset Size |
|-----------|----------------|--------------|
| **Feb 28** | **75%** | 150 cases × 5 tracks |
| **Mar 4** | **75% (validated)** | 200 cases × 5 tracks |

**Comparison to Vanilla PPG:**
- Vanilla PPG: 38% good cases (MAE ≤20), 64 mg/dL average MAE
- Multi-modal target: 75% accuracy (likely MAE ≤15-20)
- **Expected improvement: ~2x better performance**

---

## Alternative Architectures to Consider

### 1. **Transformer-Based (if dataset is large enough)**

```python
class TransformerMultiModal(nn.Module):
    """
    Use attention to model long-range dependencies in waveforms.
    Requires larger dataset (>500 cases).
    """
    def __init__(self, num_channels=4, seq_len=100):
        super().__init__()
        self.embedding = nn.Linear(num_channels, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=6
        )
        self.fc = nn.Linear(128, 1)
```

**When to Use:** If you have 500+ multi-modal cases and GPU resources.

---

### 2. **CNN-LSTM Hybrid**

```python
class CNN_LSTM_Fusion(nn.Module):
    """
    CNN extracts spatial features, LSTM models temporal dependencies.
    """
    def __init__(self, num_channels=4):
        super().__init__()
        self.cnn = nn.Conv1d(num_channels, 64, kernel_size=3)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2)
        self.fc = nn.Linear(128, 1)
```

**When to Use:** If PPG waveforms have strong temporal dependencies (unlikely for 100-sample windows).

---

### 3. **Wavelet-Based Feature Extraction**

```python
# Preprocess with wavelet transform before feeding to ResNet
import pywt

def extract_wavelet_features(waveform):
    """
    Decompose PPG into frequency bands (useful for removing noise).
    """
    coeffs = pywt.wavedec(waveform, 'db4', level=4)
    # Feed coeffs to ResNet instead of raw waveform
```

**When to Use:** If waveforms have strong noise/artifacts (likely in real-world data).

---

## Action Plan

### Immediate Next Steps (Week 1)

1. ✅ **Implement Early Fusion ResNet34-1D**
   - Modify `resnet34_glucose_predictor.py`
   - Change `in_channels=1` → `in_channels=4`
   - Add `num_channels` parameter

2. ✅ **Update Data Pipeline**
   - Create multi-modal data loader
   - Stack [FSR, GREEN, RED, IR] as channels
   - Normalize each channel independently

3. ✅ **Baseline Training**
   - Train on first 50 multi-modal cases
   - Compare to vanilla PPG performance on same cases
   - Target: Beat vanilla PPG MAE (currently 64 mg/dL)

### Week 2-3: Optimization

4. ✅ **Hyperparameter Tuning**
   - Learning rate: Try [1e-4, 3e-4, 1e-3]
   - Batch size: [16, 32, 64]
   - Dropout: [0.3, 0.5, 0.7]

5. ✅ **Data Augmentation** (if needed)
   - Random amplitude scaling (0.9-1.1x)
   - Random time shift (±5 samples)
   - Channel dropout (randomly zero one channel)

### Week 4: Advanced Techniques (if baseline insufficient)

6. ⚠️ **Try Attention Fusion** (if early fusion doesn't reach 75%)
   - Add channel attention module
   - Monitor which channels model relies on

7. ⚠️ **Ensemble** (if close to target)
   - Train 3-5 models with different random seeds
   - Average predictions

---

## Code Changes Summary

### Minimal Changes Required (Early Fusion)

**File:** `src/training/resnet34_glucose_predictor.py`

```python
# Line ~174: Change in_channels parameter
class MultiModalResNet34_1D(ResNet34_1D):  # Inherit from existing class
    def __init__(self, input_length=100, num_channels=4, dropout_rate=0.5):
        # Call parent __init__ but override conv1
        super().__init__(input_length, num_classes=1, dropout_rate=dropout_rate)

        # ONLY modification: Replace first conv layer
        self.conv1 = nn.Conv1d(
            in_channels=num_channels,  # ← Changed from 1
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
```

**File:** `src/training/train_glucose_predictor.py`

```python
# Line ~50: Update model instantiation
model = MultiModalResNet34_1D(
    input_length=100,
    num_channels=4,  # ← NEW parameter
    dropout_rate=0.5
)
```

**File:** `src/data/multimodal_dataset.py` (NEW FILE)

```python
class MultiModalPPGDataset(Dataset):
    def __init__(self, fsr_file, green_file, red_file, ir_file, glucose_file):
        self.fsr_data = pd.read_csv(fsr_file)
        self.green_data = pd.read_csv(green_file)
        self.red_data = pd.read_csv(red_file)
        self.ir_data = pd.read_csv(ir_file)
        self.glucose_labels = pd.read_csv(glucose_file)

    def __getitem__(self, idx):
        # Load all 4 waveforms for sample idx
        fsr = self.fsr_data.iloc[idx].values[3:]  # Skip metadata columns
        green = self.green_data.iloc[idx].values[3:]
        red = self.red_data.iloc[idx].values[3:]
        ir = self.ir_data.iloc[idx].values[3:]

        # Stack as channels
        waveforms = np.stack([fsr, green, red, ir], axis=0)  # (4, 100)

        # Normalize
        waveforms = normalize_multimodal_ppg(waveforms)

        glucose = self.glucose_labels.iloc[idx]['glucose_mg_dl']

        return torch.FloatTensor(waveforms), torch.FloatTensor([glucose])
```

---

## Key Takeaways

### ✅ ResNet1D IS Suitable for Multi-Modal PPG

**YES**, ResNet1D is excellent for this task because:
1. Conv1D naturally handles multi-channel input (just like Conv2D for RGB images)
2. Residual connections help with gradient flow in deep networks
3. Proven architecture for time-series data (ECG, audio, etc.)

### ✅ Minimal Modifications Needed

**Only change 1 parameter:** `in_channels=1` → `in_channels=4`

Everything else stays the same:
- Same training loop
- Same loss function (MSE or MAE)
- Same evaluation metrics
- Same preprocessing (just normalize 4 channels instead of 1)

### ✅ Expected Improvement: 2x Better Performance

**Vanilla PPG:** 64 mg/dL MAE → **Multi-Modal Target:** <30 mg/dL MAE (75% accuracy)

---

## References & Further Reading

1. **Multi-Channel CNNs:**
   - He et al. (2016) - "Deep Residual Learning for Image Recognition" (shows multi-channel input)
   - Hannun et al. (2019) - "Cardiologist-level arrhythmia detection" (multi-lead ECG)

2. **PPG-Based Glucose Monitoring:**
   - Monte-Moreno (2011) - "Non-invasive estimate of blood glucose and blood pressure"
   - Agarwal et al. (2019) - "Multi-wavelength PPG for glucose estimation"

3. **Multi-Modal Fusion:**
   - Ramachandram & Taylor (2017) - "Deep Multimodal Learning: A Survey"

---

**RECOMMENDATION:** Start with **Early Fusion (Option 1)**. It's the simplest, most proven approach with minimal code changes. If it doesn't reach 75% accuracy, then try Attention Fusion (Option 4) as a next step.
