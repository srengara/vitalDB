# Feature Prioritization During Training - Implementation Guide

**Date**: December 24, 2024
**Purpose**: Prevent spectral over-focus and ensure balanced physiological feature learning
**Status**: Implementation guide for new training script

---

## Butterworth Filter Status ✅

**CONFIRMED**: Your preprocessing pipeline **DOES use Butterworth filter**

**Location**: `src/data_extraction/ppg_segmentation.py:100`

```python
# 4th order Butterworth bandpass filter
b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
signal_filtered = filtfilt(b, a, signal_centered)
```

**Also includes**: Savitzky-Golay smoothing filter (window: 50ms, polynomial order 3)

This is **GOOD** - proper filtering reduces high-frequency noise that could cause spectral over-fitting.

---

## Can We Prioritize Specific Features? YES! ✅

There are **multiple approaches** to instruct the model which features to prioritize during training:

---

## Method 1: Feature-Guided Auxiliary Loss (RECOMMENDED)

### Concept
Add an auxiliary loss that monitors and encourages specific feature importance during training.

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class FeatureGuidedLoss(nn.Module):
    """
    Auxiliary loss to guide model toward learning specific features.
    Prevents over-focus on spectral features.
    """

    def __init__(self, target_features, lambda_feature=0.1):
        """
        Args:
            target_features: Dict mapping feature names to target importance
            lambda_feature: Weight for feature guidance loss (0.1 = 10% of total loss)
        """
        super().__init__()
        self.target_features = target_features
        self.lambda_feature = lambda_feature

        # Define which features should be important (from original model)
        self.desired_importance = {
            'pulse_width': 67.84,
            'kurtosis': 64.33,
            'systolic_peak_amplitude': 58.67,
            'area_under_curve': 53.64,
            'dicrotic_notch_timing': 44.37,
            'spectral_peak_power': 53.90,  # Keep but not dominant
            'skewness': 42.30
        }

    def compute_feature_importance(self, model, ppg_batch):
        """
        Compute feature importance for current batch using gradients.
        """
        # Extract features from PPG windows
        features = self.extract_features_batch(ppg_batch)

        # Compute gradient-based importance
        feature_importance = {}
        for feat_name, feat_values in features.items():
            # Correlation between feature and layer activations
            importance = self.compute_gradient_importance(model, feat_values)
            feature_importance[feat_name] = importance

        return feature_importance

    def forward(self, model, ppg_batch, current_importance):
        """
        Compute feature guidance loss.

        Args:
            model: Current model
            ppg_batch: Batch of PPG windows
            current_importance: Dict of current feature importances

        Returns:
            Feature guidance loss
        """
        loss = 0.0

        # Penalize if morphological features are too low
        morphological_features = ['pulse_width', 'systolic_peak_amplitude', 'area_under_curve']
        for feat in morphological_features:
            if feat in current_importance:
                target = self.desired_importance[feat]
                current = current_importance[feat]

                # MSE loss: encourage current importance to match target
                loss += (current - target) ** 2

        # Penalize if spectral_peak_power is too dominant (>80)
        if 'spectral_peak_power' in current_importance:
            if current_importance['spectral_peak_power'] > 80:
                # Penalty for excessive spectral focus
                excess = current_importance['spectral_peak_power'] - 80
                loss += excess ** 2

        return self.lambda_feature * loss

    def extract_features_batch(self, ppg_batch):
        """Extract PPG features for a batch."""
        # Similar to your existing feature extraction
        features = {
            'pulse_width': [],
            'kurtosis': [],
            'systolic_peak_amplitude': [],
            # ... etc
        }

        for ppg_window in ppg_batch:
            # Extract features for this window
            pulse_width = compute_pulse_width(ppg_window)
            features['pulse_width'].append(pulse_width)
            # ... etc

        return {k: torch.tensor(v) for k, v in features.items()}

    def compute_gradient_importance(self, model, feature_values):
        """Compute importance via gradient correlation."""
        # Simplified - actual implementation would use layer activations
        return feature_values.std().item()


# Usage in training loop
def train_with_feature_guidance(model, train_loader, epochs=100):
    """
    Training loop with feature guidance.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_main = nn.MSELoss()  # Main glucose prediction loss
    criterion_feature = FeatureGuidedLoss(lambda_feature=0.1)

    for epoch in range(epochs):
        for batch_idx, (ppg_batch, glucose_batch) in enumerate(train_loader):

            # Forward pass
            predictions = model(ppg_batch)

            # Main loss (glucose prediction)
            loss_main = criterion_main(predictions, glucose_batch)

            # Feature guidance loss (every N batches to save computation)
            if batch_idx % 10 == 0:
                # Compute current feature importance
                current_importance = criterion_feature.compute_feature_importance(model, ppg_batch)

                # Feature guidance loss
                loss_feature = criterion_feature(model, ppg_batch, current_importance)
            else:
                loss_feature = 0

            # Total loss
            loss_total = loss_main + loss_feature

            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Log
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Main Loss: {loss_main.item():.4f}")
                print(f"  Feature Loss: {loss_feature:.4f if isinstance(loss_feature, float) else loss_feature.item():.4f}")
                if batch_idx % 10 == 0:
                    print(f"  Current Importance: {current_importance}")
```

---

## Method 2: Multi-Task Learning with Feature Prediction

### Concept
Train the model to predict both glucose AND key features simultaneously.

### Implementation

```python
class MultiTaskResNet34(nn.Module):
    """
    ResNet34 with multiple output heads:
    - Main head: Glucose prediction
    - Auxiliary heads: Key feature predictions
    """

    def __init__(self, input_length=100):
        super().__init__()

        # Main ResNet backbone (shared)
        self.backbone = ResNet34_1D(input_length=input_length, num_classes=1)

        # Auxiliary feature prediction heads
        self.pulse_width_head = nn.Linear(512, 1)
        self.kurtosis_head = nn.Linear(512, 1)
        self.peak_amplitude_head = nn.Linear(512, 1)

    def forward(self, x, return_features=False):
        # Get features from backbone
        features = self.backbone.layer4(
            self.backbone.layer3(
                self.backbone.layer2(
                    self.backbone.layer1(
                        self.backbone.initial_layer(x)
                    )
                )
            )
        )

        # Pool features
        pooled = self.backbone.pool(features).squeeze(-1)

        # Main glucose prediction
        glucose = self.backbone.fc(pooled)

        if return_features:
            # Auxiliary feature predictions
            pulse_width_pred = self.pulse_width_head(pooled)
            kurtosis_pred = self.kurtosis_head(pooled)
            peak_amplitude_pred = self.peak_amplitude_head(pooled)

            return glucose, {
                'pulse_width': pulse_width_pred,
                'kurtosis': kurtosis_pred,
                'peak_amplitude': peak_amplitude_pred
            }

        return glucose


# Training with multi-task loss
def train_multitask(model, train_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion_glucose = nn.MSELoss()
    criterion_features = nn.MSELoss()

    for epoch in range(epochs):
        for ppg_batch, glucose_batch, feature_batch in train_loader:

            # Forward pass with features
            glucose_pred, feature_preds = model(ppg_batch, return_features=True)

            # Main loss
            loss_glucose = criterion_glucose(glucose_pred, glucose_batch)

            # Feature losses (encourage learning these features)
            loss_pulse_width = criterion_features(
                feature_preds['pulse_width'],
                feature_batch['pulse_width']
            )
            loss_kurtosis = criterion_features(
                feature_preds['kurtosis'],
                feature_batch['kurtosis']
            )
            loss_peak_amplitude = criterion_features(
                feature_preds['peak_amplitude'],
                feature_batch['peak_amplitude']
            )

            # Total loss (weighted combination)
            loss_total = (
                1.0 * loss_glucose +
                0.3 * loss_pulse_width +
                0.3 * loss_kurtosis +
                0.3 * loss_peak_amplitude
            )

            # Backward
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
```

---

## Method 3: Frequency-Aware Data Augmentation

### Concept
Augment training data to reduce spectral artifact dependence.

### Implementation

```python
def frequency_aware_augmentation(ppg_window, augment_prob=0.5):
    """
    Augment PPG data to prevent spectral over-fitting.
    """
    if np.random.random() < augment_prob:
        # Option 1: Add controlled noise in time domain
        noise = np.random.normal(0, 0.05 * ppg_window.std(), ppg_window.shape)
        ppg_augmented = ppg_window + noise

        # Option 2: Frequency domain augmentation
        fft = np.fft.fft(ppg_window)

        # Randomly attenuate some frequency components
        freq_mask = np.random.random(fft.shape) > 0.3
        fft_masked = fft * freq_mask

        ppg_augmented = np.real(np.fft.ifft(fft_masked))

        # Option 3: Time warping (preserve morphology, change frequency slightly)
        stretch_factor = np.random.uniform(0.95, 1.05)
        indices = np.linspace(0, len(ppg_window)-1, int(len(ppg_window) * stretch_factor))
        ppg_augmented = np.interp(
            np.arange(len(ppg_window)),
            indices,
            ppg_window
        )

        return ppg_augmented

    return ppg_window
```

---

## Method 4: Layer-wise Feature Monitoring (Early Stopping)

### Concept
Monitor feature importance during training and stop if spectral features dominate.

### Implementation

```python
class FeatureMonitorCallback:
    """
    Monitor feature importance during training.
    Alert or stop if spectral over-focus detected.
    """

    def __init__(self, check_every=5, spectral_threshold=80):
        self.check_every = check_every
        self.spectral_threshold = spectral_threshold
        self.history = []

    def check_features(self, epoch, model, val_loader):
        """Check feature importance on validation set."""
        if epoch % self.check_every != 0:
            return True  # Continue training

        # Compute feature importance
        feature_importance = compute_feature_importance(model, val_loader)

        # Check for spectral dominance
        spectral_features = [
            'spectral_peak_power',
            'spectral_spread',
            'spectral_centroid'
        ]

        spectral_total = sum(
            feature_importance.get(f, 0) for f in spectral_features
        )

        morphological_features = [
            'pulse_width',
            'systolic_peak_amplitude',
            'area_under_curve'
        ]

        morphological_total = sum(
            feature_importance.get(f, 0) for f in morphological_features
        )

        # Alert if spectral dominance
        if spectral_total > self.spectral_threshold:
            print(f"\n{'='*60}")
            print(f"WARNING: Spectral features dominating at epoch {epoch}")
            print(f"Spectral total: {spectral_total:.2f}")
            print(f"Morphological total: {morphological_total:.2f}")
            print(f"{'='*60}\n")

            # Option: Adjust learning rate for spectral pathway
            # Option: Increase feature guidance loss weight
            # Option: Stop training

        self.history.append({
            'epoch': epoch,
            'spectral': spectral_total,
            'morphological': morphological_total,
            'feature_importance': feature_importance
        })

        return True  # Continue training
```

---

## Method 5: Transfer Learning from Original Model (EASIEST)

### Concept
Start from the original model's weights, which already learned correct features.

### Implementation

```python
def transfer_learning_approach():
    """
    Initialize Epoch 85 model with Original Model weights.
    Fine-tune on VitalDB data with frozen early layers.
    """

    # Load original model (with good features)
    original_checkpoint = torch.load('path/to/original_model.pth')

    # Load new model
    new_model = ResNet34_1D(input_length=100, num_classes=1)

    # Transfer weights from original model
    new_model.load_state_dict(original_checkpoint['model_state_dict'])

    # Freeze early layers (preserve feature learning)
    for param in new_model.layer1.parameters():
        param.requires_grad = False
    for param in new_model.layer2.parameters():
        param.requires_grad = False

    # Fine-tune only later layers
    # This preserves morphological feature learning while adapting to VitalDB

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, new_model.parameters()),
        lr=0.0001  # Lower learning rate for fine-tuning
    )

    # Train on VitalDB data
    # ...

    return new_model
```

---

## Recommended Strategy: COMBINED APPROACH

### Phase 1: Transfer Learning (Week 1)
1. Initialize from original model
2. Freeze layers 1-2
3. Fine-tune layers 3-4 on VitalDB

### Phase 2: Feature-Guided Training (Week 2)
1. Unfreeze all layers
2. Add feature guidance loss (Method 1)
3. Monitor with callbacks (Method 4)

### Phase 3: Validation (Week 3)
1. Compute feature importance
2. Verify morphological features restored
3. Test on 84-case validation set

---

## Expected Improvements

| Method | Implementation Time | Expected MAE | Complexity |
|--------|-------------------|--------------|------------|
| **Transfer Learning** | 1-2 days | 15-20 mg/dL | ⭐ Easy |
| **Feature Guidance Loss** | 3-5 days | 10-15 mg/dL | ⭐⭐ Medium |
| **Multi-Task Learning** | 5-7 days | 10-15 mg/dL | ⭐⭐⭐ Hard |
| **Combined Approach** | 1-2 weeks | 10-12 mg/dL | ⭐⭐ Medium |

---

## Implementation Checklist

- [ ] Create new training script (don't modify existing)
- [ ] Implement transfer learning from original model
- [ ] Add feature guidance loss
- [ ] Add feature monitoring callback
- [ ] Test on small subset first
- [ ] Run full training on VitalDB
- [ ] Compute feature importance on trained model
- [ ] Validate on 84-case test set
- [ ] Compare to Epoch 85 model

---

## Next Steps

1. **IMMEDIATE**: Create `train_with_feature_regularization.py` (new file)
2. **WEEK 1**: Test transfer learning approach
3. **WEEK 2**: Add feature guidance if needed
4. **WEEK 3**: Validate and compare results

Would you like me to create the complete training script with these features?
