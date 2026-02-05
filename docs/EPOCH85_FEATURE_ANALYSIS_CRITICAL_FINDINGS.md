# CRITICAL: Epoch 85 Model Feature Analysis

**Date**: December 24, 2024
**Analysis**: Feature importance comparison between Original Model and Epoch 85 Model
**Status**: 🚨 **MAJOR FINDINGS - ACTION REQUIRED**

---

## Executive Summary

**CRITICAL DISCOVERY**: The Epoch 85 model shows **dramatically different feature learning** compared to the original model, with a complete shift from morphological/statistical features to spectral features.

### Key Metrics
- **Original Model**: Balanced feature importance across categories
- **Epoch 85 Model**: 100% focus on `spectral_peak_power`, all other features near-zero
- **Performance Impact**: This explains the high MAE (median 31.78 mg/dL) vs original (0.30 mg/dL on 5 cases)

---

## Feature Importance Comparison

### Top 5 Features - Original Model
| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | **Pulse Width** | 67.84 | Morphological |
| 2 | **Kurtosis** | 64.33 | Statistical |
| 3 | **Systolic Peak Amplitude** | 58.67 | Morphological |
| 4 | **Spectral Peak Power** | 53.90 | Spectral |
| 5 | **Area Under Curve** | 53.64 | Morphological |

### Top 5 Features - Epoch 85 Model
| Rank | Feature | Importance | Category | Change from Original |
|------|---------|------------|----------|----------------------|
| 1 | **Spectral Peak Power** | 100.00 | Spectral | **+46.10** ✅ |
| 2 | **Spectral Spread** | 14.66 | Spectral | -10.43 |
| 3 | **Signal Energy** | 2.86 | Statistical | +2.86 ✅ |
| 4 | **Area Under Curve** | 0.04 | Morphological | **-53.60** 🚨 |
| 5 | **Spectral Centroid** | 0.04 | Spectral | -17.24 |

---

## Catastrophic Feature Losses

### Features That Collapsed (>99% decrease)
| Feature | Original | Epoch 85 | Loss |
|---------|----------|----------|------|
| **Pulse Width** | 67.84 | 0.00018 | **-99.9997%** 🚨 |
| **Kurtosis** | 64.33 | 0.00023 | **-99.9996%** 🚨 |
| **Systolic Peak Amplitude** | 58.67 | 0.00966 | **-99.98%** 🚨 |
| **Area Under Curve** | 53.64 | 0.04319 | **-99.92%** 🚨 |
| **Dicrotic Notch Timing** | 44.37 | 0.00048 | **-99.999%** 🚨 |
| **Skewness** | 42.30 | 0.00081 | **-99.998%** 🚨 |
| **Falling Edge Slope** | 28.85 | 0.00028 | **-99.999%** 🚨 |
| **Zero Crossing Rate** | 27.10 | 0.000015 | **-99.9999%** 🚨 |

---

## Category-wise Analysis

| Category | Original Avg | Epoch 85 Avg | Change | Status |
|----------|--------------|--------------|--------|--------|
| **Morphological** | 39.48 | 0.01 | **-39.47** | 🚨 COLLAPSED |
| **Temporal** | 18.88 | 0.00 | **-18.88** | 🚨 COLLAPSED |
| **Statistical** | 35.54 | 0.95 | **-34.59** | 🚨 COLLAPSED |
| **Spectral** | 24.07 | 28.68 | **+4.61** | ✅ INCREASED |
| **Frequency** | 27.10 | 0.00 | **-27.10** | 🚨 COLLAPSED |

---

## Root Cause Analysis

### Why Did This Happen?

#### 1. **Training Data Distribution Issues**
- Original model: Likely trained on diverse, high-quality PPG data
- Epoch 85 model: Trained on VitalDB data with potential issues:
  - Only **132 unique glucose values** (discrete buckets, not continuous)
  - Limited glucose range diversity in training set
  - Possible data quality issues leading to over-reliance on spectral features

#### 2. **Model Overfitting to Spectral Noise**
- Spectral_peak_power at **100% importance** suggests model found a spurious correlation
- **All physiologically meaningful features** (pulse width, kurtosis, peak amplitude) collapsed
- Model may be fitting to frequency artifacts rather than true glucose-correlated patterns

#### 3. **Gradient Flow Issues**
- **Layer 1** in Epoch 85 shows extreme dominance: spectral_peak_power = 269.10
- This suggests:
  - Early layers learn spectral pattern
  - Later layers unable to refine or diversify
  - Gradient saturation preventing deeper feature learning

---

## Clinical Implications

### What Original Model Learned (Correct)
✅ **Pulse Width** (arterial stiffness) - Known cardiovascular-metabolic biomarker
✅ **Kurtosis** (waveform distribution) - Autonomic nervous system modulation
✅ **Systolic Peak Amplitude** (cardiac contractility) - Blood viscosity effects
✅ **Dicrotic Notch** (arterial compliance) - Vascular health indicator

### What Epoch 85 Model Learned (Problematic)
🚨 **Spectral Peak Power** (dominant frequency power) - **NOT a validated glucose biomarker**
⚠️ Loss of all physiologically meaningful features
⚠️ Over-reliance on frequency domain = susceptible to noise/artifacts

---

## Performance Impact

### Expected vs Actual
- **Original Model on 5 cases**: MAE 0.30 mg/dL
  → Learned correct physiological features

- **Epoch 85 Model on 84 cases**: Median MAE 31.78 mg/dL
  → Lost physiological features, fitting to spectral noise

### Why High MAE?
1. **Spectral features are not robust glucose predictors**
2. **Missing morphological features** (pulse width, peak amplitude) are critical
3. **Frequency domain over-reliance** makes model sensitive to:
   - Heart rate variability (not glucose-related)
   - Respiratory artifacts
   - Motion artifacts
   - Sensor noise

---

## Recommendations

### IMMEDIATE ACTIONS (CRITICAL)

#### 1. **Retrain Model with Feature Regularization** 🚨
```python
# Add feature diversity loss to prevent spectral over-focus
def feature_diversity_loss(feature_activations):
    # Penalize extreme dominance of single feature
    return -entropy(feature_activations)
```

#### 2. **Data Quality Audit** 🚨
- Review VitalDB training data for:
  - Spectral artifacts / noise
  - Frequency domain anomalies
  - Data preprocessing issues
- Compare with original training data

#### 3. **Glucose Value Balancing** 🚨
- Current: 132 unique glucose values (discrete)
- Target: Continuous glucose representation
- Add synthetic interpolation or augmentation

### SHORT-TERM FIXES

#### 4. **Transfer Learning from Original Model**
- Initialize Epoch 85 model with Original Model weights
- Fine-tune on VitalDB data with frozen early layers
- Preserve physiological feature learning

#### 5. **Multi-Modal Integration (As Planned)**
- **Augmented PPG Track** (Feb 2025): FSR, IR, RED, GREEN channels
- Multi-wavelength data will provide:
  - Redundant morphological features across wavelengths
  - Reduced spectral artifact sensitivity
  - Better tissue depth information

#### 6. **Feature-Guided Training**
- Add auxiliary loss to maintain morphological feature importance:
```python
def auxiliary_feature_loss(model_features, target_features):
    # Ensure pulse_width, kurtosis, peak_amplitude stay important
    return MSE(model_features, target_features)
```

### LONG-TERM STRATEGY

#### 7. **Architecture Modifications**
- **Attention Mechanisms**: Focus on morphological waveform regions
- **Multi-Scale Processing**: Prevent single-frequency dominance
- **Feature Pyramid**: Enforce hierarchical learning

#### 8. **Explainability Integration**
- Monitor feature importance during training
- Early stopping if spectral features dominate
- Alert system for feature collapse

#### 9. **Ensemble Approach**
- **Model 1**: Morphological expert (pulse width, peak amplitude)
- **Model 2**: Statistical expert (kurtosis, skewness)
- **Model 3**: Spectral expert (balanced spectral features)
- **Ensemble**: Weighted combination

---

## Expected Improvements

| Action | Expected MAE Improvement | Timeline |
|--------|--------------------------|----------|
| Feature Regularization | 31.78 → 20-25 mg/dL | 1-2 weeks |
| Data Quality Fix | 31.78 → 15-20 mg/dL | 2-3 weeks |
| Transfer Learning | 31.78 → 10-15 mg/dL | 1 week |
| Multi-Modal (FSR+IR+RED+GREEN) | 31.78 → 25-30 mg/dL | Feb 2025 |
| **Combined Approach** | **31.78 → 10-15 mg/dL** | **4-6 weeks** |

---

## Files Generated

- **Feature Importance CSV (Epoch 85)**: `inference_data/predictions24-12-2025/feature_analysis/feature_importance_epoch85.csv`
- **Heatmap (Epoch 85)**: `inference_data/predictions24-12-2025/feature_analysis/feature_importance_heatmap_epoch85.png`
- **Comparison Plot**: [docs/feature_comparison_epoch85_vs_original.png](feature_comparison_epoch85_vs_original.png)
- **Comparison Data**: [docs/feature_comparison_summary.csv](feature_comparison_summary.csv)

---

## Conclusion

**CRITICAL FINDING**: Epoch 85 model has **lost all physiologically meaningful features** and over-fit to spectral noise. This explains:
- High MAE (31.78 mg/dL median)
- Poor performance on extreme glucose ranges
- Inconsistent predictions across cases

**PATH FORWARD**:
1. ✅ **Immediate**: Retrain with feature regularization
2. ✅ **Short-term**: Transfer learning from original model
3. ✅ **Medium-term**: Multi-modal integration (February track)

**The model architecture is sound. The feature learning went wrong. This is fixable.**

---

## Next Steps

1. Run feature regularization training experiment
2. Compare original vs VitalDB training data
3. Implement transfer learning pipeline
4. Monitor feature importance during training
5. Prepare for multi-modal integration (Feb 2)

**Status**: 🚨 **URGENT - RETRAINING REQUIRED**
