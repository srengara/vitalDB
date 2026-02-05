# PPG Feature Importance Analysis Guide

## Overview

The `analyze_ppg_features.py` script analyzes what PPG waveform features your ResNet34-1D model learns and their importance across different layers.

## What It Does

### 1. **PPG Feature Extraction** (16 Features)
Extracts morphological, temporal, energy, and spectral features from PPG waveforms:

#### Morphological Features
- `systolic_peak_amplitude` - Height of main peak
- `systolic_peak_location` - Timing of peak (normalized)
- `rising_edge_slope` - Slope before peak
- `falling_edge_slope` - Slope after peak
- `pulse_width` - Duration at half maximum
- `dicrotic_notch_timing` - Secondary peak timing
- `dicrotic_notch_amplitude` - Secondary peak height
- `waveform_symmetry` - Balance of rising/falling phases

#### Energy & Statistical Features
- `area_under_curve` - Total pulse energy
- `signal_energy` - Signal power
- `zero_crossing_rate` - Oscillation frequency
- `skewness` - Distribution asymmetry
- `kurtosis` - Distribution tailedness

#### Spectral Features
- `dominant_frequency` - Primary frequency component
- `spectral_peak_power` - Power at dominant frequency
- `spectral_centroid` - Center of frequency spectrum
- `spectral_spread` - Frequency variability

### 2. **Grad-CAM Visualization**
- Highlights temporal regions of PPG waveform most important for prediction
- Shows which parts of the pulse the model focuses on
- Generates heatmaps overlaid on PPG signals

### 3. **Layer-wise Activation Analysis**
- Analyzes what each ResNet layer learns:
  - **Layer 1** (64 channels): Basic waveform patterns
  - **Layer 2** (128 channels): Pulse shape characteristics
  - **Layer 3** (256 channels): Temporal patterns
  - **Layer 4** (512 channels): High-level physiological abstractions

### 4. **Feature Importance Correlation**
- Correlates extracted PPG features with layer activations
- Computes importance scores (0-100) for each feature
- Shows which features each layer focuses on

## Usage

### Basic Usage

```bash
python analyze_ppg_features.py \
    --model_path path/to/best_model.pth \
    --data_dir path/to/training_data \
    --num_samples 100
```

### Example with Your Setup

```bash
python analyze_ppg_features.py \
    --model_path training_outputs/training_20241216_120000/best_model.pth \
    --data_dir training_data \
    --num_samples 100 \
    --output_dir model_analysis/feature_importance
```

## Output Files

The script generates:

1. **`feature_importance.csv`** - Feature importance scores across all layers
2. **`feature_importance_heatmap.png`** - Visual heatmap of top 15 features
3. **`gradcam_visualizations.png`** - Grad-CAM heatmaps on sample PPG signals
4. **`ppg_feature_importance_report.html`** - Comprehensive interactive report

## Interpreting Results

### Feature Importance Scores (0-100)

| Score Range | Interpretation |
|------------|----------------|
| 70-100 | **HIGH** - Critical feature, strongly correlated with model predictions |
| 40-69 | **MEDIUM** - Important feature, moderately influences predictions |
| 0-39 | **LOW** - Minor feature, weak influence on predictions |

### Layer-wise Interpretation

- **High importance in Layer 1-2**: Model relies on basic morphological features
- **High importance in Layer 3-4**: Model relies on complex temporal/spectral patterns
- **Consistent across all layers**: Feature is universally important

### Grad-CAM Heatmap Colors

- **Red/Yellow (Hot colors)**: High importance regions
- **Blue/Purple (Cool colors)**: Low importance regions

Look for:
- Peak regions (usually high importance)
- Dicrotic notch area (moderate importance)
- Rising/falling edges (variable importance)

## Example Output

The analysis will show you something like:

```
FEATURE IMPORTANCE SUMMARY (Top 15 Features)
================================================================================
                              layer1  layer2  layer3  layer4  overall_importance
spectral_centroid              85.2    89.3    92.1    88.7              88.8
systolic_peak_amplitude        82.1    84.5    79.3    81.2              81.8
dominant_frequency             78.9    80.1    82.3    84.6              81.5
signal_energy                  75.3    77.8    76.1    74.9              76.0
rising_edge_slope              71.2    73.4    68.9    70.1              70.9
area_under_curve               68.5    69.7    67.2    66.8              68.1
pulse_width                    65.3    67.1    64.8    63.2              65.1
spectral_spread                62.1    64.5    66.7    68.9              65.6
...
```

## Understanding the Results

### What the Model Learns

Based on feature importance scores, you'll understand:

1. **Which PPG characteristics matter most** for glucose prediction
2. **How the model's understanding evolves** from layer to layer
3. **Where to focus preprocessing efforts** (enhance high-importance features)
4. **Quality requirements for PPG signals** (e.g., need clear systolic peaks)

### Clinical Insights

High importance of specific features can reveal physiological relationships:

- **Spectral features** → Pulse rate variability relates to metabolic state
- **Peak amplitude** → Cardiac contractility correlates with glucose
- **Dicrotic notch** → Arterial compliance affected by glucose levels
- **Pulse width** → Vascular resistance changes with metabolism

## Troubleshooting

### Common Issues

**Issue**: "Model file not found"
- **Solution**: Check model path, ensure training has completed

**Issue**: "Data files not found"
- **Solution**: Verify data_dir contains `ppg_windows.csv` and `glucose_labels.csv`

**Issue**: Low correlation scores across all features
- **Solution**: Model may not be trained well, or PPG quality is poor

**Issue**: Out of memory
- **Solution**: Reduce `--num_samples` (try 50 or 20)

## Advanced Usage

### Analyze Specific Layers Only

Modify the script to focus on specific layers:

```python
# In the main() function, change:
gradcam_layer3 = GradCAM1D(model, 'layer3')  # Focus on layer 3
```

### Export Feature Values

Access raw feature values for further analysis:

```python
# After running the script, load the CSV
import pandas as pd
features = pd.read_csv('feature_importance.csv')
print(features[features['overall_importance'] > 70])  # High-importance features only
```

## Next Steps

After running the analysis:

1. **Review the HTML report** - Open in browser for interactive exploration
2. **Identify key features** - Note features with scores > 70
3. **Validate with domain knowledge** - Do important features make physiological sense?
4. **Improve preprocessing** - Enhance signal quality for high-importance features
5. **Feature selection** - Consider using only high-importance features for efficiency

## Questions?

If the analysis reveals unexpected results:
- Check PPG signal quality (clean peaks, minimal noise)
- Verify model is well-trained (low validation loss)
- Ensure sufficient samples (100+ recommended)
- Review Grad-CAM heatmaps for physiological plausibility
