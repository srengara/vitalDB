# Quick Start: PPG Feature Analysis

## Step 1: Install Dependencies

Run this to install required packages (scipy is required, seaborn is optional):

```bash
python install_analysis_deps.py
```

Or install manually:

```bash
pip install scipy seaborn
```

**Note:** The script will work without seaborn, but seaborn makes nicer heatmaps.

## Step 2: Run the Analysis

Basic usage:

```bash
python analyze_ppg_features.py \
    --model_path path/to/best_model.pth \
    --data_dir path/to/training_data \
    --num_samples 100
```

### Example with Your Project Structure:

```bash
python analyze_ppg_features.py \
    --model_path training_outputs/training_20241216_120000/best_model.pth \
    --data_dir training_data \
    --num_samples 100
```

## Step 3: View Results

The script creates an `feature_analysis` folder with:

1. **`ppg_feature_importance_report.html`** ← Open this in your browser!
2. **`feature_importance.csv`** ← Feature scores table
3. **`feature_importance_heatmap.png`** ← Visual heatmap
4. **`gradcam_visualizations.png`** ← Grad-CAM on PPG signals

## What You'll Get

### Feature Importance Table (CSV)

```
Feature Name                  Layer1  Layer2  Layer3  Layer4  Overall
spectral_centroid              85.2    89.3    92.1    88.7    88.8
systolic_peak_amplitude        82.1    84.5    79.3    81.2    81.8
dominant_frequency             78.9    80.1    82.3    84.6    81.5
...
```

### Interpretation:

- **70-100**: HIGH importance - critical feature
- **40-69**: MEDIUM importance - moderately important
- **0-39**: LOW importance - minor feature

## Troubleshooting

### Error: "No module named 'scipy'"

```bash
pip install scipy
```

### Error: "Model file not found"

Check your model path. It should be the `.pth` file from training, usually in:
```
training_outputs/training_XXXXXXXX_XXXXXX/best_model.pth
```

### Error: "Data files not found"

Make sure your `--data_dir` contains:
- `ppg_windows.csv`
- `glucose_labels.csv`

### Out of Memory

Reduce the number of samples:

```bash
python analyze_ppg_features.py \
    --model_path path/to/model.pth \
    --data_dir path/to/data \
    --num_samples 50  # Reduced from 100
```

## Understanding the Results

### Top Features to Look For:

1. **Spectral Features** (spectral_centroid, dominant_frequency)
   - High importance = model uses frequency information
   - Related to heart rate variability and metabolic state

2. **Morphological Features** (systolic_peak_amplitude, pulse_width)
   - High importance = model uses pulse shape
   - Related to cardiac output and vascular resistance

3. **Temporal Features** (peak timing, dicrotic notch)
   - High importance = model uses timing patterns
   - Related to arterial compliance and wave reflections

### Layer Analysis:

- **High in Layer 1-2**: Basic waveform features (peaks, slopes)
- **High in Layer 3-4**: Complex patterns (spectral, temporal)
- **Consistent across all**: Universally important feature

## Next Steps

1. Review the HTML report (best visualization)
2. Note features with >70 importance score
3. Check if important features make physiological sense
4. Improve PPG signal quality for high-importance features
5. Consider feature engineering based on findings

## Questions?

See the full guide: [FEATURE_ANALYSIS_GUIDE.md](FEATURE_ANALYSIS_GUIDE.md)
