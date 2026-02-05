# PPG Feature Importance Analysis - Complete Summary

## ✅ What's Been Created

I've built a comprehensive PPG feature analysis system that tells you **exactly which features** your ResNet34-1D model learns and their **importance scores (0-100)** across all layers.

## 📁 Files Created

1. **`analyze_ppg_features.py`** - Main analysis script (750+ lines)
2. **`install_analysis_deps.py`** - Dependency installer helper
3. **`FEATURE_ANALYSIS_GUIDE.md`** - Complete documentation
4. **`QUICK_START_ANALYSIS.md`** - Quick start guide

## 🎯 What It Does

### 1. Extracts 16 PPG Features

#### Morphological (8 features)
- `systolic_peak_amplitude` - Peak height
- `systolic_peak_location` - Peak timing (normalized)
- `rising_edge_slope` - Upstroke speed before peak
- `falling_edge_slope` - Downstroke speed after peak
- `pulse_width` - Duration at half maximum
- `dicrotic_notch_timing` - Secondary peak timing
- `dicrotic_notch_amplitude` - Secondary peak height
- `waveform_symmetry` - Shape balance (area before/after peak)

#### Energy & Statistical (5 features)
- `area_under_curve` - Total pulse energy
- `signal_energy` - Signal power (sum of squares)
- `zero_crossing_rate` - Oscillation frequency
- `skewness` - Distribution asymmetry
- `kurtosis` - Distribution peakedness

#### Spectral (3 features)
- `dominant_frequency` - Main frequency component
- `spectral_centroid` - Center of frequency spectrum
- `spectral_spread` - Frequency variability

### 2. Grad-CAM Visualizations

Shows **which temporal regions** of PPG waveforms are most important:
- Highlights important parts of the pulse (red/yellow = high importance)
- Overlays heatmap on actual PPG signals
- Analyzes Layer 4 (highest-level features)

### 3. Layer-wise Analysis

Correlates features with each ResNet layer:
- **Layer 1** (64 channels) → Basic waveform patterns
- **Layer 2** (128 channels) → Pulse shape characteristics
- **Layer 3** (256 channels) → Temporal patterns
- **Layer 4** (512 channels) → High-level abstractions

### 4. Feature Importance Scores

Computes **correlation-based importance (0-100)** for each feature:
- **70-100**: HIGH - Critical for predictions
- **40-69**: MEDIUM - Moderately important
- **0-39**: LOW - Minor influence

## 🚀 How to Use

### Quick Start (3 Steps)

```bash
# Step 1: Install dependencies (optional - seaborn for better plots)
pip install scipy seaborn

# Step 2: Run analysis
python analyze_ppg_features.py \
    --model_path training_outputs/training_XXXXX/best_model.pth \
    --data_dir training_data \
    --num_samples 100

# Step 3: Open the HTML report
# Look for: feature_analysis/ppg_feature_importance_report.html
```

## 📊 Expected Output

### Console Output Example:

```
================================================================================
PPG FEATURE IMPORTANCE ANALYSIS
================================================================================
Model: training_outputs/.../best_model.pth
Data: training_data
Output: model_analysis/feature_analysis
Samples: 100

Using device: cuda

Loading model...
Model loaded successfully

Loading data from training_data...
Loaded 100 samples
  PPG shape: (100, 250)
  Glucose range: 68.5 - 187.3 mg/dL

================================================================================
ANALYZING PPG FEATURE IMPORTANCE
================================================================================

1. Extracting PPG morphological features...
   Extracted 16 features from 100 windows

2. Computing layer-wise activations...
   Processed 100/100 samples
   Computed activations for 4 layers

3. Correlating PPG features with layer activations...
   Computed importance scores for 16 features

4. Generating Grad-CAM visualizations...
   Saved Grad-CAM visualizations to: gradcam_visualizations.png

5. Creating feature importance summary table...
   Saved feature importance table to: feature_importance.csv
   Saved feature importance heatmap to: feature_importance_heatmap.png

================================================================================
FEATURE IMPORTANCE SUMMARY (Top 15 Features)
================================================================================
                              layer1  layer2  layer3  layer4  overall_importance
spectral_centroid              85.2    89.3    92.1    88.7              88.8
systolic_peak_amplitude        82.1    84.5    79.3    81.2              81.8
dominant_frequency             78.9    80.1    82.3    84.6              81.5
signal_energy                  75.3    77.8    76.1    74.9              76.0
rising_edge_slope              71.2    73.4    68.9    70.1              70.9
...

6. Generating HTML report...
   Saved HTML report to: ppg_feature_importance_report.html

================================================================================
ANALYSIS COMPLETE!
================================================================================

Generated files in: model_analysis/feature_analysis
  - feature_importance.csv
  - feature_importance_heatmap.png
  - gradcam_visualizations.png
  - ppg_feature_importance_report.html

Open the HTML report in your browser for full analysis!
```

### Output Files:

1. **`feature_importance.csv`**
   ```csv
   Feature,layer1,layer2,layer3,layer4,overall_importance
   spectral_centroid,85.2,89.3,92.1,88.7,88.8
   systolic_peak_amplitude,82.1,84.5,79.3,81.2,81.8
   ...
   ```

2. **`feature_importance_heatmap.png`**
   - Visual heatmap showing top 15 features across layers
   - Color-coded (red = high importance, yellow = medium)

3. **`gradcam_visualizations.png`**
   - 5 sample PPG signals with Grad-CAM heatmaps
   - Shows which temporal regions model focuses on

4. **`ppg_feature_importance_report.html`**
   - **Interactive HTML report** with all results
   - Feature tables, visualizations, interpretations
   - Open in browser for best experience!

## 🔍 Interpreting Results

### Example Feature Importance Table:

| Rank | Feature Name | Layer1 | Layer2 | Layer3 | Layer4 | Overall | Category |
|------|-------------|--------|--------|--------|--------|---------|----------|
| 1 | spectral_centroid | 85.2 | 89.3 | 92.1 | 88.7 | 88.8 | **HIGH** |
| 2 | systolic_peak_amplitude | 82.1 | 84.5 | 79.3 | 81.2 | 81.8 | **HIGH** |
| 3 | dominant_frequency | 78.9 | 80.1 | 82.3 | 84.6 | 81.5 | **HIGH** |
| 4 | signal_energy | 75.3 | 77.8 | 76.1 | 74.9 | 76.0 | **HIGH** |
| 5 | rising_edge_slope | 71.2 | 73.4 | 68.9 | 70.1 | 70.9 | **HIGH** |

### What This Tells You:

1. **Top 5 Features are Critical** (all >70 overall importance)
2. **Spectral features dominate** (spectral_centroid, dominant_frequency)
3. **Morphological features important** (systolic_peak_amplitude, rising_edge_slope)
4. **Layer 3-4 rely on spectral** (higher scores in deeper layers)
5. **Layer 1-2 focus on morphology** (peak amplitude important early)

### Clinical Insights:

- **High spectral importance** → Model uses heart rate variability patterns
- **High peak amplitude** → Cardiac contractility correlates with glucose
- **High rising edge slope** → Arterial stiffness/resistance matters
- **Dicrotic notch features** → Arterial compliance affected by glucose

## 💡 Key Insights You'll Gain

### 1. Feature Hierarchy
- **Early layers (1-2)**: Basic pulse morphology (peaks, slopes, widths)
- **Deep layers (3-4)**: Complex patterns (spectral, temporal dynamics)

### 2. Most Important Features
You'll see exactly which features drive predictions:
- If spectral features rank high → Frequency analysis is key
- If morphological features rank high → Pulse shape is key
- If temporal features rank high → Timing patterns are key

### 3. Model Validation
- Check if important features make physiological sense
- Verify Grad-CAM focuses on relevant pulse regions
- Ensure model isn't learning artifacts

### 4. Quality Requirements
Know which PPG characteristics must be high-quality:
- If systolic peak is important → Need clear peaks
- If dicrotic notch is important → Need good waveform resolution
- If spectral features are important → Need stable pulse rate

## 🎓 What to Do Next

### 1. Run the Analysis
```bash
python analyze_ppg_features.py --model_path <your_model> --data_dir <your_data>
```

### 2. Review the HTML Report
Open `ppg_feature_importance_report.html` in browser

### 3. Identify Top Features
Note features with overall importance > 70

### 4. Validate Findings
- Do top features make physiological sense?
- Does Grad-CAM highlight reasonable pulse regions?
- Are layer patterns hierarchical (simple → complex)?

### 5. Apply Insights
- **Preprocessing**: Enhance high-importance features
- **Quality control**: Ensure clean signals for critical features
- **Feature engineering**: Consider computing top features explicitly
- **Data augmentation**: Preserve important features during augmentation

## ⚠️ Important Notes

1. **scipy is required** - Install with `pip install scipy`
2. **seaborn is optional** - Makes nicer heatmaps but not required
3. **Needs trained model** - Must have a `.pth` checkpoint file
4. **Needs training data** - Must have `ppg_windows.csv` and `glucose_labels.csv`
5. **Takes ~5-10 minutes** for 100 samples (GPU recommended)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'scipy'" | `pip install scipy` |
| "Model file not found" | Check path to `.pth` file |
| "Data files not found" | Verify `ppg_windows.csv` exists in data_dir |
| Out of memory | Reduce `--num_samples` to 50 or 20 |
| Low correlation scores | Model may need more training |

## 📚 Documentation

- **Full Guide**: [FEATURE_ANALYSIS_GUIDE.md](FEATURE_ANALYSIS_GUIDE.md)
- **Quick Start**: [QUICK_START_ANALYSIS.md](QUICK_START_ANALYSIS.md)

## ✨ Summary

You now have a complete system that:
- ✅ Extracts 16 PPG morphological/spectral features automatically
- ✅ Computes importance scores (0-100) for each feature
- ✅ Analyzes layer-by-layer what your model learns
- ✅ Generates Grad-CAM visualizations showing important pulse regions
- ✅ Creates comprehensive HTML report with all findings
- ✅ Provides actionable insights for improving your model

**Just run the script and open the HTML report to see exactly which PPG features your model uses and how important each one is!**
