# Web Application Template-Based Filtering Update

## Overview

Updated the VitalDB PPG Analysis web application to include template-based filtering with comprehensive visualization of each step in the pipeline.

## Major Changes

### 1. Backend Updates (web_app.py)

#### New Imports
```python
from peak_detection import (
    ppg_peak_detection_pipeline,
    ppg_peak_detection_pipeline_with_template,
    compute_template,
    filter_windows_by_similarity,
    cosine_similarity
)
```

#### New Plotting Functions

**`create_template_plot(template, sampling_rate)`**
- Creates a plot of the computed template waveform
- Shows the average PPG beat morphology
- Green line with fill for visual clarity
- Returns base64-encoded image

**`create_windows_comparison_plot(all_windows, filtered_windows, template, sampling_rate)`**
- Side-by-side comparison of all extracted windows vs filtered windows
- Left plot: All windows (up to 20 shown) with template overlay
- Right plot: Filtered windows (up to 20 shown) with template overlay
- Shows the effect of template-based filtering visually
- Returns base64-encoded image

#### Updated Peak Detection Endpoint

**Changes to `/api/detect_peaks`:**

1. **Now uses template-based pipeline:**
```python
peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
    ppg_signal=preprocessed_signal,
    fs=float(sampling_rate),
    window_duration=1.0,
    height_threshold=float(height_threshold),
    distance_threshold=distance_threshold,
    similarity_threshold=0.85
)
```

2. **Calculates additional statistics:**
- `num_all_windows`: Total windows extracted
- `num_filtered_windows`: Windows passing similarity threshold
- `filtering_rate`: Percentage of windows kept
- `similarity_scores`: Cosine similarity for each filtered window
- `mean_similarity`: Average similarity score
- `min_similarity`: Minimum similarity score

3. **Generates new plots:**
- Template plot showing the average beat
- Windows comparison plot showing filtering effect

4. **Enhanced JSON response:**
```json
{
    "total_peaks": 12450,
    "mean_hr": 65.2,
    "std_hr": 3.5,
    "mean_interval": 0.92,
    "std_interval": 0.05,
    "height_threshold": 45.67,
    "distance_threshold": 400.0,
    "height_multiplier": 0.3,
    "distance_multiplier": 0.8,
    "num_all_windows": 12450,           // NEW
    "num_filtered_windows": 12100,      // NEW
    "filtering_rate": 97.2,             // NEW
    "template_length": 500,             // NEW
    "mean_similarity": 0.9234,          // NEW
    "min_similarity": 0.8501,           // NEW
    "preview_data": [...],
    "plot_original": "data:image/png;base64,...",
    "plot_preprocessed": "data:image/png;base64,...",
    "plot_template": "data:image/png;base64,...",              // NEW
    "plot_windows_comparison": "data:image/png;base64,..."     // NEW
}
```

### 2. Frontend Updates (templates/index.html)

#### Enhanced Statistics Display

Added three new stat cards in Step 4:

1. **Extracted Windows**
   - Shows total number of windows extracted
   - Indicates how many potential beats were found

2. **Filtered Windows**
   - Shows number of windows after filtering
   - Displays filtering rate percentage
   - Example: "12100 (97.2% kept)"

3. **Mean Similarity**
   - Shows average cosine similarity score
   - Displays minimum similarity
   - Example: "0.923 (min: 0.850)"

#### Reorganized Step 4 Layout

**New structure with sections:**

1. **Signal Plots with Detected Peaks**
   - Original Signal with Peaks
   - Preprocessed Signal with Peaks

2. **Template Analysis** (NEW)
   - Computed Template (Average Beat)
   - Windows Comparison (All vs Filtered)

3. **Peaks Data**
   - Data table preview
   - CSV download link

#### Enhanced Visual Organization

- Added section headers with color coding
- Clearer separation between different analysis stages
- Better visual hierarchy

## Pipeline Steps Explained

### Step 1: Peak Detection
- Detects peaks using height and distance thresholds
- Controlled by user via Height Multiplier and Distance Multiplier
- Output: List of peak indices

### Step 2: Window Extraction
- Extracts windows around each detected peak
- Window size: sampling_rate × window_duration (default: 500 samples)
- Window boundaries: [peak - size/2, peak + size/2]
- Only keeps windows with exactly 1 peak
- Output: List of extracted windows

### Step 3: Template Computation
- Computes average of all extracted windows
- Creates a representative "typical" PPG beat
- Handles variable window lengths
- Output: Template waveform

### Step 4: Similarity Filtering
- Computes cosine similarity between each window and template
- Keeps only windows with similarity ≥ 0.85
- Removes artifacts, noise, and irregular beats
- Output: Filtered windows list

### Final Output
- Peaks with corresponding filtered windows
- Statistics on filtering effectiveness
- Visualizations at each stage

## Visual Outputs

### 1. Original Signal with Peaks
- Shows first 30 seconds of raw signal
- Red dots mark detected peaks
- Helps verify peak detection quality

### 2. Preprocessed Signal with Peaks
- Shows first 30 seconds of preprocessed signal
- Red dots mark detected peaks
- Shows effect of DC removal, filtering, and smoothing

### 3. Computed Template ★ NEW
- Single waveform showing average beat
- Green line with shaded area
- Represents the "ideal" PPG beat for this signal
- Useful for:
  - Assessing signal quality
  - Identifying morphological features
  - Comparing across different recordings

### 4. Windows Comparison ★ NEW
- **Left panel**: All extracted windows
  - Blue lines (semi-transparent)
  - Shows all beats before filtering
  - Red template overlay
  - Visualizes variability

- **Right panel**: Filtered windows
  - Green lines (semi-transparent)
  - Shows only beats matching template
  - Red template overlay
  - Demonstrates filtering effectiveness

## Benefits of Template-Based Approach

### 1. Quality Control
- Automatically removes poor quality beats
- Ensures consistent morphology
- Reduces manual review time

### 2. Artifact Removal
- Filters out motion artifacts
- Removes ectopic beats
- Eliminates noise-corrupted segments

### 3. Reproducibility
- Consistent criteria across recordings
- Quantifiable (similarity scores)
- Adjustable threshold (default: 0.85)

### 4. Visual Validation
- Users can see which beats were filtered
- Template shows expected morphology
- Easy to identify if filtering is too aggressive or lenient

## Usage Example

### 1. Start the application:
```bash
python web_app.py
```

### 2. Navigate to http://localhost:5000

### 3. Complete the workflow:
1. **Step 1**: Enter case ID (e.g., 1), load tracks, download data
2. **Step 2**: Review raw data, click "Cleanse Data"
3. **Step 3**:
   - Adjust Height Multiplier (default: 0.3)
   - Adjust Distance Multiplier (default: 0.8)
   - Click "Detect Peaks"
4. **Step 4**: View results:
   - Check statistics (peaks, HR, filtering rate)
   - Review signal plots with peaks
   - Examine template waveform
   - Compare all windows vs filtered windows
   - Download peaks CSV

### 4. Interpret Results

**Good Results:**
- Filtering rate: 85-98%
- Mean similarity: > 0.90
- Template: Clear PPG waveform shape
- Windows comparison: Similar shapes in both panels

**Poor Results (consider adjusting parameters):**
- Filtering rate: < 70% (too aggressive) or > 99% (not filtering)
- Mean similarity: < 0.88 (inconsistent beats)
- Template: Noisy or unclear shape
- Windows comparison: High variability in filtered panel

## Parameter Tuning Guide

### Height Multiplier
- **Increase (0.5-1.0)**: Detect only prominent peaks
  - Use for: Noisy signals, low amplitude variations
  - Effect: Fewer peaks detected, higher confidence

- **Decrease (0.1-0.2)**: Detect subtle peaks
  - Use for: Clean signals, subtle variations
  - Effect: More peaks detected, may include false positives

### Distance Multiplier
- **Increase (1.0-1.5)**: Require more spacing
  - Use for: Signals with double peaks, artifacts
  - Effect: Prevents detecting multiple peaks per heartbeat

- **Decrease (0.5-0.6)**: Allow closer peaks
  - Use for: High heart rates (>100 bpm)
  - Effect: Enables detection of rapid beats

## Technical Details

### Cosine Similarity Calculation
```
similarity = dot_product / (magnitude_window × magnitude_template)
```
- Range: -1 to 1 (typically 0 to 1 for PPG)
- Threshold: 0.85 (fixed in current implementation)
- Measures shape matching, not amplitude

### Window Extraction
- Centered on peak: `[peak - window_size/2, peak + window_size/2]`
- Default window: 1 second × sampling rate
- Example: 500 Hz → 500 samples (250 before, 250 after)

### Template Computation
- Method: Element-wise mean of all windows
- Handles variable lengths: Uses most common length
- Result: Single waveform of same length as windows

## Future Enhancements

### Potential Additions:
1. **Adjustable Similarity Threshold**
   - Add UI slider for threshold (0.7-0.95)
   - Allow user to control filtering aggressiveness

2. **Multiple Templates**
   - Detect different beat types
   - K-means clustering of windows
   - Classify beats by template match

3. **Iterative Refinement**
   - Re-compute template from filtered windows
   - Apply second filtering pass
   - Improve template quality

4. **Feature Extraction**
   - Extract template features (rise time, decay time, area)
   - Display morphological parameters
   - Export features to CSV

5. **Beat Classification**
   - Normal vs. abnormal beats
   - Color-code windows by similarity
   - Interactive window selection

6. **Real-time Updates**
   - Adjust parameters without re-detecting peaks
   - Live preview of filtering effect
   - Interactive threshold adjustment

## Files Modified

1. **c:\IITM\vitalDB\web_app.py**
   - Added template-based pipeline integration
   - Created template plot function
   - Created windows comparison plot function
   - Enhanced detect_peaks endpoint
   - Added similarity calculations

2. **c:\IITM\vitalDB\templates\index.html**
   - Added 3 new statistics cards
   - Added template plot display
   - Added windows comparison plot display
   - Reorganized Step 4 layout with sections
   - Enhanced visual organization

3. **c:\IITM\vitalDB\peak_detection.py** (already extended)
   - compute_template()
   - cosine_similarity()
   - filter_windows_by_similarity()
   - ppg_peak_detection_pipeline_with_template()

## Testing

The application is ready to use. Restart the Flask server if already running:

```bash
# Stop current server (Ctrl+C)
python web_app.py
# Open http://localhost:5000
```

Test with known good cases:
- Case 1: SNUADC/PLETH
- Case 2: SNUADC/PLETH

---

**Update Complete**: The web application now provides comprehensive visualization of all pipeline steps with template-based filtering and quality metrics.
