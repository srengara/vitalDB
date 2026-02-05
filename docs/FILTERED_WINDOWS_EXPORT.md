# Filtered Windows CSV Export

## Overview

Added functionality to export filtered windows data as CSV files for further analysis. The web application now generates three types of CSV exports after peak detection.

## Available CSV Exports

### 1. Peaks CSV
**Filename**: `case_X_TRACKNAME_cleansed_peaks.csv`

Contains information about each detected peak.

**Columns**:
- `peak_index`: Index of the peak in the signal array
- `time`: Time of the peak (seconds)
- `amplitude_original`: Amplitude in original signal
- `amplitude_preprocessed`: Amplitude in preprocessed signal

**Example**:
```csv
peak_index,time,amplitude_original,amplitude_preprocessed
100,0.2000,145.2341,0.8542
600,1.2000,152.1234,0.9123
1100,2.2000,148.5432,0.8765
```

**Use Cases**:
- Heart rate analysis
- Peak-to-peak interval calculations
- Amplitude variability studies
- Time series analysis

---

### 2. Filtered Windows Summary CSV ⭐ NEW
**Filename**: `case_X_TRACKNAME_cleansed_filtered_windows.csv`

Contains summary statistics for each filtered window.

**Columns**:
- `window_index`: Index of the window (0, 1, 2, ...)
- `peak_index`: Index of the peak this window corresponds to
- `window_length`: Number of samples in the window
- `similarity_score`: Cosine similarity to template (0-1)
- `window_mean`: Mean amplitude of the window
- `window_std`: Standard deviation of the window
- `window_min`: Minimum amplitude in the window
- `window_max`: Maximum amplitude in the window

**Example**:
```csv
window_index,peak_index,window_length,similarity_score,window_mean,window_std,window_min,window_max
0,100,500,0.9234,0.4521,0.2341,0.0123,0.9876
1,600,500,0.9156,0.4634,0.2289,0.0234,0.9765
2,1100,500,0.9345,0.4589,0.2312,0.0156,0.9812
```

**Use Cases**:
- Quality assessment of detected beats
- Identifying windows with low similarity scores
- Statistical analysis of beat morphology
- Filtering beats based on similarity threshold
- Beat-to-beat variability analysis

**Analysis Examples**:

1. **Find low-quality beats**:
   ```python
   import pandas as pd
   df = pd.read_csv('filtered_windows.csv')
   low_quality = df[df['similarity_score'] < 0.90]
   print(f"Found {len(low_quality)} low-quality beats")
   ```

2. **Analyze amplitude variability**:
   ```python
   mean_amplitude = df['window_mean'].mean()
   std_amplitude = df['window_mean'].std()
   cv = std_amplitude / mean_amplitude  # Coefficient of variation
   ```

3. **Correlation between similarity and variability**:
   ```python
   correlation = df[['similarity_score', 'window_std']].corr()
   ```

---

### 3. Filtered Windows Detailed CSV ⭐ NEW
**Filename**: `case_X_TRACKNAME_cleansed_filtered_windows_detailed.csv`

Contains every sample of every filtered window in long format.

**Columns**:
- `window_index`: Index of the window (0, 1, 2, ...)
- `sample_index`: Index within the window (0 to window_length-1)
- `amplitude`: Amplitude value at this sample

**Example**:
```csv
window_index,sample_index,amplitude
0,0,0.1234
0,1,0.1456
0,2,0.1678
...
0,499,0.1234
1,0,0.1345
1,1,0.1567
...
```

**Use Cases**:
- Detailed waveform analysis
- Machine learning feature extraction
- Custom morphology analysis
- Waveform averaging
- Time-domain feature extraction
- Frequency-domain analysis (FFT)

**Analysis Examples**:

1. **Extract individual windows**:
   ```python
   import pandas as pd
   df = pd.read_csv('filtered_windows_detailed.csv')

   # Get specific window
   window_5 = df[df['window_index'] == 5]['amplitude'].values
   ```

2. **Compute average waveform** (same as template):
   ```python
   import numpy as np

   # Group by sample index and average
   avg_waveform = df.groupby('sample_index')['amplitude'].mean()
   ```

3. **Analyze beat morphology**:
   ```python
   # Find systolic peak location for each beat
   peaks = df.groupby('window_index')['amplitude'].idxmax()
   peak_locations = df.loc[peaks, 'sample_index'].values

   mean_peak_location = np.mean(peak_locations)
   std_peak_location = np.std(peak_locations)
   ```

4. **Frequency analysis**:
   ```python
   from scipy.fft import fft

   # FFT of first window
   window_0 = df[df['window_index'] == 0]['amplitude'].values
   fft_result = fft(window_0)
   ```

---

## File Sizes

For a typical case with 12,000 peaks and 500-sample windows:

| File | Approximate Size | Rows |
|------|------------------|------|
| Peaks CSV | 500 KB | 12,000 |
| Filtered Windows Summary | 600 KB | 12,000 |
| Filtered Windows Detailed | 240 MB | 6,000,000 |

**Note**: The detailed CSV can be very large. For 10,000 windows × 500 samples = 5 million rows.

---

## Implementation Details

### Backend (web_app.py)

#### 1. Windows Summary Generation
```python
windows_data = []
for i, window in enumerate(filtered_windows):
    similarity = cosine_similarity(window, template)
    windows_data.append({
        'window_index': i,
        'peak_index': peaks[i],
        'window_length': len(window),
        'similarity_score': round(float(similarity), 4),
        'window_mean': round(float(np.mean(window)), 4),
        'window_std': round(float(np.std(window)), 4),
        'window_min': round(float(np.min(window)), 4),
        'window_max': round(float(np.max(window)), 4)
    })
```

#### 2. Detailed Windows Generation
```python
detailed_data = []
for i, window in enumerate(filtered_windows):
    for sample_idx, value in enumerate(window):
        detailed_data.append({
            'window_index': i,
            'sample_index': sample_idx,
            'amplitude': round(float(value), 4)
        })
```

#### 3. Download Endpoint
```python
@app.route('/api/download_csv/<session_id>/<file_type>')
def download_csv(session_id, file_type):
    # Supports: raw, cleansed, peaks,
    #           filtered_windows, filtered_windows_detailed
```

### Frontend (index.html)

Three download links added:
```html
<a href="/api/download_csv/{sessionId}/peaks">Download Peaks CSV</a>
<a href="/api/download_csv/{sessionId}/filtered_windows">Download Filtered Windows Summary CSV</a>
<a href="/api/download_csv/{sessionId}/filtered_windows_detailed">Download Filtered Windows Detailed CSV</a>
```

---

## Usage Workflow

1. **Complete the analysis pipeline**:
   - Load case and track
   - Cleanse data
   - Detect peaks

2. **View results in web interface**:
   - Check statistics
   - Review plots
   - Verify filtering effectiveness

3. **Download appropriate CSV**:
   - **Peaks CSV**: For basic peak analysis
   - **Summary CSV**: For quality assessment and statistics
   - **Detailed CSV**: For waveform analysis and ML

4. **Perform offline analysis**:
   - Use Python, R, MATLAB, Excel, etc.
   - Apply custom algorithms
   - Generate reports

---

## Python Analysis Examples

### Example 1: Quality Report
```python
import pandas as pd
import numpy as np

# Load summary
df = pd.read_csv('case_1_SNUADC_PLETH_cleansed_filtered_windows.csv')

# Quality metrics
print("=== Quality Report ===")
print(f"Total beats: {len(df)}")
print(f"Mean similarity: {df['similarity_score'].mean():.4f}")
print(f"Min similarity: {df['similarity_score'].min():.4f}")
print(f"Beats < 0.90: {(df['similarity_score'] < 0.90).sum()}")
print(f"Beats < 0.85: {(df['similarity_score'] < 0.85).sum()}")

# Amplitude statistics
print(f"\nMean amplitude: {df['window_mean'].mean():.4f}")
print(f"Amplitude CV: {df['window_mean'].std() / df['window_mean'].mean():.4f}")
```

### Example 2: Waveform Analysis
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load detailed data
df = pd.read_csv('case_1_SNUADC_PLETH_cleansed_filtered_windows_detailed.csv')

# Plot first 10 beats
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(10):
    window = df[df['window_index'] == i]
    ax.plot(window['sample_index'], window['amplitude'], alpha=0.5)

# Plot average
avg = df.groupby('sample_index')['amplitude'].mean()
ax.plot(avg.index, avg.values, 'r-', linewidth=2, label='Average')
ax.legend()
ax.set_xlabel('Sample Index')
ax.set_ylabel('Amplitude')
ax.set_title('First 10 Filtered Beats')
plt.show()
```

### Example 3: Feature Extraction
```python
import pandas as pd
import numpy as np

# Load detailed data
df = pd.read_csv('case_1_SNUADC_PLETH_cleansed_filtered_windows_detailed.csv')

# Extract features for each beat
features = []
for window_idx in df['window_index'].unique():
    window = df[df['window_index'] == window_idx]['amplitude'].values

    # Time-domain features
    features.append({
        'window_index': window_idx,
        'peak_amplitude': np.max(window),
        'peak_location': np.argmax(window),
        'rise_time': np.argmax(window),  # Samples to peak
        'fall_time': len(window) - np.argmax(window),  # Samples from peak
        'area_under_curve': np.trapz(window),
        'skewness': pd.Series(window).skew(),
        'kurtosis': pd.Series(window).kurtosis()
    })

features_df = pd.DataFrame(features)
print(features_df.head())
```

---

## Best Practices

### 1. Summary CSV
- **Use for**: Quick analysis, quality checks, filtering
- **Fast to load**: Suitable for Excel, quick scripts
- **Good for**: Statistical analysis, outlier detection

### 2. Detailed CSV
- **Use for**: Deep waveform analysis, ML training
- **Large files**: Use chunked reading for big datasets
- **Good for**: Morphology studies, feature extraction

### 3. Combined Analysis
```python
# Load both files
summary = pd.read_csv('filtered_windows.csv')
detailed = pd.read_csv('filtered_windows_detailed.csv')

# Merge quality scores with waveforms
detailed = detailed.merge(
    summary[['window_index', 'similarity_score']],
    on='window_index'
)

# Analyze only high-quality beats
high_quality = detailed[detailed['similarity_score'] > 0.95]
```

---

## Limitations and Considerations

1. **Detailed CSV size**:
   - Can be hundreds of MB for long recordings
   - Consider using HDF5 or Parquet for large datasets
   - Use chunked reading if memory is limited

2. **Window index mapping**:
   - Window index may not equal peak index
   - Some peaks may have been filtered out during extraction
   - Use `peak_index` column to map back to original signal

3. **Precision**:
   - Values rounded to 4 decimal places
   - Sufficient for most analyses
   - Original precision available in session files

---

## Files Modified

1. **c:\IITM\vitalDB\web_app.py**:
   - Added filtered windows summary generation
   - Added detailed windows generation
   - Updated download endpoint
   - Save both CSV formats after peak detection

2. **c:\IITM\vitalDB\templates\index.html**:
   - Added two new download links
   - Updated download section UI

3. **c:\IITM\vitalDB\FILTERED_WINDOWS_EXPORT.md** (this file):
   - Complete documentation
   - Usage examples
   - Analysis recipes

---

## Summary

The web application now exports three CSV files:

1. ✅ **Peaks CSV** - Peak locations and amplitudes
2. ⭐ **Filtered Windows Summary CSV** - Statistics for each beat
3. ⭐ **Filtered Windows Detailed CSV** - Full waveform data

All files are automatically generated during peak detection and available for download from Step 4.
