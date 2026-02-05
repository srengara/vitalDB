# Peak Detection Extended Implementation

## Overview

Extended the `peak_detection.py` module to include template-based filtering functions as specified in the pseudocode (PPg_Pipeline_2.jpg).

## New Functions Added

### 1. `compute_template(windows)`

Computes a template by averaging all windows.

**Algorithm** (lines 31-34 from pseudocode):
```
function COMPUTETEMPLATE(windows)
    template ← mean(windows)
    return template
end function
```

**Implementation**:
- Finds the most common window length
- Filters windows to only include those with the most common length
- Stacks windows and computes mean along axis 0
- Returns the average template

**Parameters**:
- `windows`: List[np.ndarray] - List of signal windows extracted around peaks

**Returns**:
- `template`: np.ndarray - Average template computed from all windows

**Example**:
```python
windows = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([3, 4, 5])]
template = compute_template(windows)
# template = [2. 3. 4.]
```

---

### 2. `cosine_similarity(window, template)`

Computes cosine similarity between a window and template.

**Algorithm** (lines 35-40 from pseudocode):
```
function COSINESIMILARITY(window, template)
    dot_product ← sum(window × template)
    magnitude_window ← sqrt(sum(window²))
    magnitude_template ← sqrt(sum(template²))
    return dot_product / (magnitude_window × magnitude_template)
end function
```

**Implementation**:
- Ensures windows have the same length (truncates if needed)
- Computes dot product: sum(window × template)
- Computes magnitudes: sqrt(sum(window²)) and sqrt(sum(template²))
- Returns cosine similarity value between 0 and 1

**Parameters**:
- `window`: np.ndarray - Signal window to compare
- `template`: np.ndarray - Template signal to compare against

**Returns**:
- `similarity`: float - Cosine similarity value (0 to 1, where 1 = identical shape)

**Example**:
```python
window = np.array([1, 2, 3, 4])
template = np.array([1, 2, 3, 4])
similarity = cosine_similarity(window, template)
# similarity = 1.000 (identical)

window = np.array([1, 2, 3, 4])
template = np.array([4, 3, 2, 1])
similarity = cosine_similarity(window, template)
# similarity = 0.400 (different shape)
```

---

### 3. `filter_windows_by_similarity(windows, template, similarity_threshold)`

Filters windows by cosine similarity to template.

**Algorithm** (lines 41-50 from pseudocode):
```
function FILTERWINDOWSBYSIMILARITY(windows, template, similarity_threshold)
    filtered_windows ← []
    for each window in windows do
        similarity ← COSINESIMILARITY(window, template)
        if similarity ≥ similarity_threshold then
            filtered_windows.append(window)
        end if
    end for
    return filtered_windows
end function
```

**Implementation**:
- Iterates through each window
- Computes cosine similarity with template
- Keeps only windows that meet the similarity threshold
- Returns list of filtered windows

**Parameters**:
- `windows`: List[np.ndarray] - List of signal windows to filter
- `template`: np.ndarray - Template signal to compare against
- `similarity_threshold`: float - Minimum cosine similarity threshold (default: 0.85)

**Returns**:
- `filtered_windows`: List[np.ndarray] - List of windows that meet the similarity threshold

**Example**:
```python
windows = [
    np.array([1, 2, 3, 2, 1]),  # Matches template
    np.array([1, 2, 3, 2, 1]),  # Matches template
    np.array([5, 1, 2, 1, 5])   # Different from template
]
template = np.array([1, 2, 3, 2, 1])
filtered = filter_windows_by_similarity(windows, template, 0.9)
# Result: 2 out of 3 windows kept
```

---

### 4. `ppg_peak_detection_pipeline_with_template()`

Complete PPG peak detection pipeline WITH template-based filtering.

**Algorithm** (lines 51-57 from pseudocode):
```
function MAIN(ppg_signal)
    peaks ← DETECTPEAKS(ppg_signal, height_threshold, distance_threshold)
    windows ← EXTRACTWINDOWS(ppg_signal, peaks, window_size)
    template ← COMPUTETEMPLATE(windows)
    filtered_windows ← FILTERWINDOWSBYSIMILARITY(windows, template, similarity_threshold)
    return filtered_windows
end function
```

**Implementation**:
1. Detect peaks using height and distance thresholds
2. Extract windows around detected peaks
3. Compute template by averaging all windows
4. Filter windows by similarity to template
5. Return filtered windows that match the template

**Parameters**:
- `ppg_signal`: np.ndarray - Input PPG signal
- `fs`: float - Sampling frequency in Hz (default: 100)
- `window_duration`: float - Window duration in seconds (default: 1)
- `height_threshold`: float - Minimum peak height (default: 20)
- `distance_threshold`: float - Minimum distance between peaks in samples (default: 0.8 * fs)
- `similarity_threshold`: float - Cosine similarity threshold for filtering (default: 0.85)

**Returns**:
- `peaks`: List[int] - Detected peak indices
- `filtered_windows`: List[np.ndarray] - Filtered windows that match the template
- `template`: np.ndarray - Computed template (average of all windows)
- `all_windows`: List[np.ndarray] - All extracted windows (before filtering)

**Example**:
```python
peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
    ppg_signal,
    fs=500,
    window_duration=1.0,
    height_threshold=45.0,
    distance_threshold=400.0,
    similarity_threshold=0.85
)

print(f"Detected {len(peaks)} peaks")
print(f"Extracted {len(all_windows)} windows")
print(f"Filtered to {len(filtered_windows)} windows (similarity ≥ 0.85)")
print(f"Template shape: {template.shape}")
```

---

## Pipeline Comparison

### Basic Pipeline (`ppg_peak_detection_pipeline`)
- Detects peaks
- Extracts windows
- Returns peaks and windows
- **Use case**: Quick peak detection without quality filtering

### Template-Based Pipeline (`ppg_peak_detection_pipeline_with_template`)
- Detects peaks
- Extracts windows
- **Computes template from all windows**
- **Filters windows by similarity to template**
- Returns peaks, filtered windows, template, and all windows
- **Use case**: High-quality peak detection with artifact removal

---

## Benefits of Template-Based Filtering

1. **Artifact Removal**: Filters out irregular beats, noise artifacts, and motion artifacts
2. **Quality Control**: Ensures only beats matching the typical morphology are kept
3. **Consistency**: Creates a uniform set of beats for further analysis
4. **Robustness**: Handles noisy signals better by rejecting outliers

---

## Integration with Web Application

The web application (`web_app.py`) currently uses the basic `ppg_peak_detection_pipeline()`. To enable template-based filtering:

### Option 1: Replace with Template-Based Pipeline

```python
# In web_app.py detect_peaks() function
peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
    ppg_signal=preprocessed_signal,
    fs=float(sampling_rate),
    window_duration=1.0,
    height_threshold=float(height_threshold),
    distance_threshold=distance_threshold,
    similarity_threshold=0.85
)
```

### Option 2: Add UI Control for Template Filtering

Add a checkbox in the UI to enable/disable template-based filtering:

```html
<div class="form-group">
    <label>
        <input type="checkbox" id="useTemplateFiltering" checked>
        Use Template-Based Filtering
    </label>
</div>
```

Then in JavaScript:
```javascript
const useTemplateFiltering = document.getElementById('useTemplateFiltering').checked;
```

And send this parameter to the backend.

---

## Testing

Run the test script to see both pipelines in action:

```bash
python peak_detection.py
```

This will run:
1. **Test 1**: Basic Peak Detection Pipeline
2. **Test 2**: Template-Based Peak Detection Pipeline

The output shows:
- Number of peaks detected
- Number of windows extracted
- Number of windows after filtering
- Template shape
- Filtering rate (% of windows kept)
- Similarity scores for filtered windows

---

## Technical Notes

### Window Extraction

The extract_windows function:
- Centers window on peak: `[peak - window_size//2, peak + window_size//2]`
- Only keeps windows with exactly 1 peak (prevents overlapping peaks)
- Window size = sampling_rate × window_duration (e.g., 500 Hz × 1s = 500 samples)

### Cosine Similarity

- Value ranges from -1 to 1 (typically 0 to 1 for PPG signals)
- 1.0 = identical shape
- 0.85 = 85% similar (typical threshold for PPG)
- < 0.85 = different morphology (likely artifact)

### Adaptive Threshold in count_peaks

The `count_peaks()` function now uses adaptive thresholding:
- If no threshold provided, uses median of window
- This makes it more robust to different signal amplitudes
- Helps extract windows from signals with varying baseline

---

## Files Modified

1. **peak_detection.py**:
   - Added `compute_template()`
   - Added `cosine_similarity()`
   - Added `filter_windows_by_similarity()`
   - Added `ppg_peak_detection_pipeline_with_template()`
   - Updated `count_peaks()` with adaptive threshold
   - Enhanced test section to demonstrate both pipelines

2. **PEAK_DETECTION_EXTENDED.md** (this file):
   - Complete documentation of new functions
   - Usage examples
   - Integration guide

---

## Future Enhancements

1. **Multi-Template Approach**: Compute multiple templates for different beat types
2. **Iterative Refinement**: Re-compute template after filtering, then filter again
3. **Morphology Features**: Extract features from template (rise time, decay time, etc.)
4. **Beat Classification**: Classify beats based on similarity scores
5. **Anomaly Detection**: Identify and flag beats with very low similarity scores

---

## References

- Pseudocode: `PPg_Pipeline_2.jpg` (lines 31-57)
- Original implementation: `peak_detection.py`
- Web application: `web_app.py`
- Preprocessing: `ppg_segmentation.py`
