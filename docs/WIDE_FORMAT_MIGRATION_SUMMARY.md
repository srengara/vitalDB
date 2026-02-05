# Wide Format Migration Summary

## Overview
Successfully migrated the PPG data pipeline from **long format** (vertical stacking) to **wide format** (horizontal row layout) for improved performance.

## Changes Made

### 1. Data Format Conversion (ppg_windows_h.py)
**Status**: ✓ Complete

**Input Format (Long)**:
```
window_index,sample_index,amplitude
0,0,4.942
0,1,4.6729
0,2,4.4074
...
```

**Output Format (Wide)**:
```
window_index,case_id,window_id,amplitude_sample_0,amplitude_sample_1,...,amplitude_sample_499
180800,1,0,4.942,4.6729,4.4074,...
```

**Location**: [new_data_format/final_datasets/](new_data_format/final_datasets/)
- ppg_windows.csv (46 MB, 11,913 windows × 500 samples)
- glucose_labels.csv (281 KB, 11,913 labels)

### 2. Updated test_model_with_normalization.py
**Status**: ✓ Complete

**Modified Function**: `load_test_data()` at [test_model_with_normalization.py:24](test_model_with_normalization.py#L24)

**Key Changes**:
- Added automatic format detection (wide vs long)
- Wide format: Extracts `amplitude_sample_*` columns directly
- Long format: Still supported for backward compatibility
- Proper column sorting by sample number

**Code Logic**:
```python
if 'sample_index' in ppg_df.columns:
    # Old long format - group by window_index
    windows = []
    for window_idx in sorted(ppg_df['window_index'].unique()):
        window_df = ppg_df[ppg_df['window_index'] == window_idx].sort_values('sample_index')
        window = window_df['amplitude'].values
        windows.append(window)
    ppg_data = np.array(windows)
else:
    # New wide format - direct extraction
    amplitude_cols = [col for col in ppg_df.columns if col.startswith('amplitude_sample_')]
    amplitude_cols_sorted = sorted(amplitude_cols, key=lambda x: int(x.split('_')[-1]))
    ppg_data = ppg_df[amplitude_cols_sorted].values
```

## Verification Results

### Data Loading Test
```
PPG DataFrame Info:
  Shape: (11913, 503)
  Format: Wide format (new)
  Amplitude columns: 500

PPG Data Array:
  Shape: (11913, 500)
  First window: [4.942, 4.6729, 4.4074, 4.1455, 3.8874, ...]

Glucose Labels:
  Shape: (11913, 4)
  Columns: ['case_id', 'window_id', 'glucose_mg_dl', 'window_index']
  Range: 134.0 - 134.0 mg/dL

[OK] Data loading successful!
PPG windows: 11913, Glucose labels: 11913
```

## Performance Benefits

Based on [DATA_FORMAT_ANALYSIS.md](docs/DATA_FORMAT_ANALYSIS.md):

| Metric | Long Format | Wide Format | Improvement |
|--------|-------------|-------------|-------------|
| Load Time | ~200ms | ~13ms | **15x faster** |
| Memory | Higher | Lower | More efficient |
| Code Complexity | High (groupby/pivot) | Low (direct array) | Simpler |

## Files Modified

1. **[test_model_with_normalization.py](test_model_with_normalization.py)** - Updated `load_test_data()` function
2. **[new_data_format/case_1/ppg_windows.csv](new_data_format/case_1/ppg_windows.csv)** - Added `case_id` column
3. **[new_data_format/case_1/glucose_labels.csv](new_data_format/case_1/glucose_labels.csv)** - Added `case_id` column

## Usage

### Run Model Testing (New Format)
```bash
python test_model_with_normalization.py --test_data new_data_format/final_datasets
```

### Run Model Testing (Old Format - Still Supported)
```bash
python test_model_with_normalization.py --test_data data/web_app_data/case_1_SNUADC_PLETH
```

The script automatically detects which format is being used.

## Data Specifications

### PPG Windows (Wide Format)
- **Rows**: 11,913 windows
- **Columns**: 503 total
  - `window_index`: Combined index (case_id + "8080" + window_id)
  - `case_id`: Case identifier (e.g., 1)
  - `window_id`: Window number within case (0, 1, 2, ...)
  - `amplitude_sample_0` to `amplitude_sample_499`: 500 PPG samples per window

### Glucose Labels
- **Rows**: 11,913 labels (matches PPG windows)
- **Columns**: 4 total
  - `case_id`: Case identifier
  - `window_id`: Window number within case
  - `glucose_mg_dl`: Glucose value in mg/dL
  - `window_index`: Combined index (matches PPG)

## Next Steps

### For Multiple Cases
When processing additional cases, the combined datasets will have:
- Multiple `case_id` values (1, 2, 3, ...)
- Varying glucose values across windows
- Same wide format structure

Example with multiple cases:
```
window_index,case_id,window_id,amplitude_sample_0,...
180800,1,0,4.942,...      # Case 1, Window 0
180801,1,1,5.123,...      # Case 1, Window 1
280800,2,0,3.856,...      # Case 2, Window 0
280801,2,1,4.021,...      # Case 2, Window 1
```

### For Model Training
The wide format is now ready for use with:
```bash
python -m src.training.train_glucose_predictor --data_dir new_data_format/final_datasets
```

## Backward Compatibility

The updated `load_test_data()` function maintains **full backward compatibility**:
- Detects format automatically
- No changes needed to existing code using old format
- Seamless migration path

## Summary

✓ **Data converted** from long to wide format
✓ **Test script updated** to support both formats
✓ **Verification passed** - 11,913 windows loaded correctly
✓ **Performance improved** - 15x faster data loading
✓ **Backward compatible** - Old format still works

The migration is complete and the codebase now supports both data formats automatically.
