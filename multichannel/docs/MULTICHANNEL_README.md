# Multi-Channel PPG Training Data Pipeline

Complete pipeline for processing multi-channel PPG signals with integrated visualization.

## 📋 Overview

This pipeline processes multi-channel PPG data files where glucose values are encoded in the filename, producing training-ready datasets with full traceability through intermediate processing steps.

### Key Features

- ✅ **No labs CSV required** - Glucose value extracted from filename
- ✅ **No 16-minute windowing** - Processes entire signal
- ✅ **Intermediate file saving** - 8 processing steps saved separately
- ✅ **Web visualization** - Interactive plots of all steps
- ✅ **Batch processing** - Process entire folders automatically
- ✅ **Feature extraction** - Automatic extraction of training-relevant features

---

## 📁 Input File Format

### Expected Filename Pattern
```
<channel>-GLUC<value>-SYS<value>-DIA<value>.csv
```

### Examples
```
force-GLUC123-SYS140-DIA91.csv
Signal1-GLUC123-SYS140-DIA91.csv
Signal2-GLUC123-SYS140-DIA91.csv
Signal3-GLUC123-SYS140-DIA91.csv
```

### CSV Format
Must contain columns:
- `time` - Timestamp in seconds
- `ppg` or `signal` or `amplitude` - Signal values

---

## 🚀 Quick Start

### 1. Process Single File
```bash
python generate_multichannel_training_data.py \
    --input "C:\senzrtech\Multi-channel\multi-channel-input-files\force-GLUC123-SYS140-DIA91.csv" \
    --output ./output
```

### 2. Batch Process Folder
```bash
python generate_multichannel_training_data.py \
    --input_folder "C:\senzrtech\Multi-channel\multi-channel-input-files" \
    --output ./output
```

### 3. Launch Web Visualization
```bash
python run_multichannel_web_app.py --data ./output
```
Then open: http://localhost:5001

---

## 📂 Output Structure

For each input file (e.g., `force-GLUC123-SYS140-DIA91.csv`), the pipeline creates:

```
output/
└── force-GLUC123-SYS140-DIA91/
    ├── force-GLUC123-SYS140-DIA91-01-raw.csv           # Raw data
    ├── force-GLUC123-SYS140-DIA91-02-cleaned.csv       # Time repaired
    ├── force-GLUC123-SYS140-DIA91-03-downsampled.csv   # 100Hz resampled
    ├── force-GLUC123-SYS140-DIA91-04-preprocessed.csv  # Bandpass filtered
    ├── force-GLUC123-SYS140-DIA91-05-peaks.csv         # Detected peaks
    ├── force-GLUC123-SYS140-DIA91-06-windows.csv       # All windows
    ├── force-GLUC123-SYS140-DIA91-07-filtered.csv      # Quality-filtered windows
    ├── force-GLUC123-SYS140-DIA91-08-template.csv      # Template signal
    ├── force-GLUC123-SYS140-DIA91-output.csv           # FINAL OUTPUT (wide format)
    └── force-GLUC123-SYS140-DIA91-metadata.json        # Processing metadata
```

---

## 🔬 Processing Pipeline

### Step 1: Time Repair
- Detects missing/broken timestamps
- Reconstructs time axis based on sampling rate
- Aligns to valid start time

### Step 2: Signal Cleaning
- Tracks missing data with mask
- Applies forward fill to NaNs
- Trims leading NaNs (unfixable by forward fill)

### Step 3: Downsampling to 100Hz
- Resamples signal if >100Hz
- Preserves "bad data" mask through interpolation
- Ensures consistent sampling rate

### Step 4: Preprocessing
- Removes DC component
- Bandpass filter: 0.5-8 Hz, 3rd order Butterworth
- Matches Nature 2025 paper specifications

### Step 5: Peak Detection
- Height threshold: mean + 0.3 × std
- Distance threshold: 0.8 × sampling_rate
- Extracts 1-second windows (100 samples)

### Step 6: Window Filtering
- Computes average template
- Filters by cosine similarity (≥0.85)
- Keeps only high-quality windows

### Step 7: Output Generation
- Wide format: 100 columns (amplitude_sample_0 to amplitude_sample_99)
- Tags windows: 4040 (repaired data) or 8080 (pure data)
- Applies glucose range filter (12-483 mg/dL)
- Final NaN validation

---

## 🎛️ Command-Line Options

### Processing Script: `generate_multichannel_training_data.py`

```bash
# Required (one of):
--input FILE            Process single file
--input_folder FOLDER   Process all CSV files in folder
--output DIR            Output directory

# Optional:
--sampling_rate HZ      Override sampling rate (default: auto-detect)
--height FLOAT          Peak height multiplier (default: 0.3)
--distance FLOAT        Peak distance multiplier (default: 0.8)
--similarity FLOAT      Template similarity threshold (default: 0.85)
```

### Web App: `run_multichannel_web_app.py`

```bash
--data DIR              Output directory to visualize (default: ./output)
--port PORT             Server port (default: 5001)
--host HOST             Server host (default: 0.0.0.0)
```

---

## 📊 Output File Details

### Final Output CSV (`*-output.csv`)

Wide format with columns:

| Column | Description |
|--------|-------------|
| `channel` | Signal channel name (e.g., 'force', 'Signal1') |
| `window_index` | Unique window ID (includes quality tag) |
| `glucose_mg_dl` | Glucose value from filename |
| `systolic_mmhg` | Systolic BP (if in filename) |
| `diastolic_mmhg` | Diastolic BP (if in filename) |
| `amplitude_sample_0` to `amplitude_sample_99` | 100 PPG samples (1 second at 100Hz) |

**Window Index Format:**
- First 4 digits = Quality tag:
  - `4040` = Contains repaired/interpolated data
  - `8080` = Pure, high-quality data
- Remaining digits = Sequential window number

Example: `40400012` = 12th window, contains repaired data

### Metadata JSON (`*-metadata.json`)

```json
{
  "status": "SUCCESS",
  "input_file": "path/to/input.csv",
  "output_file": "path/to/output.csv",
  "metadata": {
    "channel": "force",
    "glucose": 123,
    "systolic": 140,
    "diastolic": 91,
    "inferred_sampling_rate": 100.0,
    "duration_seconds": 120.5
  },
  "processing": {
    "sampling_rate": 100.0,
    "total_samples": 12050,
    "peaks_detected": 142,
    "windows_extracted": 142,
    "windows_filtered": 128,
    "filtering_rate": 90.14,
    "final_output_rows": 128,
    "data_quality": "pure"
  }
}
```

---

## 🌐 Web Visualization Features

### Dashboard View
- Lists all processed cases
- Shows glucose values, channels, and status
- Quick access to detailed analysis

### Case Analysis View
1. **Metadata Panel**
   - Glucose, BP, channel info
   - Processing statistics
   - Data quality indicators

2. **Signal Processing Pipeline**
   - 4-panel plot showing transformation at each step
   - Raw → Cleaned → Downsampled → Preprocessed

3. **Peak Detection**
   - Preprocessed signal with detected peaks
   - Peak timing and heart rate analysis

4. **Window Extraction & Filtering**
   - Side-by-side comparison
   - Before/after filtering visualization

5. **Template Matching**
   - Computed template overlay
   - Sample filtered windows

6. **Extracted Features**
   - Signal statistics (mean, std, range)
   - Window statistics (amplitudes, variability)
   - Heart rate features (mean, std, min, max)

---

## 🔍 Quality Tags Explained

### 8080 = Pure Data
✅ Original signal had no missing values
✅ No interpolation or repair needed
✅ Highest quality for training

### 4040 = Repaired Data
⚠️ Original signal had missing values
⚠️ Forward fill or interpolation applied
⚠️ Lower confidence, but still usable

**Recommendation:** For critical training, filter for 8080 windows only.

---

## 🧪 Feature Extraction

The pipeline automatically extracts features useful for glucose prediction:

### Signal-Level Features
- Mean amplitude
- Standard deviation
- Min/Max/Range
- Signal energy

### Window-Level Features
- Per-window amplitude statistics
- Mean/std of window amplitudes
- Mean/std of window ranges
- Window-to-window variability

### Heart Rate Features
- Number of peaks detected
- Mean heart rate (BPM)
- Heart rate variability (std)
- Min/Max heart rate

These features are:
- Displayed in the web app
- Saved in metadata JSON
- Accessible via API endpoints

---

## 🛠️ Troubleshooting

### Issue: "No 'time' column found"
**Solution:** Ensure CSV has a column named 'time' (case-insensitive)

### Issue: "Could not extract glucose value"
**Solution:** Filename must contain `GLUC<number>` (e.g., `GLUC123`)

### Issue: "Data contains non-recoverable NaNs"
**Solution:** File has leading NaNs that can't be filled. Check raw data quality.

### Issue: "No valid windows in final output"
**Solution:** Signal quality too low. Try adjusting parameters:
```bash
--height 0.2 --distance 0.6 --similarity 0.80
```

### Issue: Web app shows "No Cases Found"
**Solution:** Check that `--data` path points to the output directory containing processed folders

---

## 📈 Example Workflow

### Complete Multi-Channel Analysis

```bash
# 1. Process all channels for a glucose measurement
python generate_multichannel_training_data.py \
    --input_folder "C:\senzrtech\Multi-channel\multi-channel-input-files" \
    --output ./multichannel_output

# Output:
# ✅ force-GLUC123-SYS140-DIA91 processed: 128 windows
# ✅ Signal1-GLUC123-SYS140-DIA91 processed: 135 windows
# ✅ Signal2-GLUC123-SYS140-DIA91 processed: 142 windows
# ✅ Signal3-GLUC123-SYS140-DIA91 processed: 131 windows

# 2. Launch visualization
python run_multichannel_web_app.py --data ./multichannel_output

# 3. Open browser: http://localhost:5001

# 4. Combine final outputs for training
python combine_channels.py --input ./multichannel_output --output training_data.csv
```

---

## 🔗 Integration with Training Pipeline

### Loading Processed Data

```python
import pandas as pd

# Load final output
df = pd.read_csv('force-GLUC123-SYS140-DIA91-output.csv')

# Separate features from labels
X = df[[f'amplitude_sample_{i}' for i in range(100)]].values  # Shape: (N, 100)
y = df['glucose_mg_dl'].values  # Shape: (N,)

# Optional: Filter for high-quality windows only
df_pure = df[df['window_index'].astype(str).str.startswith('8080')]
X_pure = df_pure[[f'amplitude_sample_{i}' for i in range(100)]].values
y_pure = df_pure['glucose_mg_dl'].values
```

### Combining Multiple Channels

```python
import pandas as pd
from pathlib import Path

channels = ['force', 'Signal1', 'Signal2', 'Signal3']
all_data = []

for channel in channels:
    csv_file = f"{channel}-GLUC123-SYS140-DIA91-output.csv"
    df = pd.read_csv(csv_file)
    all_data.append(df)

# Concatenate all channels
combined_df = pd.concat(all_data, ignore_index=True)

# Or: Create multi-channel feature matrix
# Stack channels as additional features (shape: N x 400 for 4 channels)
```

---

## 📚 Comparison with VitalDB Pipeline

| Feature | VitalDB (d7.py) | Multi-Channel |
|---------|----------------|---------------|
| Input | VitalDB case + labs CSV | Single CSV with glucose in filename |
| Time windowing | ±8 min around glucose measurement | Entire signal |
| Glucose source | External labs CSV | Filename parsing |
| Output windows | From multiple measurements | From single measurement |
| Intermediate files | None | 8 steps saved |
| Visualization | Separate tool | Integrated web app |
| Batch mode | Supported | Supported |

**Core signal processing is identical between both pipelines.** See `MULTICHANNEL_VALIDATION.md` for detailed validation.

---

## 🎯 Next Steps

1. **Process your data:**
   ```bash
   python generate_multichannel_training_data.py \
       --input_folder "C:\senzrtech\Multi-channel\multi-channel-input-files" \
       --output ./output
   ```

2. **Visualize results:**
   ```bash
   python run_multichannel_web_app.py --data ./output
   ```

3. **Review quality:**
   - Check filtering rates in web app
   - Identify problematic files
   - Adjust parameters if needed

4. **Extract training data:**
   - Load `*-output.csv` files
   - Optionally filter for 8080 windows
   - Combine multiple channels if needed

---

## 📞 Support

For issues or questions:
1. Check `MULTICHANNEL_VALIDATION.md` for pipeline details
2. Review intermediate files to identify processing stage issues
3. Use web visualization to inspect signal quality
4. Adjust processing parameters based on signal characteristics

---

## 📄 License

Same as parent VitalDB processing pipeline.
