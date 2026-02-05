# How to Run generate_vitaldb_training_data.py

## Overview

This script generates training-ready data (PPG windows + glucose labels) from VitalDB case files.

**What it does:**
1. Extracts PPG signal from VitalDB case
2. Preprocesses and cleanses the signal
3. Detects peaks and creates 1-second windows
4. Filters windows using template matching
5. Generates glucose labels (manual or auto-extracted)
6. Outputs `ppg_windows.csv` and `glucose_labels.csv`

---

## Prerequisites

### 1. Check VitalDB Data
Make sure you have VitalDB case data downloaded:
```bash
ls C:\IITM\vitalDB\data\recordings\
```

You should see case directories like `case_1/`, `case_2/`, etc.

### 2. Verify Python Environment
```bash
# Activate your virtual environment
cd C:\IITM\vitalDB
venv\Scripts\activate

# Verify required packages
python -c "import vitaldb, numpy, pandas, scipy; print('All dependencies OK')"
```

---

## Basic Usage

### Option 1: Manual Glucose Value (Recommended for testing)

```bash
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose 95.0
```

**Parameters:**
- `--case_id 1`: VitalDB case number
- `--track SNUADC/PLETH`: PPG track name (common tracks: SNUADC/PLETH, Primus/PLETH)
- `--glucose 95.0`: Glucose value in mg/dL (you provide this manually)

**Output:**
```
./training_data/
├── ppg_windows.csv      (PPG signal windows)
└── glucose_labels.csv   (Glucose labels for each window)
```

---

### Option 2: Auto-Extract Glucose from Clinical Data

```bash
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose auto
```

**Parameters:**
- `--glucose auto`: Automatically extracts preoperative glucose from VitalDB clinical data

**Note:** This only works if the case has clinical glucose data available (preop_glucose field).

---

### Option 3: Specify Custom Output Directory

```bash
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose 95.0 \
  --output C:\IITM\vitalDB\data\training_datasets\case_1
```

**Parameters:**
- `--output <path>`: Custom output directory for CSV files

---

## Advanced Usage

### Adjust Peak Detection Parameters

If you're getting too few or too many windows, tune these parameters:

```bash
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose 95.0 \
  --height 0.4 \
  --distance 0.9 \
  --similarity 0.9
```

**Parameters:**
- `--height 0.4`: Peak height threshold multiplier (default: 0.3)
  - Higher = stricter peak detection, fewer windows
  - Lower = more lenient, more windows

- `--distance 0.9`: Peak distance threshold multiplier (default: 0.8)
  - Controls minimum distance between peaks
  - Based on sampling rate (e.g., 0.8 × 100 Hz = 80 samples apart)

- `--similarity 0.9`: Template similarity threshold (default: 0.85)
  - Higher = only very similar windows kept (fewer, higher quality)
  - Lower = more windows kept (more data, but lower quality)

---

## Step-by-Step Examples

### Example 1: Generate Data for Case 1 (Top 50 Case)

```bash
# Navigate to project directory
cd C:\IITM\vitalDB

# Activate virtual environment
venv\Scripts\activate

# Run for Case 1
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose auto \
  --output data\training_datasets\case_1_SNUADC_PLETH
```

**Expected Output:**
```
================================================================================
VitalDB Training Data Generator
================================================================================
Case ID: 1
Track: SNUADC/PLETH
Glucose: auto
Output: data\training_datasets\case_1_SNUADC_PLETH

Step 1: Extracting PPG data from VitalDB...
✓ Extracted PPG data: 1234567 samples

Step 2: Loading and cleansing PPG data...
✓ Cleansed data: 1234567 samples

Step 3: Preprocessing signal...
✓ Signal preprocessed

Step 4: Detecting peaks and filtering windows...
✓ Detected 12345 peaks
✓ Extracted 12000 windows
✓ Filtered to 8500 high-quality windows
  Filtering rate: 70.8%

Step 5: Generating glucose labels...
  Attempting to extract glucose from clinical data...
✓ Using clinical glucose: 120.5 mg/dL

Step 6: Saving PPG windows...
✓ Saved PPG windows: data\training_datasets\case_1_SNUADC_PLETH\ppg_windows.csv
  Format: 8500 windows × 100 samples

Step 7: Saving glucose labels...
✓ Saved glucose labels: data\training_datasets\case_1_SNUADC_PLETH\glucose_labels.csv

Step 8: Summary Statistics
================================================================================
  Case ID: 1
  Track: SNUADC/PLETH
  Glucose source: clinical:preop_glucose
  Glucose value: 120.5 mg/dL
  Number of windows: 8500
  Window length: 100 samples
  Sampling rate: 100 Hz
  Total peaks detected: 12345
  Windows after filtering: 8500
  Filtering rate: 70.8%

✓ PPG windows file: data\training_datasets\case_1_SNUADC_PLETH\ppg_windows.csv
✓ Glucose labels file: data\training_datasets\case_1_SNUADC_PLETH\glucose_labels.csv
================================================================================
Training data generation complete!
================================================================================

✅ SUCCESS! Training data files are ready.

To train the model, run:
  python -m src.training.train_glucose_predictor --data_dir data\training_datasets\case_1_SNUADC_PLETH
```

---

### Example 2: Batch Process Top 50 Cases

Create a script to process multiple cases:

**File: `batch_generate.sh` (Git Bash) or `batch_generate.bat` (Windows CMD)**

```bash
#!/bin/bash
# Batch generate training data for Top 50 cases

# Top 50 case IDs
cases=(813 550 2541 3321 4140 1186 5070 4360 3300 1515 3205 1605 4898 4412 4251 3390 6337 4647 4686 4760 5907 3380 94 4911 3097 6351 2318 1995 722 2424 2395 4771 3689 1327 5343 1564 4703 870 2494 2272 5550 2060 5040 2653 5222 2575 1807 4179 876 1793)

track="SNUADC/PLETH"

for case_id in "${cases[@]}"
do
    echo "Processing Case $case_id..."

    python generate_vitaldb_training_data.py \
        --case_id $case_id \
        --track $track \
        --glucose auto \
        --output data/training_datasets/case_${case_id}_${track//\//_}

    if [ $? -eq 0 ]; then
        echo "✓ Case $case_id completed successfully"
    else
        echo "✗ Case $case_id failed"
    fi

    echo ""
done

echo "Batch processing complete!"
```

**Run it:**
```bash
chmod +x batch_generate.sh
./batch_generate.sh
```

---

### Example 3: Process with Lab Data Glucose Values

If you have glucose values from `lab_data.csv`:

```bash
# For Case 813 with specific glucose measurements
python generate_vitaldb_training_data.py \
  --case_id 813 \
  --track SNUADC/PLETH \
  --glucose 120.5 \
  --output data/training_datasets/case_813_glucose_120
```

---

## Output Files

### 1. ppg_windows.csv

**Format:** Long format (multiple rows per window)

```csv
window_index,sample_index,amplitude
0,0,242.355144
0,1,238.591120
0,2,233.592596
...
0,99,195.481234
1,0,250.123456
1,1,248.334567
```

**Structure:**
- Each window has 100 rows (for 100 samples at 100 Hz = 1 second)
- `window_index`: Unique identifier for each window (0, 1, 2, ...)
- `sample_index`: Sample position within window (0-99)
- `amplitude`: PPG amplitude value

### 2. glucose_labels.csv

**Format:** One row per window

```csv
window_index,glucose_mg_dl
0,120.5
1,120.5
2,120.5
...
8499,120.5
```

**Structure:**
- `window_index`: Matches the window_index from ppg_windows.csv
- `glucose_mg_dl`: Glucose value in mg/dL (same for all windows from one case)

---

## Common Issues & Troubleshooting

### Issue 1: "No valid windows extracted"

**Cause:** Peak detection is too strict or signal quality is poor.

**Solution:** Relax the parameters:
```bash
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose 95.0 \
  --height 0.2 \
  --similarity 0.75
```

---

### Issue 2: "No clinical glucose available"

**Cause:** Case doesn't have preop_glucose in clinical data.

**Solution:** Use manual glucose value:
```bash
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose 100.0
```

---

### Issue 3: "Track not found"

**Cause:** Wrong track name or track doesn't exist for this case.

**Solution:** Check available tracks:
```python
import vitaldb
vdb = vitaldb.VitalFile(1)
print(vdb.get_track_names())  # List all available tracks
```

Common track names:
- `SNUADC/PLETH` (most common)
- `Primus/PLETH`
- `Vigilance/PLETH`

---

### Issue 4: ImportError for modules

**Cause:** Python can't find the modules in `src/`.

**Solution:** Run from project root:
```bash
cd C:\IITM\vitalDB
python generate_vitaldb_training_data.py ...
```

---

## Integration with Training Pipeline

After generating data, you can immediately train the model:

```bash
# Step 1: Generate training data
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose auto \
  --output data/training_datasets/case_1

# Step 2: Train the model
python -m src.training.train_glucose_predictor \
  --data_dir data/training_datasets/case_1 \
  --epochs 50 \
  --batch_size 32
```

---

## Performance Tips

### 1. Process Multiple Cases in Parallel

```bash
# Terminal 1
python generate_vitaldb_training_data.py --case_id 1 --track SNUADC/PLETH --glucose auto &

# Terminal 2
python generate_vitaldb_training_data.py --case_id 2 --track SNUADC/PLETH --glucose auto &

# Terminal 3
python generate_vitaldb_training_data.py --case_id 3 --track SNUADC/PLETH --glucose auto &
```

### 2. Monitor Progress

```bash
# Run with output to log file
python generate_vitaldb_training_data.py \
  --case_id 1 \
  --track SNUADC/PLETH \
  --glucose auto 2>&1 | tee case_1.log
```

### 3. Batch Statistics

After processing multiple cases:
```bash
# Count generated windows across all cases
find data/training_datasets -name "ppg_windows.csv" -exec wc -l {} \;
```

---

## Quick Reference

### Minimal Command
```bash
python generate_vitaldb_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0
```

### Full Command with All Options
```bash
python generate_vitaldb_training_data.py \
  --case_id 813 \
  --track SNUADC/PLETH \
  --glucose auto \
  --output data/training_datasets/case_813_SNUADC_PLETH \
  --height 0.3 \
  --distance 0.8 \
  --similarity 0.85
```

### Check Available Tracks for a Case
```python
import vitaldb
vdb = vitaldb.VitalFile(1)
print(vdb.get_track_names())
```

### Verify Output
```bash
# Check if files were created
ls -lh data/training_datasets/case_1/

# Check number of windows
wc -l data/training_datasets/case_1/glucose_labels.csv
```

---

## Summary

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--case_id` | ✓ | - | VitalDB case number (1-6388) |
| `--track` | ✓ | - | PPG track name (e.g., SNUADC/PLETH) |
| `--glucose` | ✓ | - | Glucose value (mg/dL) or "auto" |
| `--output` |  | ./training_data | Output directory |
| `--height` |  | 0.3 | Peak height multiplier |
| `--distance` |  | 0.8 | Peak distance multiplier |
| `--similarity` |  | 0.85 | Template similarity threshold |

**Next Steps:** After generating data, train your model with:
```bash
python -m src.training.train_glucose_predictor --data_dir <output_directory>
```
