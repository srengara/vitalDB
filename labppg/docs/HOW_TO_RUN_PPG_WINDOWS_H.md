# How to Run ppg_windows_h.py

## Overview

**Purpose:** This script combines multiple case PPG windows and glucose labels into final combined datasets.

**What it does:**
1. Scans for `ppg_windows.csv` files in your training data directories
2. Sorts and processes each case's PPG windows
3. Converts PPG windows from long format to wide format (pivot operation)
4. Combines glucose labels from all cases
5. Outputs two final files:
   - `final_datasets/ppg_windows.csv` (combined, wide format)
   - `final_datasets/glucose_labels.csv` (combined)

---

## Prerequisites

### 1. Update Path Configuration

**IMPORTANT:** The script is currently configured for Ashwanth's machine. You need to update the paths:

**Open the file and change line 18:**
```python
# Current (Ashwanth's path):
ROOT = Path(r"C:\Users\ashwa\OneDrive\Desktop\VS Code\senzrTech")

# Change to YOUR path:
ROOT = Path(r"C:\IITM\vitalDB")
```

### 2. Directory Structure

Make sure your data is organized like this:

```
C:\IITM\vitalDB\
├── training_data/                  # Source directory
│   ├── case_1_SNUADC_PLETH/
│   │   ├── ppg_windows.csv
│   │   └── glucose_labels.csv
│   ├── case_2_SNUADC_PLETH/
│   │   ├── ppg_windows.csv
│   │   └── glucose_labels.csv
│   └── ...
└── final_datasets/                 # Output directory (created automatically)
    ├── ppg_windows.csv             # Combined PPG windows (wide format)
    └── glucose_labels.csv          # Combined glucose labels
```

---

## Quick Start

### Step 1: Update the Script Path

Edit `C:\IITM\vitalDB\src\training\ppg_windows_h.py` line 18:

```python
ROOT = Path(r"C:\IITM\vitalDB")
```

### Step 2: Run the Script

```bash
# Navigate to project directory
cd C:\IITM\vitalDB

# Activate virtual environment
venv\Scripts\activate

# Run the script
python src\training\ppg_windows_h.py
```

### Step 3: Choose How Many Cases

When prompted:
```
How many files to process? (1-50, blank for all):
```

- Press **Enter** to process all cases
- Enter a number (e.g., **10**) to process only the first 10 cases

### Step 4: Check Output

```bash
# Check the combined files
ls final_datasets\

# You should see:
#   ppg_windows.csv
#   glucose_labels.csv
```

---

## Detailed Usage Examples

### Example 1: Process All Cases

```bash
cd C:\IITM\vitalDB
venv\Scripts\activate
python src\training\ppg_windows_h.py
# Press Enter when prompted
```

**Output:**
```
OK case_1_SNUADC_PLETH - PPG processed
OK case_1_SNUADC_PLETH - glucose labels sorted
OK case_2_SNUADC_PLETH - PPG processed
OK case_2_SNUADC_PLETH - glucose labels sorted
...
OK case_50_SNUADC_PLETH - PPG processed
OK case_50_SNUADC_PLETH - glucose labels sorted
```

**Result:**
- `final_datasets/ppg_windows.csv` - All PPG windows combined (wide format)
- `final_datasets/glucose_labels.csv` - All glucose labels combined

### Example 2: Process Only 10 Cases

```bash
cd C:\IITM\vitalDB
venv\Scripts\activate
python src\training\ppg_windows_h.py
# Enter: 10
```

**Output:**
```
How many files to process? (1-50, blank for all): 10
OK case_1_SNUADC_PLETH - PPG processed
OK case_1_SNUADC_PLETH - glucose labels sorted
...
OK case_10_SNUADC_PLETH - PPG processed
OK case_10_SNUADC_PLETH - glucose labels sorted
```

### Example 3: Process Cases for Testing

```bash
# Test with just 3 cases first
python src\training\ppg_windows_h.py
# Enter: 3

# Verify output
head -20 final_datasets\ppg_windows.csv
head -20 final_datasets\glucose_labels.csv
```

---

## What the Script Does in Detail

### Input Format (Long Format PPG)

**Before processing:**

`case_1_SNUADC_PLETH/ppg_windows.csv`:
```csv
case_id,window_id,sample_index,amplitude
1,0,0,242.355
1,0,1,238.591
1,0,2,233.592
...
1,0,99,195.481
1,1,0,250.123
```

**Before processing:**

`case_1_SNUADC_PLETH/glucose_labels.csv`:
```csv
case_id,window_id,glucose_mg_dl
1,0,120.5
1,1,120.5
```

### Output Format (Wide Format PPG)

**After processing:**

`final_datasets/ppg_windows.csv`:
```csv
window_index,case_id,window_id,amplitude_sample_0,amplitude_sample_1,...,amplitude_sample_99
18080,1,0,242.355,238.591,...,195.481
18081,1,1,250.123,248.334,...,198.765
28080,2,0,245.678,243.123,...,201.234
```

**After processing:**

`final_datasets/glucose_labels.csv`:
```csv
case_id,window_id,glucose_mg_dl,window_index
1,0,120.5,18080
1,1,120.5,18081
2,0,135.2,28080
```

---

## Key Features

### 1. Wide Format Conversion (Pivot)

The script converts PPG windows from **long format** (100 rows per window) to **wide format** (1 row per window with 100 columns).

**Benefits:**
- **15-25x faster** data loading during training
- Easier to work with in pandas
- More efficient memory usage

### 2. Combined Window Index

Creates a unique identifier: `window_index = case_id + "8080" + window_id`

**Example:**
- Case 1, Window 0 → `18080`
- Case 1, Window 1 → `18081`
- Case 2, Window 0 → `28080`

### 3. Sorted Output

Final files are sorted by:
1. `case_id`
2. `window_id`

This ensures consistent ordering across runs.

### 4. Memory Efficient

- Processes files in chunks (200,000 rows at a time)
- Uses downcast to reduce memory footprint
- Streams data to output files

---

## Configuration Options

You can modify these constants at the top of the script:

```python
# Line 18: Project root directory
ROOT = Path(r"C:\IITM\vitalDB")

# Line 19: Where to find case data
DATA_ROOT = ROOT / "training_data"

# Line 20: Where to save combined output
FINAL_DIR = ROOT / "final_datasets"

# Line 23: Chunk size for writing (reduce if memory issues)
CSV_CHUNKSIZE = 200_000
```

---

## Common Issues & Solutions

### Issue 1: "No ppg_windows.csv files found"

**Cause:** Script can't find input data

**Solution:** Check your directory structure
```bash
# Verify training data exists
ls training_data\

# Should see case directories with ppg_windows.csv files
ls training_data\case_*\ppg_windows.csv
```

### Issue 2: Path Error

**Cause:** ROOT path is still set to Ashwanth's machine

**Solution:** Edit line 18 in the script
```python
ROOT = Path(r"C:\IITM\vitalDB")  # Your path
```

### Issue 3: "Missing expected columns"

**Cause:** Input CSV has wrong format or missing columns

**Solution:** Check input file format
```bash
# Check columns in your PPG file
head -1 training_data\case_1_SNUADC_PLETH\ppg_windows.csv
```

Should have: `case_id`, `window_id` (or `window_index`), `sample_index`, `amplitude`

### Issue 4: Memory Error

**Cause:** Processing too much data at once

**Solution:** Reduce chunk size in the script
```python
CSV_CHUNKSIZE = 100_000  # Reduce from 200,000
```

Or process fewer cases:
```bash
python src\training\ppg_windows_h.py
# Enter: 10  (process only 10 cases at a time)
```

---

## Verification Steps

After running, verify your output:

### 1. Check Files Exist
```bash
ls -lh final_datasets\
```

Should see:
- `ppg_windows.csv` (larger file)
- `glucose_labels.csv` (smaller file)

### 2. Check Row Counts
```bash
# Count PPG windows
python -c "import pandas as pd; df=pd.read_csv('final_datasets/ppg_windows.csv'); print(f'PPG windows: {len(df)}')"

# Count glucose labels
python -c "import pandas as pd; df=pd.read_csv('final_datasets/glucose_labels.csv'); print(f'Glucose labels: {len(df)}')"
```

Numbers should match (same number of windows).

### 3. Check Wide Format
```bash
# Check number of columns (should be ~103: window_index, case_id, window_id, + 100 amplitude_sample_* columns)
python -c "import pandas as pd; df=pd.read_csv('final_datasets/ppg_windows.csv'); print(f'Columns: {len(df.columns)}'); print(df.columns.tolist()[:10])"
```

### 4. Inspect Data
```python
import pandas as pd

# Load combined PPG
ppg = pd.read_csv('final_datasets/ppg_windows.csv')
print(ppg.head())
print(f"\nShape: {ppg.shape}")

# Load combined glucose
glucose = pd.read_csv('final_datasets/glucose_labels.csv')
print(glucose.head())
print(f"\nShape: {glucose.shape}")
```

---

## Integration with Training

After running this script, you can train directly on the combined dataset:

**Option 1: Use the wide format (recommended for faster training)**
```bash
python -m src.training.train_glucose_predictor \
  --data_dir final_datasets \
  --epochs 100 \
  --batch_size 32
```

**Note:** You may need to update `train_glucose_predictor.py` to handle wide format. See the data format analysis document.

**Option 2: Train on individual cases**
```bash
# Train on specific case
python -m src.training.train_glucose_predictor \
  --data_dir training_data/case_1_SNUADC_PLETH \
  --epochs 50
```

---

## Summary

| Step | Command |
|------|---------|
| 1. Edit path | Change line 18 to `ROOT = Path(r"C:\IITM\vitalDB")` |
| 2. Run script | `python src\training\ppg_windows_h.py` |
| 3. Choose cases | Press Enter (all) or enter number (e.g., 10) |
| 4. Verify output | `ls final_datasets\` |
| 5. Train model | `python -m src.training.train_glucose_predictor --data_dir final_datasets` |

**Output Files:**
- `final_datasets/ppg_windows.csv` - Combined PPG in wide format (1 row per window)
- `final_datasets/glucose_labels.csv` - Combined glucose labels

**Benefits:**
- All cases combined in one file
- Wide format for faster training (15-25x faster loading)
- Sorted and organized data
- Ready for model training
