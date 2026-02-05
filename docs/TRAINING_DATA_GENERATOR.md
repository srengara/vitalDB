# VitalDB Training Data Generator

A standalone command-line application to generate training data files (`ppg_windows.csv` and `glucose_labels.csv`) from VitalDB cases.

## Quick Start

### Basic Usage

```bash
# Use manual glucose value (95.0 mg/dL)
python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0

# Auto-extract glucose from VitalDB clinical data
python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose auto
```

### Specify Output Directory

```bash
python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0 --output ./my_training_data
```

## What It Does

The application performs the following steps automatically:

1. **Extracts PPG data** from VitalDB for the specified case and track
2. **Cleanses the data** (removes NaN values, fixes timestamps)
3. **Preprocesses the signal** (bandpass filtering, normalization)
4. **Detects peaks** using adaptive thresholding
5. **Filters windows** using template matching (keeps only high-quality beats)
6. **Generates glucose labels** (manual value or auto from clinical data)
7. **Saves training files** in the correct format

## Output Files

The application generates two CSV files ready for training:

### 1. `ppg_windows.csv`
```csv
window_index,sample_index,amplitude
0,0,0.521
0,1,0.534
0,2,0.548
...
```
- Contains N filtered PPG windows
- Each window has ~500 samples (1 second at 500 Hz)

### 2. `glucose_labels.csv`
```csv
window_index,glucose_mg_dl
0,95.0
1,95.0
2,95.0
...
```
- Contains N glucose values (one per window)
- Same glucose value replicated across all windows

## Parameters

### Required

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--case_id` | VitalDB case ID | `1`, `2`, `100` |
| `--track` | PPG track name | `SNUADC/PLETH`, `Primus/PLETH` |
| `--glucose` | Glucose value (mg/dL) or "auto" | `95.0`, `auto` |

### Optional

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output` | `./training_data` | Output directory for CSV files |
| `--height` | `0.3` | Peak height threshold multiplier |
| `--distance` | `0.8` | Peak distance threshold multiplier |
| `--similarity` | `0.85` | Template similarity threshold |

## Examples

### Example 1: Basic Usage with Manual Glucose

```bash
python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0
```

**Output:**
```
VitalDB Training Data Generator
==================================================
Case ID: 1
Track: SNUADC/PLETH
Glucose: 95.0
Output: ./training_data

✓ Extracted PPG data: 150000 samples
✓ Detected 250 peaks
✓ Filtered to 250 high-quality windows
✓ Saved PPG windows: ./training_data/ppg_windows.csv
✓ Saved glucose labels: ./training_data/glucose_labels.csv

Training data generation complete!
```

### Example 2: Auto-Extract Glucose from Clinical Data

```bash
python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose auto
```

The application will:
- Query VitalDB clinical information for case 1
- Extract `preop_glucose` value
- Use that value for all windows

### Example 3: Custom Output Directory

```bash
python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0 --output ./case1_data
```

### Example 4: Adjust Peak Detection Parameters

```bash
python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0 \
    --height 0.4 \
    --distance 0.9 \
    --similarity 0.9
```

Adjust these if you're getting:
- **Too many peaks**: Increase `--height` (e.g., 0.4, 0.5)
- **Too few peaks**: Decrease `--height` (e.g., 0.2, 0.25)
- **Poor quality windows**: Increase `--similarity` (e.g., 0.9, 0.95)

## Training the Model

After generating the data, train your model:

```bash
python -m src.training.train_glucose_predictor --data_dir ./training_data --epochs 100
```

## Common VitalDB Track Names

| Track | Description | Sampling Rate |
|-------|-------------|---------------|
| `SNUADC/PLETH` | PPG from SNU ADC | 500 Hz |
| `SNUADC/ART` | Arterial pressure | 500 Hz |
| `Primus/PLETH` | PPG from Primus | 300 Hz |
| `Vigilance/PLETH` | PPG from Vigilance | 100 Hz |

## Troubleshooting

### "No valid windows extracted"
- Try lowering `--similarity` threshold (e.g., `--similarity 0.7`)
- Try adjusting `--height` parameter

### "No preop_glucose found in clinical data"
- Use manual glucose value instead of `auto`
- Example: `--glucose 95.0`

### "Track not found"
- Check available tracks for your case using the web app
- Make sure track name is exactly correct (case-sensitive)

## File Format Compatibility

The generated files are compatible with:
- ✅ `train_glucose_predictor.py` (ResNet34-1D training script)
- ✅ Any PyTorch training pipeline expecting PPG windows + glucose labels
- ✅ Standard numpy/pandas data loaders

## Notes

- All glucose values in a single run have the same value (constant glucose)
- To create a dataset with varying glucose values, run the script multiple times with different cases
- The script automatically filters out low-quality PPG windows using template matching
