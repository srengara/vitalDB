# Glucose Labels Generator

A simple utility to generate `glucose_labels.csv` with a constant glucose value repeated N times.

## Quick Start

```bash
# Generate 1000 rows with glucose value 95.0 mg/dL
python generate_glucose_labels.py --n 1000 --glucose 95.0
```

## What It Does

Takes two inputs:
- **N** (number of windows/rows)
- **Glucose value** (in mg/dL)

Generates a CSV file with N rows, all containing the same glucose value.

## Usage

### Basic Example

```bash
python generate_glucose_labels.py --n 1000 --glucose 95.0
```

This creates `./training_data/glucose_labels.csv` with 1000 rows:
```csv
window_index,glucose_mg_dl
0,95.0
1,95.0
2,95.0
...
999,95.0
```

### Custom Output Directory

```bash
python generate_glucose_labels.py --n 500 --glucose 120.5 --output ./my_data
```

This creates `./my_data/glucose_labels.csv` with 500 rows, all with glucose value 120.5.

## Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--n` | Yes | Number of windows (rows) | `1000`, `500` |
| `--glucose` | Yes | Glucose value in mg/dL | `95.0`, `120.5` |
| `--output` | No | Output directory | `./my_data` (default: `./training_data`) |

## Output Format

The generated file `glucose_labels.csv` has this format:

```csv
window_index,glucose_mg_dl
0,95.0
1,95.0
2,95.0
...
```

- **Column 1**: `window_index` - Sequential indices from 0 to N-1
- **Column 2**: `glucose_mg_dl` - Constant glucose value (all rows same)

## Use Cases

### 1. Testing the Training Pipeline

Generate dummy glucose labels to test your training script:

```bash
python generate_glucose_labels.py --n 100 --glucose 95.0
```

### 2. Creating Training Data with Known Glucose

When you know the glucose value for a patient/case, generate labels for all PPG windows:

```bash
python generate_glucose_labels.py --n 1500 --glucose 110.0
```

### 3. Multiple Cases with Different Glucose Values

Generate separate files for different glucose levels:

```bash
# Low glucose case
python generate_glucose_labels.py --n 800 --glucose 70.0 --output ./case1_low

# Normal glucose case
python generate_glucose_labels.py --n 1200 --glucose 95.0 --output ./case2_normal

# High glucose case
python generate_glucose_labels.py --n 1000 --glucose 150.0 --output ./case3_high
```

## Integration with Training

After generating `glucose_labels.csv`, you need `ppg_windows.csv` to train:

```bash
# Step 1: Generate glucose labels
python generate_glucose_labels.py --n 1000 --glucose 95.0

# Step 2: Make sure ppg_windows.csv exists in ./training_data/
# (Use generate_training_data.py or web app to create it)

# Step 3: Train the model
python -m src.training.train_glucose_predictor --data_dir ./training_data
```

## Comparison with Other Tools

| Tool | Purpose | Input Required |
|------|---------|----------------|
| `generate_glucose_labels.py` | Generate glucose labels only | N, glucose value |
| `generate_training_data.py` | Generate complete training data (PPG + glucose) | case_id, track, glucose |
| Web App | Interactive PPG analysis + training data generation | Browser interaction |

## Example Output

```
======================================================================
Glucose Labels Generator
======================================================================
Number of windows (N): 1000
Glucose value: 95.0 mg/dL
Output directory: ./training_data

✓ Generated glucose array of shape (1000,)
  All values: 95.0 mg/dL
✓ Saved glucose labels: ./training_data/glucose_labels.csv

File format:
  - Rows: 1000
  - Columns: ['window_index', 'glucose_mg_dl']

Sample data:
   window_index  glucose_mg_dl
              0           95.0
              1           95.0
              2           95.0
              3           95.0
              4           95.0
              5           95.0
              6           95.0
              7           95.0
              8           95.0
              9           95.0
======================================================================
Glucose labels file generated successfully!
======================================================================
```

## Notes

- The value of **N** should match the number of PPG windows in your `ppg_windows.csv`
- All N rows will contain the exact same glucose value
- The file is compatible with `train_glucose_predictor.py`
- Glucose values are validated to be in the range 0-1000 mg/dL

## Troubleshooting

### "Number of windows must be positive"
Make sure `--n` is a positive integer (e.g., `--n 1000`, not `--n 0` or `--n -100`)

### "Glucose value out of valid range"
Glucose must be between 0 and 1000 mg/dL (e.g., `--glucose 95.0`, not `--glucose -10` or `--glucose 5000`)

### "Window count mismatch during training"
The value of `--n` must match the number of windows in `ppg_windows.csv`. Check how many unique `window_index` values exist in your PPG windows file.
