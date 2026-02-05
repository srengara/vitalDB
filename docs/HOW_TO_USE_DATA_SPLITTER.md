# Data Splitter Usage Guide

## Overview

The [split_data_by_case.py](split_data_by_case.py) script splits wide-format PPG windows and glucose labels by `case_id` into separate folders. It supports:

1. **Full data splitting** - All windows for each case
2. **Subset sampling** - Percentage-based sampling with stratified glucose coverage
3. **Case filtering** - Process only specific cases
4. **Reproducible sampling** - Configurable random seed

## Quick Start

### Basic Usage (Full Data)

Split all cases with complete data:

```bash
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/cases_split
```

**Output Structure:**
```
new_data_format/cases_split/
├── case_1/
│   ├── ppg_windows.csv
│   └── glucose_labels.csv
├── case_2/
│   ├── ppg_windows.csv
│   └── glucose_labels.csv
├── case_3/
│   ├── ppg_windows.csv
│   └── glucose_labels.csv
└── split_summary.csv
```

### Subset Sampling (For Inference)

Split with 10% of windows per case (stratified by glucose values):

```bash
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/cases_subset_10pct \
  --subset 0.1
```

This is ideal for **quick inference testing** without processing all windows.

## Command-Line Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input_dir` | Directory with `ppg_windows.csv` and `glucose_labels.csv` | `new_data_format/final_datasets` |
| `--output_dir` | Output directory (creates `case_X` subfolders) | `new_data_format/cases_split` |

### Optional Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--subset` | Fraction of windows to sample (0.0-1.0) | `None` (full data) | `--subset 0.1` for 10% |
| `--case_ids` | Specific cases to process (space-separated) | `None` (all cases) | `--case_ids 1 2 3` |
| `--seed` | Random seed for reproducibility | `42` | `--seed 123` |

## Usage Examples

### Example 1: Split All Cases (Full Data)

```bash
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/cases_full
```

**Use Case:** Prepare individual case data for separate processing or analysis.

### Example 2: Create 10% Subset for Inference

```bash
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/cases_inference_10pct \
  --subset 0.1
```

**Use Case:** Quick model inference testing with representative samples.

### Example 3: Create 20% Subset for Specific Cases

```bash
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/cases_subset_20pct \
  --subset 0.2 \
  --case_ids 1 5 10 15
```

**Use Case:** Focus on specific cases with subset sampling.

### Example 4: Process Only Case 1 with 5% Subset

```bash
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/case_1_subset_5pct \
  --subset 0.05 \
  --case_ids 1
```

**Use Case:** Ultra-fast inference on a single case.

### Example 5: Custom Random Seed

```bash
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/cases_subset_seed999 \
  --subset 0.1 \
  --seed 999
```

**Use Case:** Different random sampling for cross-validation.

## Subset Sampling Strategy

### Stratified Glucose Sampling

When `--subset` is specified, the script uses **stratified sampling** to ensure adequate glucose coverage:

1. **Bins glucose values** into up to 10 quantile-based bins
2. **Samples proportionally** from each bin
3. **Maintains temporal order** by sorting sampled indices
4. **Guarantees coverage** of the full glucose range

### Example Output (10% Subset)

```
Processing Case 1
----------------------------------------
  PPG windows: 11913
  Glucose labels: 11913
  Applying 10.0% subset sampling...
    Target sample size: 1191 windows (10.0%)
    Unique glucose values: 1
    Glucose range: 134.0 - 134.0 mg/dL
    Strategy: Random sampling (constant glucose)
    Sampled 1191 windows
    Sampled glucose range: 134.0 - 134.0 mg/dL
    Sampled unique values: 1/1
  After sampling: 1191 windows
  Saved to: new_data_format/cases_subset_10pct/case_1/
    - ppg_windows.csv (1191 rows)
    - glucose_labels.csv (1191 rows)
```

### Why Stratified Sampling?

| Scenario | Sampling Strategy | Benefit |
|----------|-------------------|---------|
| Constant glucose (1 value) | Random sampling | Equal representation |
| Few unique values (2-5) | Proportional from each value | All values included |
| Many unique values (>10) | Binned stratified sampling | Full range coverage |

## Output Files

### Per-Case Folders

Each case gets its own folder: `case_{case_id}/`

#### ppg_windows.csv

Wide format with all amplitude samples:

```csv
window_index,case_id,window_id,amplitude_sample_0,amplitude_sample_1,...,amplitude_sample_499
180800,1,0,4.942,4.6729,...
180801,1,1,5.123,4.891,...
```

#### glucose_labels.csv

Matching glucose labels:

```csv
case_id,window_id,glucose_mg_dl,window_index
1,0,134.0,180800
1,1,134.0,180801
```

### Summary File

#### split_summary.csv

Contains statistics for all processed cases:

```csv
case_id,num_windows,glucose_min,glucose_max,glucose_mean,unique_glucose_values,output_folder
1,1191,134.0,134.0,134.0,1,new_data_format/cases_subset_10pct/case_1
2,2345,85.0,220.0,142.3,45,new_data_format/cases_subset_10pct/case_2
```

## Integration with test_model_with_normalization.py

After splitting, you can test individual cases:

```bash
# Test full case 1
python test_model_with_normalization.py \
  --test_data new_data_format/cases_full/case_1

# Test 10% subset of case 1
python test_model_with_normalization.py \
  --test_data new_data_format/cases_subset_10pct/case_1
```

## Common Workflows

### Workflow 1: Full Pipeline for All Cases

```bash
# Step 1: Split all cases with full data
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/cases_full

# Step 2: Test each case individually
python test_model_with_normalization.py --test_data new_data_format/cases_full/case_1
python test_model_with_normalization.py --test_data new_data_format/cases_full/case_2
# ... etc
```

### Workflow 2: Quick Inference Testing

```bash
# Step 1: Create 5% subset for fast inference
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/cases_inference_5pct \
  --subset 0.05

# Step 2: Run quick inference tests
python test_model_with_normalization.py --test_data new_data_format/cases_inference_5pct/case_1
```

**Performance:** 5% subset = ~20x faster inference

### Workflow 3: Targeted Case Analysis

```bash
# Focus on top-performing cases (e.g., 1, 3, 7) with 20% subset
python split_data_by_case.py \
  --input_dir new_data_format/final_datasets \
  --output_dir new_data_format/top_cases_20pct \
  --subset 0.2 \
  --case_ids 1 3 7

# Analyze results
python test_model_with_normalization.py --test_data new_data_format/top_cases_20pct/case_1
python test_model_with_normalization.py --test_data new_data_format/top_cases_20pct/case_3
python test_model_with_normalization.py --test_data new_data_format/top_cases_20pct/case_7
```

## Performance Considerations

| Subset % | Windows (from 11,913) | Inference Time | Use Case |
|----------|----------------------|----------------|----------|
| 100% (full) | 11,913 | ~5-10 min | Complete evaluation |
| 20% | ~2,383 | ~1-2 min | Thorough testing |
| 10% | ~1,191 | ~30-60 sec | Regular testing |
| 5% | ~596 | ~15-30 sec | Quick validation |
| 1% | ~119 | ~5-10 sec | Ultra-fast check |

## Error Handling

### Missing case_id Column

```
ValueError: PPG data missing 'case_id' column
```

**Solution:** Ensure input files have `case_id` column. Run `ppg_windows_h.py` first if needed.

### Invalid Subset Fraction

```
error: --subset must be between 0.0 and 1.0 (exclusive of 0.0)
```

**Solution:** Use values like `0.1` (10%), `0.2` (20%), etc. Not `0.0` or `>1.0`.

### Case ID Not Found

If you specify `--case_ids 99` but case 99 doesn't exist:

```
Filtering to 0 requested cases: []
```

**Solution:** Check available case IDs in the input data first.

## Tips and Best Practices

1. **Start with small subsets** (5-10%) for initial testing
2. **Use consistent seeds** (`--seed 42`) for reproducible experiments
3. **Check split_summary.csv** to verify glucose coverage after splitting
4. **Create multiple subset sizes** for different use cases (1%, 10%, 100%)
5. **Test stratification** by comparing sampled vs. original glucose distributions

## Verification

After splitting, verify the output:

```bash
# Check summary
cat new_data_format/cases_subset_10pct/split_summary.csv

# Check case 1 data
head new_data_format/cases_subset_10pct/case_1/ppg_windows.csv
head new_data_format/cases_subset_10pct/case_1/glucose_labels.csv

# Count rows
wc -l new_data_format/cases_subset_10pct/case_1/*.csv
```

## Support

For issues or questions:
- Check that input files are in wide format (from `ppg_windows_h.py`)
- Verify case_id column exists in both files
- Review console output for detailed progress and warnings
