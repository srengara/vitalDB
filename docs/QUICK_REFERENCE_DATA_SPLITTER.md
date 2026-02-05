# Data Splitter Quick Reference

## Super Simple Usage

### Split All Cases with Subset (Most Common)

```bash
# 10% subset for fast inference (stratified - preserves distribution)
python split_data_by_case.py --subset 0.1

# 10% balanced subset (equal glucose representation)
python split_data_by_case.py --subset 0.1 --sampling balanced

# 5% subset for ultra-fast inference
python split_data_by_case.py --subset 0.05

# 20% subset for thorough testing
python split_data_by_case.py --subset 0.2

# Full data (no subset)
python split_data_by_case.py --case_ids all
```

**Output:** Auto-creates `new_data_format/cases_subset_Xpct/case_Y/` folders

### Sampling Strategies

| Strategy | Command | Use Case |
|----------|---------|----------|
| **Stratified** (default) | `--sampling stratified` | Realistic testing (preserves glucose distribution) |
| **Balanced** | `--sampling balanced` | Comprehensive testing (equal glucose representation) |

### Advanced Options

```bash
# Custom input/output directories
python split_data_by_case.py \
  --input_dir path/to/combined/data \
  --output_dir path/to/output \
  --subset 0.1 \
  --case_ids all

# Specific cases only
python split_data_by_case.py --subset 0.1 --case_ids 1 5 10

# Single case
python split_data_by_case.py --subset 0.2 --case_ids 1
```

## Parameters

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--subset` | `None` | Fraction to sample (0.0-1.0) | `--subset 0.1` (10%) |
| `--sampling` | `stratified` | Sampling strategy | `--sampling balanced` |
| `--case_ids` | `all` | Cases to process | `--case_ids 1 2 3` or `--case_ids all` |
| `--input_dir` | `new_data_format/final_datasets` | Input directory | `--input_dir my/data` |
| `--output_dir` | Auto-generated | Output directory | `--output_dir my/output` |
| `--seed` | `42` | Random seed | `--seed 999` |

## Auto-Generated Output Paths

| Command | Output Path |
|---------|-------------|
| `--subset 0.1` | `new_data_format/cases_subset_10pct/` |
| `--subset 0.05` | `new_data_format/cases_subset_5pct/` |
| `--subset 0.2` | `new_data_format/cases_subset_20pct/` |
| `--case_ids all` (no subset) | `new_data_format/cases_split/` |

## Output Structure

```
new_data_format/cases_subset_10pct/
├── case_1/
│   ├── ppg_windows.csv      (wide format, 10% of windows)
│   └── glucose_labels.csv   (matching labels)
├── case_2/
│   ├── ppg_windows.csv
│   └── glucose_labels.csv
├── ...
└── split_summary.csv        (statistics for all cases)
```

## Quick Workflows

### Workflow 1: Fast Inference Testing

```bash
# Step 1: Create 5% subset
python split_data_by_case.py --subset 0.05

# Step 2: Test on case 1 (fast!)
python test_model_with_normalization.py \
  --test_data new_data_format/cases_subset_5pct/case_1
```

**Time:** ~15-30 seconds per case

### Workflow 2: Standard Testing

```bash
# Step 1: Create 10% subset
python split_data_by_case.py --subset 0.1

# Step 2: Test on case 1
python test_model_with_normalization.py \
  --test_data new_data_format/cases_subset_10pct/case_1
```

**Time:** ~30-60 seconds per case

### Workflow 3: Full Evaluation

```bash
# Step 1: Split all cases (full data)
python split_data_by_case.py --case_ids all

# Step 2: Test on case 1 (complete)
python test_model_with_normalization.py \
  --test_data new_data_format/cases_split/case_1
```

**Time:** ~5-10 minutes per case

## Examples from Your Setup

### Example 1: Current Dataset (Case 1, 11,913 windows)

```bash
# 10% subset = 1,191 windows
python split_data_by_case.py --subset 0.1
```

**Output:**
- `new_data_format/cases_subset_10pct/case_1/ppg_windows.csv` (1,191 rows, 4.6 MB)
- `new_data_format/cases_subset_10pct/case_1/glucose_labels.csv` (1,191 rows, 29 KB)

### Example 2: Multiple Cases with Varying Glucose

Assuming you have cases 1, 2, 3 with different glucose values:

```bash
# 10% subset of all cases
python split_data_by_case.py --subset 0.1 --case_ids all
```

**Stratified Sampling:**
- Case with glucose range 80-200 mg/dL → Samples across full range
- Case with constant glucose → Random sampling
- Ensures glucose distribution is preserved

## Verification

```bash
# Check created folders
ls new_data_format/cases_subset_10pct/

# Check case 1 files
ls -lh new_data_format/cases_subset_10pct/case_1/

# View summary statistics
cat new_data_format/cases_subset_10pct/split_summary.csv

# Check row counts
wc -l new_data_format/cases_subset_10pct/case_1/*.csv
```

## Common Use Cases

| Use Case | Command | Speed | Accuracy |
|----------|---------|-------|----------|
| Ultra-fast check | `--subset 0.01` | 5-10 sec | Lower |
| Quick validation | `--subset 0.05` | 15-30 sec | Good |
| Standard testing | `--subset 0.1` | 30-60 sec | Very Good |
| Thorough testing | `--subset 0.2` | 1-2 min | Excellent |
| Full evaluation | `--case_ids all` | 5-10 min | Complete |

## Tips

1. **Start small:** Use `--subset 0.05` for initial testing
2. **Default is smart:** Just run `python split_data_by_case.py --subset 0.1` for most use cases
3. **Reproducible:** Same `--seed` gives same sampling
4. **Multiple subsets:** Create different percentages for different use cases
   ```bash
   python split_data_by_case.py --subset 0.05  # Fast
   python split_data_by_case.py --subset 0.1   # Standard
   python split_data_by_case.py --subset 0.2   # Thorough
   ```

## Integration

The split data works seamlessly with `test_model_with_normalization.py`:

```bash
# After splitting
python test_model_with_normalization.py \
  --test_data new_data_format/cases_subset_10pct/case_1 \
  --model_path model/best_model.pth
```

Results saved to:
- `new_data_format/cases_subset_10pct/case_1/test_results_normalized/predictions.csv`
