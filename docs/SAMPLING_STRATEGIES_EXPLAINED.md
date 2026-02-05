# Sampling Strategies Explained

## Overview

The data splitter now supports two sampling strategies when creating subsets:

1. **Stratified Sampling** (default) - Preserves original glucose distribution
2. **Balanced Sampling** (new) - Equal representation across all glucose values

## When to Use Each Strategy

| Strategy | Use Case | Best For |
|----------|----------|----------|
| **Stratified** | Realistic evaluation | Production testing, final metrics, representative sampling |
| **Balanced** | Comprehensive testing | Inference testing, model validation across all glucose ranges, debugging |

## Case Study: Case 94

### Original Distribution (17,827 windows)

| Glucose (mg/dL) | Count | Percentage |
|-----------------|-------|------------|
| 88.0 | 15,875 | 89.2% |
| 127.0 | 817 | 4.6% |
| 148.0 | 536 | 3.0% |
| 200.0 | 599 | 3.4% |

### Stratified Sampling (10% subset = 1,783 windows)

**Preserves the original 89:5:3:3 distribution**

| Glucose (mg/dL) | Count | Percentage | Comment |
|-----------------|-------|------------|---------|
| 88.0 | ~1,588 | 89.2% | Dominant value preserved |
| 127.0 | ~82 | 4.6% | Minority value underrepresented |
| 148.0 | ~53 | 3.0% | Very few samples |
| 200.0 | ~60 | 3.4% | Very few samples |

**Problem:** With only ~53-82 samples for minority glucose values, model performance on these ranges may not be well-tested.

### Balanced Sampling (10% subset = 1,782 windows)

**Equal representation: 25:25:25:25 distribution**

| Glucose (mg/dL) | Count | Percentage | Comment |
|-----------------|-------|------------|---------|
| 88.0 | 446 | 25.0% | Reduced but still well-represented |
| 127.0 | 446 | 25.0% | **5x more samples** than stratified |
| 148.0 | 445 | 25.0% | **8x more samples** than stratified |
| 200.0 | 445 | 25.0% | **7x more samples** than stratified |

**Benefit:** All glucose ranges are tested equally, revealing model performance across the full spectrum.

## Usage

### Stratified Sampling (Default)

```bash
# Preserves original distribution
python split_data_by_case.py --subset 0.1 --case_ids 94

# Or explicitly specify stratified
python split_data_by_case.py --subset 0.1 --sampling stratified --case_ids 94
```

**Output:** 89% of samples will have glucose=88.0 (realistic distribution)

### Balanced Sampling

```bash
# Equal representation across all glucose values
python split_data_by_case.py --subset 0.1 --sampling balanced --case_ids 94
```

**Output:** 25% of samples for each glucose value (comprehensive testing)

## Detailed Comparison

### Scenario: 10% Subset of Case 94

```bash
# Stratified (realistic)
python split_data_by_case.py \
  --input_dir new_data_format/top_50_cases \
  --subset 0.1 \
  --sampling stratified \
  --case_ids 94

# Balanced (comprehensive)
python split_data_by_case.py \
  --input_dir new_data_format/top_50_cases \
  --subset 0.1 \
  --sampling balanced \
  --case_ids 94
```

### Results Comparison

| Metric | Stratified | Balanced |
|--------|------------|----------|
| Total samples | 1,783 | 1,782 |
| Glucose 88.0 | 1,588 (89%) | 446 (25%) |
| Glucose 127.0 | 82 (5%) | 446 (25%) |
| Glucose 148.0 | 53 (3%) | 445 (25%) |
| Glucose 200.0 | 60 (3%) | 445 (25%) |
| **Min samples/value** | **53** | **445** |
| **Coverage quality** | Realistic | Comprehensive |

## Recommendations

### For Production/Final Evaluation

Use **stratified sampling**:
```bash
python split_data_by_case.py --subset 0.1 --sampling stratified --case_ids all
```

**Why:** Matches real-world distribution, provides realistic performance metrics.

### For Model Validation/Debugging

Use **balanced sampling**:
```bash
python split_data_by_case.py --subset 0.1 --sampling balanced --case_ids all
```

**Why:** Ensures adequate testing across all glucose ranges, reveals weaknesses in minority value predictions.

### For Comprehensive Testing

Create **both**:
```bash
# Realistic testing
python split_data_by_case.py --subset 0.1 --sampling stratified --output_dir cases_stratified_10pct

# Comprehensive testing
python split_data_by_case.py --subset 0.1 --sampling balanced --output_dir cases_balanced_10pct
```

Then compare model performance on both sets.

## Technical Details

### Stratified Sampling Algorithm

1. Group windows by glucose value
2. Calculate proportion of each glucose value in original data
3. Sample from each group proportionally
4. Result: Subset maintains original distribution

```python
# Example: 89% glucose=88.0 in original
# → 89% glucose=88.0 in subset
```

### Balanced Sampling Algorithm

1. Identify unique glucose values (e.g., 4 values)
2. Calculate samples per value: `total_samples / unique_values`
3. Sample equally from each glucose value group
4. Result: Equal representation regardless of original distribution

```python
# Example: 10% of 17,827 = 1,783 samples
# → 1,783 / 4 values = ~446 samples per value
```

## Common Questions

### Q: Which strategy is more accurate?

**A:** Depends on your goal:
- **Stratified** gives more accurate real-world performance metrics
- **Balanced** gives more accurate understanding of model behavior across all glucose ranges

### Q: Will balanced sampling bias my model evaluation?

**A:** Not if you understand the context:
- For **final evaluation metrics** (MAE, RMSE): Use stratified
- For **debugging and validation**: Use balanced to ensure all ranges are tested

### Q: Can I use balanced sampling for training?

**A:** Generally no. Training should use stratified or full data to learn the real distribution. Use balanced for **inference testing only**.

### Q: What if I have only 1 unique glucose value?

**A:** Both strategies automatically fall back to random sampling when there's only one unique value.

## Example Workflow

### Step 1: Create Both Subset Types

```bash
# 10% stratified (realistic)
python split_data_by_case.py --subset 0.1 --sampling stratified --output_dir cases_realistic_10pct

# 10% balanced (comprehensive)
python split_data_by_case.py --subset 0.1 --sampling balanced --output_dir cases_comprehensive_10pct
```

### Step 2: Test on Stratified (Realistic Metrics)

```bash
python test_model_with_normalization.py \
  --test_data cases_realistic_10pct/case_94
```

Expected: Metrics dominated by glucose=88.0 performance (realistic)

### Step 3: Test on Balanced (Comprehensive Validation)

```bash
python test_model_with_normalization.py \
  --test_data cases_comprehensive_10pct/case_94
```

Expected: Reveals model performance on minority glucose values (127, 148, 200)

### Step 4: Compare Results

- If balanced MAE >> stratified MAE → Model struggles with minority values
- If balanced MAE ≈ stratified MAE → Model performs well across all ranges

## Performance Impact

Both strategies have similar performance (O(n) time, O(n) space):
- Stratified: Slightly faster (proportional sampling)
- Balanced: Slightly more memory (needs to track all glucose values)

Difference is negligible in practice.

## Summary

| Aspect | Stratified | Balanced |
|--------|------------|----------|
| **Distribution** | Realistic | Uniform |
| **Use Case** | Production testing | Validation/debugging |
| **Pros** | Real-world metrics | Comprehensive coverage |
| **Cons** | May miss minority values | Not realistic distribution |
| **Best For** | Final evaluation | Model analysis |

**Recommendation:** Use both! Test with balanced for validation, report with stratified for realistic metrics.
