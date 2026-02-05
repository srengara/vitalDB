# Glucose Range Performance Analysis & Recommendations

Generated: 2025-12-22

## Problem Statement
Model shows poor performance (MAE ~64 mg/dL overall) despite normalization fix. Need to improve performance specifically in the 40-450 mg/dL range.

---

## Current Performance Analysis

### Test Set Performance by Glucose Range

| Glucose Range | Cases | Avg MAE (mg/dL) | Performance |
|---------------|-------|-----------------|-------------|
| 70-100 (Normal) | 9 | 58.97 | POOR |
| 100-126 (Pre-diabetic) | 8 | 63.91 | POOR |
| **126-200 (Diabetic)** | **9** | **29.28** | **GOOD** ✓ |
| 200-300 (High) | 13 | 96.96 | VERY POOR |
| 300+ (Extreme) | 9 | 47.06 | POOR |

**Key Observation:** Model performs well ONLY in the 126-200 mg/dL range!

### Training Data Distribution

| Glucose Range | Samples | Percentage |
|---------------|---------|------------|
| <70 (Hypo) | 1,544 | 0.21% |
| 70-100 (Normal) | 196,456 | 27.32% |
| 100-126 (Pre-diabetic) | 100,556 | 13.98% |
| **126-200 (Diabetic)** | **146,560** | **20.38%** |
| 200-300 (High) | 102,363 | 14.24% |
| 300+ (Extreme) | 96,789 | 13.46% |

**Total samples:** 719,074
**Glucose range:** 58-845 mg/dL
**Mean:** 229.8 mg/dL, **Std:** 176.0 mg/dL

---

## Root Cause Analysis

### Why Performance is Poor

1. **Distribution Mismatch:**
   - Training has MANY samples in 70-100 range (27.32%)
   - But test performance is POOR in this range (MAE: 59 mg/dL)
   - This suggests the model hasn't learned the PPG → glucose relationship well in normal ranges

2. **Best Performance in 126-200 Range:**
   - Training: 20.38% of data
   - Test MAE: 29.28 mg/dL (acceptable)
   - This range has good representation AND good learning

3. **Poor Performance in 200+ Range:**
   - Training: 27.70% of data (200-300 + 300+)
   - Test MAE: 96.96 mg/dL (200-300), 47.06 mg/dL (300+)
   - Despite having sufficient data, model fails

### Hypothesis: Discrete Glucose Values Problem

Looking at top glucose values:
- **97.0 mg/dL:** 34,673 samples (4.82%)
- **87.0 mg/dL:** 22,638 samples (3.15%)
- **292.0, 222.0, 377.0 mg/dL:** ~19,872 samples each

The training data has only **132 unique glucose values** out of 719K samples. This means:
- Model is learning to predict **discrete buckets**, not continuous values
- PPG signals might be very similar but model forces prediction to nearest training value
- This causes quantization errors, especially when test glucose values fall between training values

---

## Recommendations

### Option 1: Balanced Glucose Sampling (RECOMMENDED)

**Pros:**
- Ensures equal representation across ALL glucose values
- Prevents model from overfitting to common glucose values
- Forces model to learn PPG patterns for each glucose level equally
- Already implemented in your data splitter (`--sampling balanced`)

**Cons:**
- Reduces total training samples
- May lose some PPG signal variations within same glucose value

**Implementation:**
```bash
# Generate balanced training set
python split_data_by_case.py \
    --input_ppg C:\IITM\vitalDB\new_data_format\top_50_cases\ppg_windows.csv \
    --input_glucose C:\IITM\vitalDB\new_data_format\top_50_cases\glucose_labels.csv \
    --output_dir C:\IITM\vitalDB\balanced_training_data \
    --case_ids all \
    --subset_fraction 1.0 \
    --sampling balanced \
    --random_seed 42
```

**Expected Outcome:**
- Each of 132 unique glucose values gets equal samples
- Model learns to distinguish PPG patterns for each glucose level
- Should reduce bias toward common values (97, 87 mg/dL)

---

### Option 2: Add More Cases in 40-450 Range

**Pros:**
- More data generally helps
- Can capture more PPG variability within each glucose range
- Preserves natural glucose distribution

**Cons:**
- May not solve the discrete value problem
- If new cases also have similar discrete values, won't help much
- Requires more data collection/labeling effort

**When to use:**
- If you have access to more cases with continuous glucose values
- If current cases have limited PPG diversity
- As a supplement to balanced sampling

---

### Option 3: Hybrid Approach (BEST)

Combine balanced sampling with data augmentation:

1. **Balance existing data:** Use balanced sampling on current 719K samples
2. **Stratified split:** Ensure train/val/test have all glucose ranges
3. **Add targeted cases:** If possible, add cases with:
   - Glucose values between existing discrete values (e.g., 89, 93 mg/dL)
   - More PPG diversity in poorly performing ranges (70-100, 200-300)

**Implementation Steps:**

```bash
# Step 1: Create balanced dataset
python split_data_by_case.py \
    --input_ppg C:\IITM\vitalDB\new_data_format\top_50_cases\ppg_windows.csv \
    --input_glucose C:\IITM\vitalDB\new_data_format\top_50_cases\glucose_labels.csv \
    --output_dir C:\IITM\vitalDB\balanced_training_data \
    --case_ids all \
    --subset_fraction 1.0 \
    --sampling balanced

# Step 2: Retrain model
python src/training/train_glucose_predictor.py \
    --ppg_file C:\IITM\vitalDB\balanced_training_data\ppg_windows.csv \
    --glucose_file C:\IITM\vitalDB\balanced_training_data\glucose_labels.csv \
    --output_dir C:\IITM\vitalDB\model_balanced

# Step 3: Test on same test set
python test_model_with_normalization.py \
    --model_path C:\IITM\vitalDB\model_balanced\best_model.pth \
    --test_data <test_case_dir>

# Step 4: Compare results
```

---

### Option 4: Multi-Head Prediction (Advanced)

Train separate models for different glucose ranges:
- **Model A:** 40-150 mg/dL (Normal-Prediabetic)
- **Model B:** 150-300 mg/dL (Diabetic-High)
- **Model C:** 300-450 mg/dL (Extreme)

Route predictions based on initial glucose estimate or use ensemble.

**Pros:**
- Each model specializes in its range
- Can handle range-specific PPG characteristics

**Cons:**
- More complex pipeline
- Need routing mechanism
- 3x training/maintenance cost

---

## Immediate Action Plan

### Priority 1: Test Balanced Sampling (1-2 days)

1. Generate balanced training dataset
2. Retrain model with same hyperparameters
3. Run inference on same 50 test cases
4. Compare MAE by glucose range vs current model

**Success Criteria:**
- MAE in 70-100 range improves from 59 to <30 mg/dL
- MAE in 200-300 range improves from 97 to <40 mg/dL
- Overall MAE improves from 64 to <30 mg/dL

### Priority 2: Analyze Balanced Model Results (1 day)

1. Run the same analysis script on new predictions
2. Check if performance is more uniform across ranges
3. Identify remaining problem ranges

### Priority 3: Iterative Improvement (ongoing)

Based on balanced model results:
- If still poor in specific ranges → investigate PPG signal quality in those cases
- If discrete value problem persists → consider adding interpolated glucose values
- If overall improvement insufficient → consider multi-head or ensemble approach

---

## Expected Improvements

### Conservative Estimate (Balanced Sampling)
- **70-100 mg/dL:** MAE 59 → 25-35 mg/dL
- **100-126 mg/dL:** MAE 64 → 25-35 mg/dL
- **126-200 mg/dL:** MAE 29 → 20-25 mg/dL (already good)
- **200-300 mg/dL:** MAE 97 → 35-50 mg/dL
- **300+ mg/dL:** MAE 47 → 30-40 mg/dL
- **Overall:** MAE 64 → 30-40 mg/dL

### Optimistic Estimate (Balanced + More Cases)
- **Overall:** MAE 64 → 15-25 mg/dL (clinically acceptable)

---

## Why Balanced Sampling Should Work

1. **Addresses Bias:** Current model overfits to 97 mg/dL (4.82% of training data)
2. **Equal Learning:** Each glucose value gets equal training signal
3. **Better Generalization:** Forces model to learn PPG → glucose mapping, not memorize common values
4. **Proven Technique:** Similar to class balancing in classification tasks
5. **Low Risk:** Easy to test, reversible if doesn't work

---

## Monitoring Metrics

After implementing balanced sampling, track:
- **MAE by glucose range** (as shown above)
- **MAE by glucose value** (check if specific values still perform poorly)
- **Prediction distribution** (ensure predictions are diverse, not clustered)
- **R² score** (should become positive and >0.5)
- **Standard deviation of predictions** (should match test set std, not be compressed)

---

## Alternative: Check PPG Signal Quality

Before adding more data, verify PPG signal quality in poorly performing ranges:

```python
# Check if PPG signals differ by glucose range
import pandas as pd
import numpy as np

ppg_df = pd.read_csv('ppg_windows.csv')
glucose_df = pd.read_csv('glucose_labels.csv')

# Group by glucose range
for low, high in [(70, 100), (100, 126), (200, 300)]:
    mask = (glucose_df['glucose_mg_dl'] >= low) & (glucose_df['glucose_mg_dl'] < high)
    ppg_subset = ppg_df[mask].iloc[:, 3:].values  # amplitude columns

    print(f'Glucose {low}-{high}:')
    print(f'  PPG mean: {ppg_subset.mean():.2f}')
    print(f'  PPG std: {ppg_subset.std():.2f}')
    print(f'  PPG range: {ppg_subset.min():.2f} - {ppg_subset.max():.2f}')
```

If PPG signals look similar across ranges, the problem is likely learning/data imbalance, not signal quality.

---

## Conclusion

**Recommended approach:** Start with balanced sampling (Option 3 - Hybrid Approach)

This addresses the likely root cause (discrete value bias) with minimal risk and effort. If balanced sampling doesn't achieve <30 mg/dL MAE, then consider adding more diverse cases or multi-head models.

**Next steps:**
1. Generate balanced training dataset
2. Retrain model
3. Evaluate on same test cases
4. Generate new inference report
5. Compare before/after performance
