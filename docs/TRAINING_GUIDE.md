# Training Guide: Single Case vs. Multiple Cases

## Quick Answer

✅ **Always combine multiple cases** for better model performance and generalization.

## Why Combine Multiple Cases?

### Machine Learning Fundamentals

Your goal is to build a model that predicts glucose from PPG signals **for any patient**, not just one specific patient.

**Single Case Training Problems:**
- Model learns patient-specific patterns (e.g., "this person's PPG looks like X")
- Overfits to individual physiology
- Won't work on new patients
- Very limited data (250-500 windows)

**Multi-Case Training Benefits:**
- Model learns universal PPG→glucose relationship
- Generalizes across different patients
- More diverse training data
- Better performance on unseen patients

### Example Comparison

| Metric | Single Case | Combined Cases (10+) |
|--------|-------------|----------------------|
| Training data | 250 windows | 2,500+ windows |
| Generalization | Poor | Good |
| Test accuracy (new patient) | ~30-40% error | ~10-15% error |
| Overfitting risk | Very high | Low |

## Step-by-Step: Training with Multiple Cases

### Step 1: Generate Training Data for Multiple Cases

Use the web app or CLI to generate data for multiple cases:

```bash
# Generate data for cases 1-10
python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose auto --output ./data/web_app_data/case_1_SNUADC_PLETH
python generate_training_data.py --case_id 2 --track SNUADC/PLETH --glucose auto --output ./data/web_app_data/case_2_SNUADC_PLETH
python generate_training_data.py --case_id 3 --track SNUADC/PLETH --glucose auto --output ./data/web_app_data/case_3_SNUADC_PLETH
# ... continue for more cases
```

**Recommendation**: Generate data for at least **10-20 cases** for decent performance, **50+ cases** for good performance.

### Step 2: Combine All Cases into One Dataset

```bash
# Combine all case_* directories
python combine_training_data.py --input_dirs ./data/web_app_data/case_* --output ./training_data_combined
```

**Output:**
```
Combining Training Data from Multiple Cases
======================================================================
Processing: case_1_SNUADC_PLETH
  Windows: 250
  Glucose: 95.0 mg/dL

Processing: case_2_SNUADC_PLETH
  Windows: 320
  Glucose: 110.0 mg/dL

Processing: case_3_SNUADC_PLETH
  Windows: 280
  Glucose: 88.0 mg/dL

...

Combined Dataset Summary
======================================================================
Total cases: 10
Total windows: 2,850
Glucose range: 70.0 - 150.0 mg/dL

✓ Saved: ./training_data_combined/ppg_windows.csv
✓ Saved: ./training_data_combined/glucose_labels.csv
```

### Step 3: Train the Model

```bash
python -m src.training.train_glucose_predictor --data_dir ./training_data_combined --epochs 100
```

The training script automatically:
- Splits data into train (70%), validation (15%), test (15%)
- The split ensures **windows from the same case** can be in different sets
- This tests if the model generalizes to **different time periods** from the same patient

### Step 4: Evaluate on Completely New Patients

For true generalization testing, reserve some cases for final testing:

```bash
# Training: cases 1-15
python combine_training_data.py --input_dirs ./data/web_app_data/case_{1..15}_* --output ./training_data

# Testing: cases 16-20 (never seen during training)
python combine_training_data.py --input_dirs ./data/web_app_data/case_{16..20}_* --output ./test_data

# Train on cases 1-15
python -m src.training.train_glucose_predictor --data_dir ./training_data --epochs 100

# Evaluate on unseen cases 16-20
python -m src.training.train_glucose_predictor --data_dir ./test_data --test_only --model_path ./output/best_model.pth
```

## Data Diversity Recommendations

### Minimum Viable Dataset

- **Number of cases**: 10-15 cases minimum
- **Total windows**: 2,000+ windows
- **Glucose diversity**: Include cases with varying glucose levels (70-180 mg/dL range)

### Good Dataset

- **Number of cases**: 30-50 cases
- **Total windows**: 8,000-15,000 windows
- **Glucose diversity**: Wide range of glucose levels
- **Patient diversity**: Different ages, BMI, health conditions

### Excellent Dataset

- **Number of cases**: 100+ cases
- **Total windows**: 30,000+ windows
- **Glucose diversity**: Full physiological range (50-300 mg/dL)
- **Temporal diversity**: Different times of day, pre/post-op measurements

## Common Mistakes to Avoid

### ❌ Mistake 1: Training on One Case

```bash
# DON'T DO THIS
python -m src.training.train_glucose_predictor --data_dir ./data/web_app_data/case_1_SNUADC_PLETH
```

**Problem**: Model only learns from one patient, won't generalize.

### ❌ Mistake 2: Not Enough Data Diversity

```bash
# DON'T: All cases have the same glucose value
case 1: glucose = 95.0
case 2: glucose = 95.0
case 3: glucose = 95.0
```

**Problem**: Model can't learn the PPG→glucose relationship if glucose never varies.

### ❌ Mistake 3: Data Leakage in Testing

```bash
# DON'T: Test on data from training cases
# If case 1 is in training, don't use case 1 windows for final testing
```

**Problem**: Artificially inflated performance metrics.

## Recommended Workflow

### For Research/Development

```bash
# 1. Generate data for 20 cases
for i in {1..20}; do
    python generate_training_data.py --case_id $i --track SNUADC/PLETH --glucose auto --output ./data/cases/case_${i}
done

# 2. Combine cases 1-15 for training
python combine_training_data.py --input_dirs ./data/cases/case_{1..15} --output ./training_data

# 3. Combine cases 16-20 for final testing
python combine_training_data.py --input_dirs ./data/cases/case_{16..20} --output ./test_data

# 4. Train model
python -m src.training.train_glucose_predictor --data_dir ./training_data --epochs 100

# 5. Evaluate on held-out test set
python -m src.training.train_glucose_predictor --data_dir ./test_data --test_only --model_path ./output/best_model.pth
```

### For Quick Testing (Minimal Dataset)

```bash
# 1. Generate data for 5 cases
for i in {1..5}; do
    python generate_training_data.py --case_id $i --track SNUADC/PLETH --glucose auto --output ./data/cases/case_${i}
done

# 2. Combine all cases
python combine_training_data.py --input_dirs ./data/cases/case_* --output ./training_data_quick

# 3. Train (will use automatic train/val/test split)
python -m src.training.train_glucose_predictor --data_dir ./training_data_quick --epochs 50
```

## Key Takeaway

**Always combine multiple cases** for training. The model needs to see diverse PPG patterns from different patients to learn the true PPG→glucose relationship, not patient-specific patterns.

Think of it like learning to recognize faces:
- Training on 1 person = only recognize that person
- Training on 100 people = recognize faces in general
