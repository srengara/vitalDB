# VitalDB Glucose Prediction - Training Roadmap

## Objective
Systematically scale up model training from 5 cases to optimal dataset size, achieving clinical-grade glucose prediction accuracy (MAE < 15 mg/dL).

---

## Milestone 0 (M0): Foundation & Pipeline Validation ✅ COMPLETED

**Status:** ✅ COMPLETED (2025-12-09)

**Objective:** Build and validate complete pipeline with 5 training cases and 10 test cases.

### What Was Built:
- ✅ Training pipeline (`train_glucose_predictor.py`)
- ✅ Model architecture (ResNet34-1D, 7.2M parameters)
- ✅ Inference system with normalization (`test_model_with_normalization.py`)
- ✅ Data preprocessing tools (`generate_training_data.py`, `combine_training_data.py`)
- ✅ Trained model checkpoint (`best_model.pth`)
- ✅ Complete documentation

### Results:
- **Training:** 5 cases, ~60K PPG windows, converged in 4 epochs
- **Testing:** 10 cases validated successfully
- **MAE:** 1.24 mg/dL (on constant glucose - validates pipeline works)

### Key Lessons:
1. Pipeline works end-to-end ✅
2. Normalization is critical (found and fixed bug)
3. Need variable glucose labels from Labs.csv (not constant values)
4. ResNet34-1D architecture is suitable

**📄 Full Details:** See [MILESTONE_0_FOUNDATION.md](MILESTONE_0_FOUNDATION.md)

---

## Phase 0: Data Preparation (PREREQUISITE)

### Tasks:
1. **Download Labs.csv** from PhysioNet
   - Contains ~35,358 glucose measurements across 6,388 cases
   - 4 glucose values per case on average

2. **Data Quality Analysis**
   - Identify cases with multiple intraoperative glucose readings
   - Filter cases with <2 glucose measurements (insufficient for learning)
   - Calculate glucose value distribution and variability

3. **Time Alignment**
   - Match PPG window timestamps with glucose measurement times
   - Create interpolated/nearest-neighbor glucose labels for each PPG window
   - Handle sparse glucose measurements (e.g., one reading every 30-60 minutes)

4. **Train/Val/Test Split**
   - **Training**: 70% of cases (never seen in validation/test)
   - **Validation**: 15% of cases (for hyperparameter tuning, early stopping)
   - **Test**: 15% of cases (final evaluation, never touched until end)

**Expected Output:**
- `training_data_aligned/` directory with time-aligned PPG-glucose pairs
- `data_quality_report.txt` with statistics on glucose variability
- Fixed train/val/test case IDs

**Time Estimate:** 1-2 weeks (includes manual PhysioNet download and processing)

---

## Milestone 1 (M1): Proof of Concept - 50 Cases

**Target Date:** December 2024 (Rest of Month)

### Goal
Verify that the model can learn from **variable glucose data** (not constant values).

### Training Setup
- **Cases**: 50 high-quality cases (most glucose readings per case)
- **Split**: 35 train / 8 val / 7 test
- **Epochs**: 50
- **Expected Samples**: ~600K PPG windows (assuming 12K windows/case)

### Success Criteria
- ✅ MAE < 30 mg/dL on test set
- ✅ Model learns glucose variability (std > 0)
- ✅ Loss decreases steadily (no plateaus early)
- ✅ Training completes in <24 hours

### Deliverables
- Trained model: `m1_50cases_best_model.pth`
- Training curves: Loss, MAE, RMSE vs epochs
- Test results: predictions.csv, metrics.txt
- Analysis: Error distribution, prediction vs actual plots

**Time Estimate:** 3-5 days (training + analysis)

---

## Milestone 2 (M2): Small-Scale Validation - 200 Cases

**Target Date:** January 2025

### Goal
Confirm model scales to larger dataset without overfitting.

### Training Setup
- **Cases**: 200 cases
- **Split**: 140 train / 30 val / 30 test
- **Epochs**: 100 (with early stopping patience=20)
- **Expected Samples**: ~2.4M PPG windows

### Success Criteria
- ✅ MAE < 20 mg/dL on test set
- ✅ Validation loss continues to improve (not overfitting)
- ✅ R² > 0.5 (moderate correlation)
- ✅ Training completes in <3 days

### Hyperparameter Tuning
Experiment with:
- Learning rates: [0.001, 0.0005, 0.0001]
- Batch sizes: [32, 64, 128]
- Dropout rates: [0.3, 0.5, 0.7]
- Optimizers: [Adam, AdamW]

### Deliverables
- Best model: `m2_200cases_best_model.pth`
- Hyperparameter comparison report
- Cross-validation results (if time permits)

**Time Estimate:** 1-2 weeks (multiple training runs)

---

## Milestone 3 (M3): Medium-Scale Training - 500 Cases

**Target Date:** January 2025

### Goal
Approach production-ready accuracy with substantial data.

### Training Setup
- **Cases**: 500 cases
- **Split**: 350 train / 75 val / 75 test
- **Epochs**: 150 (early stopping patience=30)
- **Expected Samples**: ~6M PPG windows

### Success Criteria
- ✅ MAE < 15 mg/dL on test set (clinical acceptable threshold)
- ✅ RMSE < 20 mg/dL
- ✅ R² > 0.7 (strong correlation)
- ✅ Clarke Zone A+B > 85% (clinically safe predictions)

### Advanced Techniques
- Data augmentation: jittering, scaling PPG signals
- Regularization: L2 weight decay, dropout tuning
- Learning rate scheduling: ReduceLROnPlateau or Cosine Annealing
- Mixed precision training (if GPU available)

### Deliverables
- Production candidate: `m3_500cases_best_model.pth`
- Clinical validation report
- Error analysis by glucose range (hypoglycemia, normal, hyperglycemia)

**Time Estimate:** 2-3 weeks (training + extensive evaluation)

---

## Milestone 4 (M4): Large-Scale Training - 1000 Cases

**Target Date:** February 2025

### Goal
Maximize model generalization with large-scale data.

### Training Setup
- **Cases**: 1000 cases
- **Split**: 700 train / 150 val / 150 test
- **Epochs**: 200 (early stopping patience=40)
- **Expected Samples**: ~12M PPG windows

### Success Criteria
- ✅ MAE < 12 mg/dL on test set
- ✅ RMSE < 18 mg/dL
- ✅ R² > 0.8 (very strong correlation)
- ✅ Clarke Zone A > 70%, Zone A+B > 90%

### Evaluation
- **Subgroup Analysis**: Performance by age, gender, surgery type
- **Temporal Stability**: Does accuracy degrade over time?
- **Edge Cases**: Hypoglycemia (<70 mg/dL), hyperglycemia (>180 mg/dL)
- **Robustness**: Test on different PPG devices (if available)

### Deliverables
- Final model: `m4_1000cases_best_model.pth`
- Comprehensive clinical validation report
- Manuscript-ready figures and tables
- Model interpretability analysis (attention maps, feature importance)

**Time Estimate:** 3-4 weeks

---

## Milestone 5 (M5): Final Model - Optional Full Dataset

### Goal
Determine if additional data improves performance (may saturate).

### Training Setup
- **Cases**: 2000-3000 cases (or until performance plateaus)
- **Split**: 70% train / 15% val / 15% test
- **Epochs**: 300 (with checkpointing)

### Decision Point
**STOP training if:**
- MAE improvement < 0.5 mg/dL compared to M4
- Training time exceeds 2 weeks
- Validation loss plateaus for 50+ epochs

**Continue to full 6,388 cases ONLY if:**
- Significant improvement expected (based on learning curves)
- Computational resources allow (GPU cluster available)
- Research paper requires full dataset claim

### Expected Outcome
- MAE: 10-12 mg/dL (clinical gold standard)
- R²: 0.85-0.90
- Publication-ready model

**Time Estimate:** 4-8 weeks (if pursued)

---

## Recommended Approach

### **Best Strategy: Progressive Scaling**

1. **Start Small** (M1: 50 cases)
   - Fast iteration
   - Identify data quality issues early
   - Validate pipeline end-to-end

2. **Scale Gradually** (M2-M4: 200 → 500 → 1000 cases)
   - Monitor performance gains at each step
   - Plot learning curve: MAE vs number of training cases
   - Stop when gains diminish (law of diminishing returns)

3. **Decide on Full Dataset**
   - If MAE drops significantly with each doubling of data → continue
   - If MAE plateaus after 1000 cases → stop there
   - Most likely, **1000-2000 cases will be optimal**

### **Why NOT Train All 6,388 Cases Immediately?**

❌ **Overfitting Risk**
- More data doesn't always help if data quality is poor
- Model may memorize noise in larger dataset

❌ **Computational Cost**
- Training 6,388 cases could take 2-4 weeks
- Wasted time if hyperparameters aren't tuned

❌ **Diminishing Returns**
- Research shows performance plateaus after a certain dataset size
- Better to optimize on smaller dataset first

✅ **Progressive approach allows:**
- Faster experimentation
- Early detection of issues
- Efficient use of compute resources

---

## Expected Performance Trajectory

| Milestone | Cases | Expected MAE | Expected R² | Training Time |
|-----------|-------|--------------|-------------|---------------|
| M1 | 50 | 25-30 mg/dL | 0.3-0.4 | 1 day |
| M2 | 200 | 18-22 mg/dL | 0.5-0.6 | 2-3 days |
| M3 | 500 | 12-16 mg/dL | 0.7-0.8 | 5-7 days |
| M4 | 1000 | 10-13 mg/dL | 0.8-0.85 | 10-14 days |
| M5 (opt) | 2000-6388 | 8-12 mg/dL | 0.85-0.90 | 2-8 weeks |

**Clinical Benchmarks:**
- **Excellent**: MAE < 10 mg/dL, Clarke Zone A > 80%
- **Good**: MAE < 15 mg/dL, Clarke Zone A+B > 85%
- **Acceptable**: MAE < 20 mg/dL, Clarke Zone A+B > 80%

---

## Implementation Checklist

### Phase 0: Preparation
- [ ] Download Labs.csv from PhysioNet
- [ ] Run data quality analysis script
- [ ] Create time-aligned training data
- [ ] Split into train/val/test sets (fixed)
- [ ] Document glucose value statistics

### M1: 50 Cases
- [ ] Select 50 high-quality cases
- [ ] Train ResNet34-1D model
- [ ] Evaluate on test set
- [ ] Analyze errors and create report
- [ ] **Decision: Proceed to M2?**

### M2: 200 Cases
- [ ] Scale to 200 cases
- [ ] Run hyperparameter sweep
- [ ] Compare with M1 performance
- [ ] **Decision: Proceed to M3?**

### M3: 500 Cases
- [ ] Scale to 500 cases
- [ ] Implement data augmentation
- [ ] Clinical validation (Clarke Error Grid)
- [ ] **Decision: Proceed to M4 or stop here?**

### M4: 1000 Cases
- [ ] Scale to 1000 cases
- [ ] Comprehensive evaluation
- [ ] Subgroup analysis
- [ ] **Decision: Proceed to M5 or finalize?**

### M5: Full Dataset (Optional)
- [ ] Plot learning curve from M1-M4
- [ ] Extrapolate expected gain from full dataset
- [ ] Train on full dataset if justified
- [ ] Final model evaluation

---

## Key Scripts Needed

### 1. Data Preparation
```bash
python prepare_training_data_from_labs.py \
    --labs_csv data/labs/Labs.csv \
    --ppg_dir data/web_app_data/ \
    --output_dir training_data_aligned/ \
    --min_glucose_readings 2
```

### 2. Training
```bash
python src/training/train_glucose_predictor.py \
    --data_dir training_data_aligned/m1_50cases \
    --output_dir training_outputs/m1_50cases \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001
```

### 3. Evaluation
```bash
python test_model_with_normalization.py \
    --model_path training_outputs/m1_50cases/best_model.pth \
    --test_data training_data_aligned/m1_50cases/test/
```

---

## Success Metrics Dashboard

Track these metrics across all milestones:

### Model Performance
- ✅ **MAE** (Mean Absolute Error)
- ✅ **RMSE** (Root Mean Squared Error)
- ✅ **R² Score** (Coefficient of Determination)
- ✅ **MAPE** (Mean Absolute Percentage Error)

### Clinical Metrics
- ✅ **Clarke Error Grid Zone A** (clinically accurate)
- ✅ **Clarke Error Grid Zone A+B** (clinically acceptable)
- ✅ **Hypoglycemia Detection Rate** (<70 mg/dL)
- ✅ **Hyperglycemia Detection Rate** (>180 mg/dL)

### Training Metrics
- ✅ Training time (hours)
- ✅ Model size (MB)
- ✅ Inference speed (samples/sec)
- ✅ GPU memory usage

---

## Project Timeline (December 2024 - February 2025)

| Milestone | Target Date | Cases | Expected MAE | Duration |
|-----------|-------------|-------|--------------|----------|
| M0: Foundation | ✅ Dec 9, 2024 | 5 | 1.24 mg/dL | COMPLETED |
| Phase 0: Data Prep | Dec 10-20, 2024 | - | - | 1-2 weeks |
| M1: Proof of Concept | Dec 21-31, 2024 | 50 | 25-30 mg/dL | 3-5 days |
| M2: Small-Scale | Jan 1-15, 2025 | 200 | 18-22 mg/dL | 1-2 weeks |
| M3: Medium-Scale | Jan 16-31, 2025 | 500 | 12-16 mg/dL | 2-3 weeks |
| M4: Large-Scale | Feb 1-28, 2025 | 1000 | 10-13 mg/dL | 3-4 weeks |
| M5: Full Dataset (opt) | Mar+ 2025 | 2000+ | 8-12 mg/dL | 4-8 weeks |

**🎯 Primary Target: Fully trained VitalDB model by end of February 2025**

**Note:** Sensor integration work can begin in parallel with M2/M3 milestones (January 2025).

---

## Next Immediate Steps

1. ✅ **Download Labs.csv** (requires PhysioNet account)
2. ✅ **Analyze glucose data quality** (how many cases have >2 readings?)
3. ✅ **Create time-alignment script** (match PPG windows to glucose timestamps)
4. ✅ **Select M1 cases** (50 cases with best glucose coverage)
5. ✅ **Train M1 model** (first real test with variable glucose!)

---

## Questions to Answer During Roadmap

- **Q1**: How many cases have ≥3 glucose readings? (determines max usable dataset)
- **Q2**: Does performance scale linearly with data? (plot MAE vs cases)
- **Q3**: What's the optimal train/val/test split ratio?
- **Q4**: Should we use data augmentation? (jitter, noise, scaling)
- **Q5**: Is ResNet34 the right architecture or should we try Transformers?

---

## References

- VitalDB Paper: https://www.nature.com/articles/s41597-022-01411-5
- Clarke Error Grid: https://en.wikipedia.org/wiki/Clarke_Error_Grid
- Best Practices in Medical ML: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device

---

**Created**: 2025-12-09
**Last Updated**: 2025-12-09
**Status**: M0 Completed - Moving to Phase 0 (Data Preparation)

---

## Parallel Work Streams

While progressing through the training milestones, the following parallel work can begin:

### Sensor Integration (Starting January 2025)
- Can begin alongside M2/M3 milestones
- Design multi-sensor data fusion architecture
- Integrate additional physiological signals (ECG, SpO2, etc.)
- Develop unified data collection pipeline
- Test sensor synchronization and timing alignment

**Timeline:** The sensor integration work can proceed independently while VitalDB training continues. By February 2025, both streams should converge into a unified multi-sensor glucose prediction system.
