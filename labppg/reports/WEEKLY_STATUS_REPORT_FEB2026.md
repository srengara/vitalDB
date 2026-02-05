# Weekly Status Report
## PPG-to-Glucose Prediction Project
**Week Ending: February 5, 2026**

---

> **TL;DR:** Model 143853 achieves 26.1 mg/dL MAE with 94.2% clinical safety (Clarke A+B). Lab validation on 81 samples revealed BP as a key confound—filtering normotensive subjects improves MAE by 19%. Multi-channel pipeline operational. Kicked off Risk Assessment & FMEA engagement with Agrani Solutions. Priority: expand lab data, integrate BP features, address extreme glucose ranges.

---

## 1. Summary Dashboard

| Metric | Current Value | Target | Status |
|--------|--------------|--------|--------|
| Model MAE | 26.1 mg/dL | <20 mg/dL | In Progress |
| Clarke Zone A+B | 94.2% | >95% | Near Target |
| Lab PPG Test Samples | 81 | 150+ | Collecting |
| Multi-Channel Support | Complete | Complete | Done |

**Key Highlights:**
- Completed BP-constrained analysis revealing cardiovascular confounds
- Lab PPG validation underway with 81 samples across 8 subjects
- Multi-channel (Red/IR PPG) pipeline fully operational
- Model 143853 deployed as current production candidate
- **Kicked off Risk Assessment & FMEA with Agrani Solutions**

---

## 2. Lab PPG Experiments

### 2.1 Vanilla PPG Batch Testing

| Batch | Samples | Subjects | Channel | Status |
|-------|---------|----------|---------|--------|
| Batch 1 | 42 | Akshaya, Logesh, Mks, Rohan | Red & IR | Complete |
| Batch 2 | 39 | Rohit, Pramod, Sudha, Vikraman | Red & IR | Complete |
| **Total** | **81** | **8** | - | - |

### 2.2 Performance Summary (All Samples)

| Metric | Value | Assessment |
|--------|-------|------------|
| MAE | 30.6 mg/dL | Moderate |
| RMSE | 40.9 mg/dL | - |
| MARD | 27.9% | Needs Improvement |
| Clarke Zone A | 53.8% | Below Target |
| Clarke Zone A+B | 89.5% | Acceptable |

**Observation:** Lab PPG shows higher error than VitalDB training data, indicating domain shift between clinical and controlled lab environments.

---

## 3. Blood Pressure Constraints Analysis

### 3.1 Rationale
Hypertensive subjects exhibit altered PPG morphology due to vascular stiffness, potentially confounding glucose predictions. We analyzed performance with BP constraints applied.

### 3.2 Constraint Parameters
- **Systolic:** ≤ 130 mmHg
- **Diastolic:** ≤ 90 mmHg
- **Filtered Dataset:** 46 of 81 samples (57%)

### 3.3 Results Comparison

| Metric | Unconstrained (n=81) | BP Constrained (n=46) | Delta |
|--------|---------------------|----------------------|-------|
| MAE | 30.6 mg/dL | 24.8 mg/dL | -19% |
| Clarke Zone A | 53.8% | 63.0% | +9.2pp |
| Clarke Zone A+B | 89.5% | 93.5% | +4.0pp |

### 3.4 Key Finding
**BP is a significant confound.** Correlation analysis shows:
- Systolic BP correlation with error: **+0.301**
- Diastolic BP correlation with error: **+0.235**

**Recommendation:** Consider BP as an input feature or develop BP-stratified models.

---

## 4. Multi-Channel Training

### 4.1 Pipeline Status
The dual-channel PPG processing pipeline is fully operational:

| Component | Red PPG (620-700nm) | IR PPG (870-920nm) |
|-----------|--------------------|--------------------|
| Data Collection | Complete | Complete |
| Preprocessing | Operational | Operational |
| Peak Detection | Calibrated | Calibrated |
| Inference | Deployed | Deployed |

### 4.2 Channel Comparison (Lab Data)

| Metric | Red PPG | IR PPG | Better |
|--------|---------|--------|--------|
| MAE | 29.8 mg/dL | 31.4 mg/dL | Red |
| Success Rate | 87% | 91% | IR |
| Signal Quality | Good | Excellent | IR |

**Observation:** Red PPG yields slightly better accuracy; IR PPG provides more robust signal acquisition. Production deployment uses both channels with Red as primary.

---

## 5. Model Inference Report

### 5.1 Latest Models Evaluated

| Model ID | Date | Architecture | Training Data |
|----------|------|--------------|---------------|
| **143853** | Feb 2, 2026 | ResNet34-1D | VitalDB 3,123 cases |
| **163632** | Jan 28, 2026 | ResNet34-1D | VitalDB Extended |

### 5.2 Model 143853 (Current Production Candidate)

| Metric | VitalDB Test Set | Lab PPG Test |
|--------|-----------------|--------------|
| MAE | 26.1 mg/dL | 30.6 mg/dL |
| RMSE | 35.2 mg/dL | 40.9 mg/dL |
| Clarke A | 55.1% | 53.8% |
| Clarke A+B | 94.2% | 89.5% |

### 5.3 Performance by Glucose Range

| Range | MAE | Assessment |
|-------|-----|------------|
| Hypoglycemic (<70) | 50.4 mg/dL | Poor |
| Normal (70-100) | 17.2 mg/dL | Good |
| High-Normal (100-140) | 19.8 mg/dL | Good |
| Prediabetes (140-180) | 35.1 mg/dL | Moderate |
| Diabetic (≥180) | 82.5 mg/dL | Poor |

**Critical Gap:** Extreme glucose ranges (hypo/hyperglycemia) show significantly degraded performance due to data imbalance (only 1.5% hypoglycemic, 6.1% diabetic samples).

---

## 6. Next Steps

### Immediate (Week of Feb 10)
1. **Expand Lab PPG Dataset** - Target 50 additional samples focusing on diverse BP profiles
2. **BP Feature Integration** - Experiment with BP as auxiliary input to the model
3. **Peak Detection Tuning** - Reduce failure rate from 13% to <5%

### Short-Term (February 2026)
4. **Extreme Range Augmentation** - Apply synthetic data augmentation for hypo/diabetic ranges
5. **Cross-Validation Study** - Validate model generalization across subjects
6. **IR Channel Optimization** - Improve IR accuracy to match Red channel

### Medium-Term (Q1 2026)
7. **Multi-Modal Fusion** - Combine Red + IR channels in single model
8. **Transfer Learning** - Fine-tune on lab PPG data to address domain shift
9. **Clinical Pilot Preparation** - Protocol development for controlled study

---

## 7. Risk Assessment & FMEA Initiative

**Partner:** Agrani Solutions
**Status:** Kick-off completed this week

**Scope:**
- Failure Mode and Effects Analysis (FMEA) for PPG-glucose prediction system
- Risk identification across data acquisition, signal processing, and model inference
- Regulatory pathway assessment for medical device classification

**Initial Focus Areas:**
- Signal quality failures and mitigation strategies
- Model confidence thresholds for clinical deployment
- Patient safety boundaries for glucose alerts

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| BP confounding | High | BP-stratified models or BP input feature |
| Data imbalance | High | Weighted sampling + synthetic augmentation |
| Domain shift | Medium | Transfer learning on lab data |
| Peak detection failures | Low | Relaxed filtering + fallback methods |

---

**Report Prepared By:** PPG-Glucose Research Team
**Date:** February 5, 2026
**Next Review:** February 12, 2026
