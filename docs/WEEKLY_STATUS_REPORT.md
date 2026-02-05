# PPG-to-Glucose Prediction Status (Jan 6, 2026)

## SLIDE 1: Performance Summary

| Metric | Model 1 (Dec 31) | Model 2 (Jan 5) | Target |
|--------|------------------|-----------------|---------|
| **Validation MAE** | 29.35 mg/dL | 27.42 mg/dL | <15 mg/dL |
| **Test MAE** | 51.56 mg/dL | 69.59 mg/dL | <15 mg/dL |
| **ISO Accuracy** | 25.4% | ~30% | >95% |
| **Status** | ❌ FAILED | ❌ FAILED | - |

**Root Cause**: Training data biased to diabetic range (mean=170.54 mg/dL). Both models fail on normal/low glucose (80-140 mg/dL). Model 2's L1 regularization (0.001) + weighted sampling too weak. Layer 4 drops all morphological features (information bottleneck).

---

## SLIDE 2: Critical Actions

### Priority 1 (Immediate)
1. **Balanced Data**: 30% normal (80-100), 30% high-normal (100-120), 20% prediabetes, 20% diabetic → target mean 115 mg/dL
2. **10x Stronger Regularization**: L1 0.001→0.01, dropout 0.25, layer-wise learning rates
3. **Fix Architecture**: Add layer 3 auxiliary loss, morphological attention, LeakyReLU

### Priority 2 (This Week)
4. **Stratified Batching**: Equal samples per batch from all 6 glucose ranges
5. **Range-Specific Validation**: Track MAE separately for 60-80, 80-100, 100-120, 120-140, 140-180, 180-250 mg/dL

**Expected Outcome**: MAE 51-70→12-18 mg/dL (75-83% improvement), ISO accuracy 25%→>95%
