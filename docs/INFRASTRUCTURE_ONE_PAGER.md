# VitalDB Training Infrastructure - One-Pager
## Hardware Selection & Cost Optimization

**Report Date:** December 10, 2025
**Purpose:** Training infrastructure planning for M1-M3 milestones

---

## Available Hardware Options (Google Colab)

| Hardware | VRAM | FP32 TFLOPS | FP16 TFLOPS | Best For | Cost Tier |
|----------|------|-------------|-------------|----------|-----------|
| **T4 GPU** | 16 GB | 8.1 | 65 | Small models, inference | $ (Free/Pro) |
| **L4 GPU** | 24 GB | 30 | 120 | Medium models, efficient training | $$ (Pro+) |
| **A100 GPU** | 40 GB | 19.5 | 312 (Tensor) | Large-scale training, highest throughput | $$$ (Premium) |

---

## Model Requirements Analysis

### ResNet34-1D for VitalDB
- **Model Size:** ~21M parameters (~84 MB)
- **Input:** 1-second PPG windows (100 samples)
- **Batch Processing:** 32-256 samples per batch
- **Memory Footprint:** ~2-4 GB (model + gradients + optimizer states)

### Training Dataset Sizes
| Milestone | Cases | PPG Windows | Glucose Labels | Est. Dataset Size | Training Time Factor |
|-----------|-------|-------------|----------------|-------------------|---------------------|
| **M1** | 50 | ~600K | Variable | ~1.2 GB | 1x baseline |
| **M2** | 200 | ~2.4M | Variable | ~4.8 GB | 4x |
| **M3** | 1000 | ~12M | Variable | ~24 GB | 20x |

---

## Hardware Recommendations by Milestone

### 🎯 M1 (50 Cases, Proof of Concept) - **Recommended: L4 GPU**

**Rationale:**
- ✅ **24 GB VRAM:** More than sufficient for 600K windows
- ✅ **4x faster than T4:** FP16 training (120 vs 65 TFLOPS)
- ✅ **Cost-effective:** ~50% cheaper than A100 for this scale
- ✅ **Batch size flexibility:** Can use larger batches (128-256) for faster training
- ✅ **Lower latency:** Direct Colab Pro+ access without queuing

**Expected Performance:**
- Training time: ~8-12 hours for 50 epochs
- Batch size: 128-256 (optimal throughput)
- Cost: ~$2-3 for full M1 training run

**Alternative: T4 GPU (Budget Option)**
- Training time: ~24-36 hours (3x slower)
- Batch size: 64-128 (limited by compute, not memory)
- Cost: Free with Colab Pro (~$10/month)
- ⚠️ Risk: Session timeouts with long runs

---

### 🚀 M2 (200 Cases, Validation) - **Recommended: L4 GPU** → **Consider A100 for final run**

**Strategy: Hybrid Approach**

**Phase 1 - Experimentation (L4 GPU):**
- Use L4 for hyperparameter tuning, architecture experiments
- ~4x dataset size vs M1, but still fits comfortably in 24 GB
- Training time: ~24-36 hours per full run
- Cost: ~$8-12 per run

**Phase 2 - Final Training (A100 GPU):**
- Switch to A100 for final production model
- 2.6x faster training (312 vs 120 FP16 TFLOPS)
- Training time: ~10-14 hours for 100 epochs
- Cost: ~$20-30 for final run
- **Benefit:** Faster iterations during critical evaluation phase

**Memory Requirements:**
- Dataset: ~4.8 GB
- Model + optimizer: ~4 GB
- Batch processing: ~8 GB (batch size 256)
- **Total: ~17 GB** (L4: ✅ Sufficient, T4: ⚠️ Tight)

---

### ⚡ M3 (1000 Cases, Production) - **Recommended: A100 GPU (Essential)**

**Rationale:**
- ✅ **40 GB VRAM:** Required for ~24 GB dataset + 16 GB training overhead
- ✅ **312 TFLOPS FP16:** 10x faster than T4, 2.6x faster than L4
- ✅ **Tensor Core optimization:** Best for large-scale deep learning
- ✅ **Higher batch sizes:** 512-1024 samples for maximum throughput
- ✅ **Multi-day training feasible:** Stable for 48-72 hour runs

**Expected Performance:**
- Training time: ~36-48 hours for 100 epochs
- Batch size: 512-1024 (optimal for A100)
- Cost: ~$80-120 for full training run
- **ROI:** Time savings justify premium cost at this scale

**Why Not L4 for M3?**
- ❌ 24 GB VRAM insufficient for full dataset + large batches
- ❌ Training time: ~120+ hours (5+ days) - impractical
- ❌ Higher risk of session timeouts and data loss

---

## Cost-Benefit Analysis

### Total Training Cost Estimates

| Milestone | Recommended HW | Training Runs | Cost per Run | Total Cost | Time to Model |
|-----------|---------------|---------------|--------------|------------|---------------|
| **M1** | L4 GPU | 3-5 runs | $2-3 | **$10-15** | 2-3 days |
| **M2** | L4 (exp) + A100 (final) | 5-8 runs | $8-12 (L4), $25 (A100) | **$70-100** | 5-7 days |
| **M3** | A100 GPU | 3-5 runs | $80-120 | **$300-500** | 7-10 days |
| | | | **TOTAL** | **$380-615** | **14-20 days** |

### Budget vs Performance Trade-offs

**Option 1: Cost-Optimized (T4 Only)**
- Total cost: $50-100 (mostly Colab Pro subscription)
- Total time: 40-60 days
- ⚠️ Risk: Session timeouts, slow iteration

**Option 2: Balanced (Recommended)**
- M1: L4 ($15)
- M2: L4 + A100 ($100)
- M3: A100 ($500)
- Total cost: **$615**
- Total time: **20 days**
- ✅ Optimal: Fast iteration, manageable cost

**Option 3: Performance-Optimized (A100 All)**
- Total cost: $800-1200
- Total time: 12-15 days
- ⚠️ Overkill for M1, marginal gains vs Option 2

---

## Infrastructure Setup Guide

### Google Colab Configuration

#### For L4 GPU (M1, M2 Experimentation)
```python
# Runtime: Python 3 + GPU (L4)
# Colab Pro+ Required

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Optimal batch size for L4
batch_size = 256  # Recommended for L4
```

#### For A100 GPU (M2 Final, M3)
```python
# Runtime: Python 3 + GPU (A100)
# Colab Pro+ or Pay-as-you-go

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Enable Tensor Core optimization
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Optimal batch size for A100
batch_size = 512  # or 1024 for M3
```

---

## Training Optimization Strategies

### Memory Optimization
1. **Gradient Accumulation:** Simulate larger batches without memory increase
   ```python
   accumulation_steps = 4  # Effective batch size = batch_size × 4
   ```

2. **Mixed Precision (FP16):** 2x memory reduction, 2-3x speedup
   ```python
   with autocast():
       output = model(input)
   ```

3. **Gradient Checkpointing:** Trade compute for memory (if needed for M3)
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

### Throughput Optimization
1. **Data Loading:** Use multiple workers
   ```python
   DataLoader(..., num_workers=4, pin_memory=True)
   ```

2. **Batch Size Tuning:**
   - T4: 64-128
   - L4: 128-256
   - A100: 256-1024

3. **Learning Rate Scaling:** Scale LR with batch size
   ```python
   lr = base_lr * (batch_size / 32)
   ```

---

## Risk Mitigation

### Session Timeout Prevention
1. **Checkpoint saving:** Every 5 epochs
2. **Resume capability:** Load from last checkpoint
3. **Progress monitoring:** Log metrics to external storage (Weights & Biases, TensorBoard)

### Data Management
1. **Google Drive mounting:** Persistent storage
2. **Dataset caching:** Load once, cache in session
3. **Incremental loading:** Stream data for M3 if needed

### Cost Control
1. **Budget alerts:** Monitor Colab compute units
2. **Training schedules:** Run overnight to avoid interruptions
3. **Early stopping:** Prevent unnecessary epochs

---

## Team Resource Allocation

### Parallel Development Strategy

**Track 1: VitalDB Model Training (Aswanth Kumar)**
- Hardware: L4 (M1) → A100 (M2-M3)
- Focus: Model optimization, hyperparameter tuning
- Timeline: Dec 21 - Jan 31 (6 weeks)

**Track 2: Multi-Sensor Development (Jayanth Tatineni)**
- Hardware: T4 (sufficient for prototyping)
- Focus: Sensor fusion, data pipeline for ECG/EDA/temp
- Timeline: Dec 11 - Jan 15 (5 weeks)
- **Benefit:** Free up L4/A100 for critical VitalDB training

**Resource Sharing:**
- Week 1-2: Both on T4/L4 (experimentation)
- Week 3-4: Aswanth on A100 (M2 training), Jayanth on T4
- Week 5-6: Aswanth on A100 (M3 training), Jayanth on L4

---

## Final Recommendation Summary

### 🏆 Optimal Infrastructure Plan

| Phase | Duration | Hardware | Cost | Justification |
|-------|----------|----------|------|---------------|
| **M1 Training** | Dec 21-23 | **L4 GPU** | $15 | Cost-effective, sufficient power |
| **M2 Experimentation** | Dec 24-28 | **L4 GPU** | $50 | Fast iteration for hyperparameter tuning |
| **M2 Final Run** | Dec 29-30 | **A100 GPU** | $30 | Production model quality |
| **M3 Training** | Jan 10-17 | **A100 GPU** | $500 | Required for scale, fastest time-to-model |
| | | **TOTAL** | **$595** | **Optimal balance of speed and cost** |

### Key Benefits
- ✅ **Fast iteration:** L4 for experimentation, A100 for production
- ✅ **Cost-effective:** Use premium hardware only when necessary
- ✅ **Team scalability:** Parallel tracks enabled
- ✅ **Risk-managed:** Sufficient VRAM headroom, checkpoint strategy
- ✅ **Timeline achievable:** M1-M3 completion in 6 weeks

---

## Next Steps

### Week of Dec 11-17 (Phase 0)
- [ ] Set up Google Colab Pro+ account (for L4 access)
- [ ] Test L4 GPU availability and performance
- [ ] Configure training scripts for mixed precision
- [ ] Set up checkpoint/resume infrastructure
- [ ] Validate data loading pipeline on L4

### Week of Dec 21-23 (M1 Training)
- [ ] Run M1 training on L4 GPU
- [ ] Monitor training metrics and convergence
- [ ] Evaluate model performance
- [ ] Document training configuration for M2

---

**Infrastructure Lead:** Aswanth Kumar Karibindi
**Budget Owner:** Project Lead
**Review Date:** December 17, 2025 (before M1 training starts)
