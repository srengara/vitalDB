# Training Hardware Recommendations and Azure Costing

## Overview

This document provides detailed hardware recommendations for training the ResNet34-1D glucose prediction model and estimated costs for running training on Microsoft Azure.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Local Hardware Options](#local-hardware-options)
3. [Azure Virtual Machine Options](#azure-virtual-machine-options)
4. [Cost Estimates](#cost-estimates)
5. [Training Time Estimates](#training-time-estimates)
6. [Cost Optimization Strategies](#cost-optimization-strategies)
7. [Recommended Setup](#recommended-setup)

---

## Hardware Requirements

### Minimum Requirements

**For Small Datasets (< 10,000 samples)**:
- **CPU**: 4+ cores (Intel i5/i7 or AMD Ryzen 5/7)
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with 6+ GB VRAM (e.g., GTX 1660, RTX 3050)
- **Storage**: 50 GB SSD
- **Training time**: 2-4 hours for 100 epochs

### Recommended Requirements

**For Medium Datasets (10,000 - 50,000 samples)**:
- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3070, RTX 4060 Ti)
- **Storage**: 100 GB SSD
- **Training time**: 4-8 hours for 100 epochs

### Optimal Requirements

**For Large Datasets (50,000+ samples)**:
- **CPU**: 16+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 64 GB
- **GPU**: NVIDIA GPU with 12+ GB VRAM (e.g., RTX 3090, RTX 4090, A5000)
- **Storage**: 200 GB NVMe SSD
- **Training time**: 8-16 hours for 100 epochs

### Enterprise/Production Training

**For Very Large Datasets (100,000+ samples) or Multi-GPU**:
- **CPU**: 32+ cores (Dual Xeon or AMD EPYC)
- **RAM**: 128+ GB
- **GPU**: Multiple NVIDIA A100 (40GB or 80GB) or H100
- **Storage**: 500 GB+ NVMe SSD in RAID
- **Training time**: 16-48 hours for 100 epochs

---

## Local Hardware Options

### Option 1: Entry-Level GPU Workstation

**Configuration**:
- CPU: Intel Core i5-13600K (14 cores)
- RAM: 32 GB DDR5
- GPU: NVIDIA RTX 4060 Ti (8GB VRAM)
- Storage: 500 GB NVMe SSD
- PSU: 650W 80+ Gold

**Estimated Cost**: $1,500 - $2,000

**Pros**:
- One-time purchase
- No recurring costs
- Good for small to medium datasets
- Can be used for other tasks

**Cons**:
- Limited scalability
- Power consumption costs
- Requires maintenance
- Depreciation over time

### Option 2: High-End GPU Workstation

**Configuration**:
- CPU: AMD Ryzen 9 7950X (16 cores)
- RAM: 64 GB DDR5
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Storage: 1 TB NVMe SSD
- PSU: 1000W 80+ Platinum

**Estimated Cost**: $3,500 - $4,500

**Pros**:
- Excellent performance
- Can handle large datasets
- Future-proof for 3-5 years
- No cloud costs

**Cons**:
- High upfront cost
- Power consumption (~500W under load)
- Limited to single GPU
- Space and cooling requirements

### Option 3: Pre-built Deep Learning Workstation

**Configuration**:
- Lambda Labs Tensorbook or similar
- CPU: Intel Xeon or AMD Threadripper
- RAM: 128 GB ECC
- GPU: 2x NVIDIA RTX 4090 or 1x A6000 (48GB)
- Storage: 2 TB NVMe SSD

**Estimated Cost**: $8,000 - $12,000

**Pros**:
- Pre-configured for deep learning
- Multi-GPU support
- Professional-grade components
- Excellent performance

**Cons**:
- Very high upfront cost
- Overkill for single model training
- Expensive to upgrade

---

## Azure Virtual Machine Options

Microsoft Azure offers GPU-enabled VMs through the NCv3, NC A100 v4, and ND series.

### Option 1: Azure NC6s v3 (Entry-Level GPU)

**Specifications**:
- **vCPUs**: 6
- **RAM**: 112 GB
- **GPU**: 1x NVIDIA Tesla V100 (16 GB VRAM)
- **Storage**: 736 GB SSD (temporary)
- **Network**: Up to 24 Gbps

**Use Case**: Small to medium datasets (< 20,000 samples)

**Azure Region Availability**:
- East US
- West US 2
- North Europe
- Southeast Asia

### Option 2: Azure NC12s v3 (Mid-Range GPU)

**Specifications**:
- **vCPUs**: 12
- **RAM**: 224 GB
- **GPU**: 2x NVIDIA Tesla V100 (16 GB VRAM each, 32 GB total)
- **Storage**: 1,474 GB SSD (temporary)
- **Network**: Up to 24 Gbps

**Use Case**: Medium to large datasets (20,000 - 80,000 samples)

**Azure Region Availability**: Same as NC6s v3

### Option 3: Azure NC24s v3 (High-End GPU)

**Specifications**:
- **vCPUs**: 24
- **RAM**: 448 GB
- **GPU**: 4x NVIDIA Tesla V100 (16 GB VRAM each, 64 GB total)
- **Storage**: 2,948 GB SSD (temporary)
- **Network**: Up to 24 Gbps

**Use Case**: Large datasets (80,000+ samples) or multi-model training

**Azure Region Availability**: Same as NC6s v3

### Option 4: Azure NC A100 v4 (Latest Generation)

**Specifications**:
- **vCPUs**: 24
- **RAM**: 220 GB
- **GPU**: 1x NVIDIA A100 (40 GB or 80 GB VRAM)
- **Storage**: 1,123 GB SSD (temporary)
- **Network**: Up to 24 Gbps
- **NVIDIA NVLink**: Yes

**Use Case**: Large-scale training, research, production deployments

**Azure Region Availability**:
- East US
- West US 2
- West Europe

### Option 5: Azure ND96asr v4 (Enterprise GPU)

**Specifications**:
- **vCPUs**: 96
- **RAM**: 900 GB
- **GPU**: 8x NVIDIA A100 (40 GB VRAM each, 320 GB total)
- **Storage**: 6,000 GB SSD (temporary)
- **Network**: Up to 200 Gbps
- **InfiniBand**: 200 Gb/s

**Use Case**: Massive datasets, distributed training, production ML pipelines

**Azure Region Availability**: Limited (East US, West US 2)

---

## Cost Estimates

### Azure Pricing (as of November 2024)

Prices are based on **Pay-As-You-Go** rates in **East US** region.

#### NC6s v3 (1x V100, 16GB)
- **Hourly Rate**: $3.06/hour
- **Daily Rate**: $73.44/day (24 hours)
- **Weekly Rate**: $514.08/week
- **Monthly Rate**: ~$2,203/month (720 hours)

**Training Cost Estimates**:
- 4 hours training: $12.24
- 8 hours training: $24.48
- 16 hours training: $48.96
- 100 hours training: $306.00

#### NC12s v3 (2x V100, 32GB)
- **Hourly Rate**: $6.12/hour
- **Daily Rate**: $146.88/day
- **Weekly Rate**: $1,028.16/week
- **Monthly Rate**: ~$4,406/month

**Training Cost Estimates**:
- 4 hours training: $24.48
- 8 hours training: $48.96
- 16 hours training: $97.92
- 100 hours training: $612.00

#### NC24s v3 (4x V100, 64GB)
- **Hourly Rate**: $12.24/hour
- **Daily Rate**: $293.76/day
- **Weekly Rate**: $2,056.32/week
- **Monthly Rate**: ~$8,813/month

**Training Cost Estimates**:
- 4 hours training: $48.96
- 8 hours training: $97.92
- 16 hours training: $195.84
- 100 hours training: $1,224.00

#### NC24ads A100 v4 (1x A100, 80GB)
- **Hourly Rate**: $3.67/hour (with 1-year reserved instance)
- **Hourly Rate**: $4.89/hour (pay-as-you-go)
- **Daily Rate**: $117.36/day (pay-as-you-go)
- **Monthly Rate**: ~$3,521/month (pay-as-you-go)

**Training Cost Estimates** (pay-as-you-go):
- 4 hours training: $19.56
- 8 hours training: $39.12
- 16 hours training: $78.24
- 100 hours training: $489.00

#### ND96asr v4 (8x A100, 320GB)
- **Hourly Rate**: $27.20/hour
- **Daily Rate**: $652.80/day
- **Weekly Rate**: $4,569.60/week
- **Monthly Rate**: ~$19,584/month

**Training Cost Estimates**:
- 4 hours training: $108.80
- 8 hours training: $217.60
- 16 hours training: $435.20
- 100 hours training: $2,720.00

### Storage Costs

**Azure Managed Disks**:
- **Premium SSD (P10 - 128 GB)**: $19.71/month
- **Premium SSD (P20 - 512 GB)**: $73.22/month
- **Premium SSD (P30 - 1 TB)**: $135.17/month

**Azure Blob Storage** (for datasets):
- **Hot tier**: $0.0184/GB/month
- **Cool tier**: $0.0100/GB/month
- **For 100 GB dataset**: ~$1.84/month (hot) or $1.00/month (cool)

### Data Transfer Costs

**Outbound Data Transfer** (from Azure to internet):
- First 100 GB/month: Free
- 100 GB - 10 TB/month: $0.087/GB
- Over 10 TB/month: $0.083/GB

**Inbound Data Transfer**: Free

**Example**: Downloading 10 GB of trained models = Free (under 100 GB)

---

## Training Time Estimates

### ResNet34-1D Model (7.2M parameters)

Training time depends on:
1. Dataset size
2. Batch size
3. Number of epochs
4. GPU performance

#### Small Dataset (10,000 samples, 100 epochs)

| GPU | Batch Size | Time per Epoch | Total Time (100 epochs) |
|-----|------------|----------------|-------------------------|
| RTX 3070 (8GB) | 32 | 1.5 min | 2.5 hours |
| RTX 4090 (24GB) | 64 | 0.8 min | 1.3 hours |
| V100 (16GB) | 32 | 1.2 min | 2.0 hours |
| A100 (40GB) | 64 | 0.6 min | 1.0 hour |

#### Medium Dataset (50,000 samples, 100 epochs)

| GPU | Batch Size | Time per Epoch | Total Time (100 epochs) |
|-----|------------|----------------|-------------------------|
| RTX 3070 (8GB) | 32 | 7.5 min | 12.5 hours |
| RTX 4090 (24GB) | 64 | 4.0 min | 6.7 hours |
| V100 (16GB) | 32 | 6.0 min | 10.0 hours |
| A100 (40GB) | 64 | 3.0 min | 5.0 hours |

#### Large Dataset (100,000 samples, 100 epochs)

| GPU | Batch Size | Time per Epoch | Total Time (100 epochs) |
|-----|------------|----------------|-------------------------|
| RTX 3070 (8GB) | 32 | 15 min | 25 hours |
| RTX 4090 (24GB) | 64 | 8 min | 13.3 hours |
| V100 (16GB) | 32 | 12 min | 20 hours |
| A100 (40GB) | 64 | 6 min | 10 hours |

**Note**: Times are approximate and assume:
- Window length: 500 samples
- No data augmentation
- Standard preprocessing
- Single GPU training

---

## Cost Optimization Strategies

### 1. Use Reserved Instances

**Azure Reserved VM Instances** (1-year or 3-year commitment):
- **1-year reservation**: Save up to 40%
- **3-year reservation**: Save up to 60%

**Example**: NC6s v3
- Pay-as-you-go: $3.06/hour
- 1-year reserved: $2.14/hour (30% savings)
- 3-year reserved: $1.53/hour (50% savings)

**Best for**: Regular training workloads, ongoing research

### 2. Use Spot Instances

**Azure Spot VMs**: Up to 90% discount on unused capacity
- Available when Azure has excess capacity
- Can be evicted with 30-second notice
- Must implement checkpointing to resume training

**Example**: NC6s v3
- Pay-as-you-go: $3.06/hour
- Spot price: $0.31 - $0.92/hour (70-90% savings)

**Best for**: Non-urgent training, experimentation, when checkpointing is implemented

**Risk**: Job may be interrupted if capacity is needed

### 3. Schedule Training During Off-Peak Hours

Some regions have lower demand during certain hours:
- Run training overnight or weekends
- Use auto-shutdown to avoid idle costs

### 4. Optimize Batch Size

Larger batch sizes = fewer iterations = faster training:
- RTX 3070 (8GB): Batch size 32-48
- V100 (16GB): Batch size 64-96
- A100 (40GB): Batch size 128-256

**Trade-off**: Very large batches may affect convergence

### 5. Use Mixed Precision Training

**FP16 (half precision)** instead of FP32:
- 2x faster training
- 2x memory efficiency (can double batch size)
- Minimal accuracy impact

**PyTorch implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefit**: Can reduce training time by 30-50%

### 6. Use Early Stopping

Stop training when validation loss stops improving:
- Typical patience: 10-20 epochs
- Can save 20-50% of training time
- Already implemented in `train_glucose_predictor.py`

### 7. Transfer Learning / Pre-training

If you have unlabeled PPG data:
1. Pre-train on unlabeled data (self-supervised)
2. Fine-tune on labeled PPG-glucose pairs
3. Requires less labeled data
4. Faster convergence

### 8. Use Smaller Models for Prototyping

For initial experiments:
- Use ResNet18 (11M params) instead of ResNet34 (21M params)
- Train on subset of data first
- Scale up once hyperparameters are tuned

### 9. Data Pipeline Optimization

- Use `num_workers > 0` in DataLoader (parallel data loading)
- Use `pin_memory=True` for GPU training
- Preprocess data once and save (don't preprocess every epoch)

### 10. Multi-GPU Training (for large datasets)

If using multiple GPUs:
```python
model = nn.DataParallel(model)  # Simple multi-GPU
# or
model = DistributedDataParallel(model)  # Better performance
```

**Speedup**: Near-linear scaling (2 GPUs = ~1.8x faster)

---

## Recommended Setup

### For Researchers / Students (Budget: < $100)

**Option**: Azure NC6s v3 with Spot Instances

**Configuration**:
- VM: NC6s v3 (1x V100, 16GB)
- Pricing: Spot instance (~$0.50/hour)
- Storage: 128 GB Premium SSD (~$20/month)
- Training duration: 8-16 hours

**Total Cost**: $4-$8 for one training run

**Best for**:
- Small datasets (< 20,000 samples)
- Experimentation and prototyping
- Learning and coursework

### For Startups / Small Teams (Budget: $200-500/month)

**Option**: Azure NC6s v3 with Reserved Instance

**Configuration**:
- VM: NC6s v3 (1x V100, 16GB)
- Pricing: 1-year reserved (~$2.14/hour)
- Storage: 512 GB Premium SSD (~$73/month)
- Usage: 80 hours/month training

**Total Cost**: $244/month (80 hours × $2.14 + $73)

**Best for**:
- Regular training workloads
- Medium datasets (20,000 - 50,000 samples)
- Production model updates (monthly/quarterly)

### For Medium Enterprises (Budget: $1,000-3,000/month)

**Option**: Azure NC24ads A100 v4

**Configuration**:
- VM: 1x A100 (80GB VRAM)
- Pricing: 1-year reserved (~$3.67/hour)
- Storage: 1 TB Premium SSD (~$135/month)
- Usage: 200 hours/month training

**Total Cost**: $869/month (200 hours × $3.67 + $135)

**Best for**:
- Large datasets (50,000 - 200,000 samples)
- Multiple models in production
- Frequent retraining (weekly/biweekly)

### For Large Enterprises (Budget: $5,000+/month)

**Option**: Azure ND96asr v4 (Multi-GPU)

**Configuration**:
- VM: 8x A100 (320GB VRAM total)
- Pricing: Reserved instance (~$18/hour estimated)
- Storage: 2 TB Premium SSD + Blob storage
- Usage: 200 hours/month

**Total Cost**: $3,735/month (200 hours × $18 + storage)

**Best for**:
- Very large datasets (200,000+ samples)
- Multi-model training pipelines
- Real-time model updates
- Research and development

---

## Cost Comparison: Local vs Cloud

### Scenario: Medium Dataset (50,000 samples, 100 epochs)

#### Local RTX 4090 Workstation

**Upfront Cost**: $4,000

**Training Cost** (electricity):
- Power consumption: 500W under load
- Training time: 13 hours
- Energy cost: 13 hours × 0.5 kW × $0.12/kWh = $0.78

**Total Cost (1st year)**: $4,000 + ($0.78 × 12 trainings) = $4,009
**Total Cost (3 years)**: $4,000 + ($0.78 × 36 trainings) = $4,028

**Break-even**: If you train < 1,300 times in 3 years, local is cheaper

#### Azure NC6s v3 (V100)

**Training Cost**: 10 hours × $3.06/hour = $30.60 per training

**Total Cost (1st year)**: $30.60 × 12 = $367
**Total Cost (3 years)**: $30.60 × 36 = $1,102

**Break-even**: If you train > 130 times in 3 years, local is cheaper

### Recommendation

**Use Cloud (Azure) if**:
- Training < 10 times per month
- Need latest hardware (A100, H100)
- Want flexibility to scale up/down
- Don't want upfront capital expense
- Need to experiment with different GPUs

**Buy Local Hardware if**:
- Training > 20 times per month
- Long-term commitment (3+ years)
- Have budget for upfront purchase
- Need 24/7 access for development
- Privacy/security requirements

---

## Azure Setup Guide

### Step 1: Create Azure Account

1. Go to https://azure.microsoft.com/
2. Sign up for free account (includes $200 credit for 30 days)
3. Verify identity and add payment method

### Step 2: Request GPU Quota

By default, Azure limits GPU VMs. Request quota increase:

1. Navigate to: Azure Portal → Subscriptions → Usage + quotas
2. Search for "NCv3" or "NCA100v4"
3. Click on quota and request increase
   - NC6s v3: Request 6 vCPUs
   - NC24ads A100 v4: Request 24 vCPUs
4. Wait for approval (typically 1-3 business days)

### Step 3: Create Virtual Machine

**Via Azure Portal**:
1. Navigate to: Virtual machines → Create → Azure virtual machine
2. **Basics**:
   - Subscription: Pay-As-You-Go
   - Resource group: Create new → "glucose-training"
   - VM name: "glucose-trainer-01"
   - Region: East US (or nearest)
   - Image: Ubuntu 20.04 LTS - Gen2
   - Size: NC6s v3 (or desired GPU VM)
   - Authentication: SSH public key
3. **Disks**:
   - OS disk: Premium SSD (128 GB minimum)
   - Data disk: Add new → Premium SSD (512 GB)
4. **Networking**:
   - Create new virtual network
   - Public IP: Enable
   - SSH (22): Allow
   - HTTPS (443): Allow
5. **Management**:
   - Auto-shutdown: Enable (save costs)
   - Shutdown time: 11:00 PM local time
6. Review + Create

**Estimated Time**: 5-10 minutes to provision

### Step 4: Connect to VM

```bash
# SSH into VM
ssh azureuser@<VM_PUBLIC_IP>

# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot

# Verify GPU (after reboot)
nvidia-smi
```

### Step 5: Install Dependencies

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init

# Restart shell
source ~/.bashrc

# Create environment
conda create -n glucose python=3.10 -y
conda activate glucose

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy pandas scipy scikit-learn matplotlib tensorboard

# Verify PyTorch sees GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 6: Upload Data and Code

**Option A: Using SCP**:
```bash
# From local machine
scp -r ./vitalDB azureuser@<VM_IP>:~/
```

**Option B: Using Azure Storage**:
```bash
# Upload to Azure Blob Storage first, then download to VM
az storage blob upload-batch -d mycontainer -s ./vitalDB
```

### Step 7: Run Training

```bash
# SSH into VM
ssh azureuser@<VM_IP>

# Activate environment
conda activate glucose

# Navigate to code directory
cd ~/vitalDB

# Start training
python train_glucose_predictor.py \
  --data_dir ./training_data \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --output_dir ./training_outputs

# Monitor with TensorBoard (optional)
tensorboard --logdir=./training_outputs/training_*/tensorboard --host=0.0.0.0 --port=6006
```

### Step 8: Download Results

```bash
# From local machine
scp -r azureuser@<VM_IP>:~/vitalDB/training_outputs ./
```

### Step 9: Stop/Delete VM

**Stop VM** (preserves VM, only pay for storage):
```bash
az vm deallocate --resource-group glucose-training --name glucose-trainer-01
```

**Delete VM** (removes everything, no charges):
```bash
az vm delete --resource-group glucose-training --name glucose-trainer-01 --yes
az group delete --name glucose-training --yes
```

---

## Monitoring and Cost Control

### Azure Cost Management

1. Navigate to: Azure Portal → Cost Management + Billing
2. Set up **Budget Alerts**:
   - Budget: $500/month
   - Alert at: 80%, 100%, 120%
   - Email notifications

### Monitor VM Usage

**Azure Portal Dashboard**:
- CPU utilization
- GPU utilization
- Network in/out
- Disk IOPS

**Using Azure CLI**:
```bash
# List all VMs and their status
az vm list --show-details --output table

# Get specific VM details
az vm show --resource-group glucose-training --name glucose-trainer-01
```

### Auto-Shutdown

Configure auto-shutdown to prevent idle costs:
1. VM → Auto-shutdown
2. Enable: Yes
3. Shutdown time: 11:00 PM
4. Time zone: Your local time
5. Notification: Enable (15 min before shutdown)

---

## Final Recommendations

### Best Option for Most Users

**Azure NC6s v3 with 1-Year Reserved Instance**

**Why**:
- Affordable: ~$2.14/hour
- Sufficient for most datasets (< 100,000 samples)
- No upfront hardware cost
- Pay only for what you use
- Easy to scale up if needed

**Total Cost for Typical Project**:
- Initial experimentation: 20 hours × $2.14 = $42.80
- Full training (5 runs): 50 hours × $2.14 = $107.00
- Total: ~$150 for complete project

### Best for Production Systems

**Azure NC24ads A100 v4 with Reserved Instance**

**Why**:
- Latest hardware (A100 GPU)
- 80 GB VRAM (can handle any dataset)
- Faster training (2-3x vs V100)
- Better for production workloads

**Total Cost**:
- 1-year reserved: ~$3.67/hour
- 100 hours/month: $367/month
- Affordable for businesses

### Best for Research Labs

**Combination Strategy**:
1. **Prototyping**: Azure Spot instances (save 90%)
2. **Development**: NC6s v3 reserved instance
3. **Final training**: NC24ads A100 v4 (pay-as-you-go)

**Annual Cost**: ~$2,000 - $5,000 depending on usage

---

## Conclusion

For the ResNet34-1D glucose prediction model:

**Recommended Path**:
1. **Start**: Azure free trial ($200 credit)
2. **Prototype**: Use Spot instances for experimentation
3. **Production**: Switch to reserved instance once stable
4. **Scale**: Move to A100 VMs for large datasets

**Cost Projection** (first year):
- Prototyping: $50 - $100
- Development: $500 - $1,000
- Production: $1,000 - $3,000
- **Total**: $1,550 - $4,100

This is significantly cheaper than buying local hardware ($4,000+) unless you plan to train hundreds of times per year.

---

## Additional Resources

### Azure Documentation
- Azure ML documentation: https://docs.microsoft.com/azure/machine-learning/
- GPU VM pricing: https://azure.microsoft.com/pricing/details/virtual-machines/linux/
- Cost management: https://docs.microsoft.com/azure/cost-management-billing/

### Deep Learning Optimization
- PyTorch profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Mixed precision training: https://pytorch.org/docs/stable/amp.html
- Distributed training: https://pytorch.org/tutorials/beginner/dist_overview.html

### Alternative Cloud Providers

**Google Cloud Platform (GCP)**:
- Similar pricing to Azure
- TPU options available
- Good for TensorFlow workloads

**Amazon Web Services (AWS)**:
- EC2 P3/P4 instances (V100/A100)
- SageMaker for managed training
- Typically 10-15% more expensive than Azure

**Lambda Labs**:
- GPU cloud specifically for ML
- Cheaper than big cloud providers
- Limited regions

---

**Document Version**: 1.0
**Last Updated**: November 2024
**Author**: Training Infrastructure Team
