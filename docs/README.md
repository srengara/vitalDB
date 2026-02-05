# VitalDB PPG Glucose Prediction

Complete end-to-end pipeline for extracting PPG signals from VitalDB, processing them, and predicting blood glucose levels using deep learning (ResNet34-1D).

## Quick Start

### 1. Run Web Application
```bash
python run_web_app.py
# Open http://localhost:5000
```

### 2. Train Model
```bash
python train_model.py --data_dir ./training_data --epochs 100
```

### 3. Predict Glucose
```bash
python predict_glucose.py filtered_windows.csv
```

---

## Project Structure

```
vitalDB/
├── src/                          # Source code
│   ├── data_extraction/         # Extract & process PPG/glucose from VitalDB
│   │   ├── ppg_extractor.py    # PPG signal extraction
│   │   ├── ppg_segmentation.py # Signal preprocessing
│   │   ├── peak_detection.py   # Peak detection with template filtering
│   │   ├── glucose_extractor.py # Glucose data extraction
│   │   ├── ppg_plotter.py      # Plotting utilities
│   │   └── ppg_visualizer.py   # Visualization tools
│   │
│   ├── training/                # Model training
│   │   ├── resnet34_glucose_predictor.py  # ResNet34-1D architecture
│   │   └── train_glucose_predictor.py     # Training script
│   │
│   ├── inference/               # Prediction on new data
│   │   └── glucose_from_csv.py # Predict from CSV files
│   │
│   ├── web_app/                 # Web interface
│   │   ├── web_app.py          # Flask application
│   │   └── templates/          # HTML templates
│   │
│   └── utils/                   # Utilities
│       ├── vitaldb_utility.py  # VitalDB API wrapper
│       ├── ppg_analysis_pipeline.py  # Batch processing
│       └── ppg_peak_detection_pipeline.py
│
├── docs/                         # Documentation (21 MD + 4 PDF files)
│   ├── INDEX.md                 # Documentation index
│   ├── PROJECT_SUMMARY.md       # Complete project overview
│   └── *.pdf                    # PDF documentation (51 pages)
│
├── examples/                     # Example scripts
│   ├── example_usage.py         # Basic usage
│   ├── example_download.py      # Download case
│   ├── test_glucose_extraction.py  # Test glucose extraction
│   └── find_valid_ppg_cases.py # Find cases with PPG
│
# Entry Points
├── run_web_app.py               # Start web application
├── train_model.py               # Train glucose prediction model
├── predict_glucose.py           # Run inference on CSV files
│
# Data (user-created)
├── training_data/               # Training datasets
├── training_outputs/            # Training results
└── web_app_data/                # Web app session data
```

---

## Modules

### 1. Data Extraction (`src/data_extraction/`)

Extract and process PPG and glucose data from VitalDB.

**Key Classes**:
- `PPGExtractor` - Extract PPG signals from VitalDB
- `GlucoseExtractor` - Extract glucose measurements
- `PPGSegmenter` - Preprocess signals (bandpass filtering)

**Key Functions**:
- `ppg_peak_detection_pipeline_with_template()` - Detect peaks with quality filtering

**Usage**:
```python
from src.data_extraction import PPGExtractor, GlucoseExtractor

# Extract PPG
extractor = PPGExtractor()
ppg_data = extractor.extract_ppg(case_id=2, track='SNUADC/PLETH')

# Extract glucose
glucose_extractor = GlucoseExtractor()
glucose_df = glucose_extractor.extract_glucose_data(case_id=2)
```

---

### 2. Training (`src/training/`)

Train ResNet34-1D model for glucose prediction.

**Key Classes**:
- `ResNet34_1D` - 34-layer residual network (7.2M parameters)
- `GlucosePredictor` - High-level prediction interface
- `ResidualBlock1D` - Basic building block

**Training Script**:
```bash
python train_model.py \
  --data_dir ./training_data \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001
```

**Training Features**:
- TensorBoard logging
- Model checkpointing
- Early stopping
- Validation metrics (MAE, RMSE)

---

### 3. Inference (`src/inference/`)

Run predictions on new PPG data.

**Usage**:
```bash
python predict_glucose.py filtered_windows.csv
```

**Output**: `glucose_predictions.csv` with predicted glucose values

---

### 4. Web Application (`src/web_app/`)

Interactive 5-step pipeline:
1. Select Case & Track
2. View Raw Data
3. View Cleansed Data
4. Peak Detection & Filtering
5. Glucose Labels (extract or enter)

**Run**:
```bash
python run_web_app.py
# Open http://localhost:5000
```

---

### 5. Utilities (`src/utils/`)

**VitalDBUtility** - Interface to VitalDB API
- Download cases
- List available tracks
- Get track data

**Batch Processing** - `ppg_analysis_pipeline.py`

---

## Installation

### Requirements
```bash
pip install torch numpy pandas scipy scikit-learn matplotlib flask tensorboard
```

### Optional (for GPU training)
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage Examples

### Example 1: Web Application Workflow

```bash
# Start web app
python run_web_app.py

# In browser (http://localhost:5000):
# 1. Enter case ID: 2
# 2. Select track: SNUADC/PLETH
# 3. Download data → Cleanse → Detect peaks → Filter windows
# 4. Extract glucose from VitalDB
# 5. Download glucose_labels.csv
```

### Example 2: Programmatic Data Extraction

```python
from src.data_extraction import PPGExtractor, GlucoseExtractor

# Extract PPG
ppg_extractor = PPGExtractor()
ppg_data = ppg_extractor.extract_ppg(case_id=2, track='SNUADC/PLETH')

# Extract glucose
glucose_extractor = GlucoseExtractor()
glucose_df = glucose_extractor.extract_glucose_data(case_id=2)

# Match glucose to PPG windows
glucose_values = glucose_extractor.match_glucose_to_ppg_windows(
    glucose_df,
    ppg_window_times,
    method='interpolate'
)
```

### Example 3: Train Model

```bash
# Prepare data (from web app or programmatically)
# - training_data/ppg_windows.csv
# - training_data/glucose_labels.csv

# Train model
python train_model.py \
  --data_dir ./training_data \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --early_stopping 20

# Monitor training
tensorboard --logdir=./training_outputs
```

### Example 4: Run Inference

```bash
# Predict glucose from new PPG windows
python predict_glucose.py \
  web_app_data/case_2_SNUADC_PLETH/filtered_windows_detailed.csv

# Output: glucose_predictions.csv
```

---

## Model Architecture

**ResNet34-1D**:
- **Input**: PPG window (500 samples, 1 second @ 500 Hz)
- **Architecture**: 34 convolutional layers with residual connections
  - Layer 1: 3 blocks (64 channels)
  - Layer 2: 4 blocks (128 channels)
  - Layer 3: 6 blocks (256 channels)
  - Layer 4: 3 blocks (512 channels)
- **Output**: Glucose value (mg/dL)
- **Parameters**: 7,218,753

**Key Innovation**: Residual connections enable training very deep networks without gradient vanishing.

---

## Documentation

📚 **Complete Documentation**: See [docs/INDEX.md](docs/INDEX.md)

**Key Documents**:
- [PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) - Complete project overview
- [RESNET34_PROCESSING_EXPLAINED.pdf](docs/RESNET34_PROCESSING_EXPLAINED.pdf) - How model works (16 pages)
- [TRAINING_HARDWARE_AND_COSTING.pdf](docs/TRAINING_HARDWARE_AND_COSTING.pdf) - Hardware & Azure setup (17 pages)
- [GLUCOSE_PREDICTION_ARCHITECTURE.pdf](docs/GLUCOSE_PREDICTION_ARCHITECTURE.pdf) - Full API reference (18 pages)

---

## Performance

### Signal Processing
- **Processing time**: ~5 seconds for 30-minute recording
- **Peak detection**: 500-5000 peaks per case
- **Quality filtering**: 85-95% pass rate

### Model (After Training)
- **Target MAE**: < 15 mg/dL
- **Target RMSE**: < 20 mg/dL
- **Inference time**: < 50ms per window (CPU)
- **GPU training**: 2-10 hours (depends on dataset size)

---

## Hardware Requirements

### Local Training
- **CPU**: 8+ cores
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3070 (8GB) or better
- **Storage**: 100 GB SSD

### Cloud Training (Azure)
- **Budget**: NC6s v3 (1x V100, 16GB) - $3.06/hour
- **Recommended**: NC24ads A100 v4 (1x A100, 80GB) - $4.89/hour
- **Training cost**: $50-$200 for typical dataset

See [TRAINING_HARDWARE_AND_COSTING.pdf](docs/TRAINING_HARDWARE_AND_COSTING.pdf) for details.

---

## Data Requirements

### For Training
- **Minimum**: 10,000 PPG-glucose pairs
- **Recommended**: 50,000+ pairs
- **Format**:
  - `ppg_windows.csv`: window_index, sample_index, amplitude
  - `glucose_labels.csv`: window_index, glucose_mg_dl

### VitalDB Access
- 6,388 surgical cases with physiological signals
- Open-source, de-identified data
- Access via Python API

---

## Common Tasks

### Task 1: Process a New VitalDB Case

```bash
# Option A: Use web app
python run_web_app.py
# Follow 5-step pipeline

# Option B: Use Python script
python examples/example_usage.py --case_id 100
```

### Task 2: Find Cases with PPG Data

```bash
python examples/find_valid_ppg_cases.py
```

### Task 3: Extract Glucose for Training

```bash
python examples/test_glucose_extraction.py --case_id 2
```

### Task 4: Train on Multiple Cases

```bash
# 1. Process multiple cases via web app
# 2. Combine glucose_labels.csv files
# 3. Train model
python train_model.py --data_dir ./training_data
```

---

## Troubleshooting

### Issue: Import errors after reorganization

**Solution**: Imports are now relative to `src/`. Update any custom scripts:
```python
# Old
from ppg_extractor import PPGExtractor

# New
from src.data_extraction.ppg_extractor import PPGExtractor
```

### Issue: Web app not starting

**Solution**: Run from root directory:
```bash
cd /c/IITM/vitalDB
python run_web_app.py
```

### Issue: No glucose data for case

**Solution**:
1. Try different case IDs
2. Use manual glucose entry
3. Check [docs/PPG_TROUBLESHOOTING.md](docs/PPG_TROUBLESHOOTING.md)

---

## Contributing

### Adding New Features

1. Place code in appropriate module:
   - Data extraction → `src/data_extraction/`
   - Model changes → `src/training/`
   - Inference → `src/inference/`
   - Web UI → `src/web_app/`

2. Update `__init__.py` to export classes/functions

3. Add documentation to `docs/`

4. Add examples to `examples/`

---

## License

- **VitalDB Data**: Open-source, research use, IRB approved
- **Code**: Research and educational use
- **Clinical Use**: Requires FDA approval and extensive validation

---

## Citation

If you use this project in research, please cite:

```
VitalDB PPG Glucose Prediction Pipeline
https://vitaldb.net/
```

---

## Contact

- **Documentation**: See [docs/INDEX.md](docs/INDEX.md)
- **Issues**: Check [docs/PPG_TROUBLESHOOTING.md](docs/PPG_TROUBLESHOOTING.md)
- **VitalDB**: https://vitaldb.net/

---

## Version

**Version**: 1.0.0
**Last Updated**: November 2024
**Status**: Production-ready for research and development

---

## Summary

This project provides a **complete, modular pipeline** for glucose prediction from PPG signals:

✅ **Data Extraction** - Extract PPG and glucose from VitalDB
✅ **Signal Processing** - Cleanse, detect peaks, filter quality
✅ **Deep Learning** - ResNet34-1D with 7.2M parameters
✅ **Training** - Complete training script with validation
✅ **Inference** - Predict on new data
✅ **Web Interface** - User-friendly 5-step pipeline
✅ **Documentation** - 51 pages of professional docs

**Get Started**:
```bash
python run_web_app.py
```
