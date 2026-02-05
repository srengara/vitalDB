# Module Organization and Structure

## Overview

The VitalDB PPG Glucose Prediction codebase has been reorganized into a clean, modular structure with clear separation of concerns.

---

## New Directory Structure

```
vitalDB/
│
├── src/                              # All source code
│   ├── __init__.py                   # Package initialization
│   │
│   ├── data_extraction/              # Data extraction and preprocessing
│   │   ├── __init__.py
│   │   ├── ppg_extractor.py          # Extract PPG from VitalDB
│   │   ├── ppg_segmentation.py       # Signal preprocessing
│   │   ├── peak_detection.py         # Peak detection with template filtering
│   │   ├── glucose_extractor.py      # Extract glucose from VitalDB
│   │   ├── ppg_plotter.py            # Plotting utilities
│   │   └── ppg_visualizer.py         # Visualization tools
│   │
│   ├── training/                     # Model training
│   │   ├── __init__.py
│   │   ├── resnet34_glucose_predictor.py  # Model architecture (7.2M params)
│   │   └── train_glucose_predictor.py     # Training script with validation
│   │
│   ├── inference/                    # Prediction on new data
│   │   ├── __init__.py
│   │   └── glucose_from_csv.py       # Predict from CSV files
│   │
│   ├── web_app/                      # Web interface
│   │   ├── __init__.py
│   │   ├── web_app.py                # Flask application (5-step pipeline)
│   │   └── templates/                # HTML templates
│   │       └── index.html
│   │
│   └── utils/                        # Utilities and helpers
│       ├── __init__.py
│       ├── vitaldb_utility.py        # VitalDB API wrapper
│       ├── ppg_analysis_pipeline.py  # Batch processing
│       └── ppg_peak_detection_pipeline.py
│
├── docs/                             # Documentation (30 files)
│   ├── INDEX.md                      # Documentation index
│   ├── PROJECT_SUMMARY.md            # Complete project overview
│   ├── MODULE_ORGANIZATION.md        # This file
│   ├── *.md                          # 21 markdown documents
│   ├── *.pdf                         # 4 PDF documents (51 pages)
│   └── generate_*.py                 # PDF generation scripts
│
├── examples/                         # Example scripts
│   ├── example_usage.py              # Basic usage examples
│   ├── example_download.py           # Download VitalDB case
│   ├── example_glucose_prediction.py # Glucose prediction demo
│   ├── test_glucose_extraction.py    # Test glucose extraction
│   ├── find_valid_ppg_cases.py       # Find cases with PPG
│   ├── find_valid_snuadc_cases.py    # Find SNUADC cases
│   ├── list_case_ids.py              # List available cases
│   └── download_case.py              # Download specific case
│
├── training_data/                    # Training datasets (user-created)
│   ├── ppg_windows.csv               # PPG signal windows
│   └── glucose_labels.csv            # Glucose measurements
│
├── training_outputs/                 # Training results (user-created)
│   └── training_YYYYMMDD_HHMMSS/
│       ├── best_model.pth            # Best model checkpoint
│       ├── training_history.csv      # Loss/metrics per epoch
│       ├── final_metrics.json        # Test set results
│       └── tensorboard/              # TensorBoard logs
│
├── web_app_data/                     # Web app session data (user-created)
│   └── case_X_TRACK/
│       ├── raw.csv
│       ├── cleansed.csv
│       ├── peaks.csv
│       ├── filtered_windows.csv
│       ├── filtered_windows_detailed.csv
│       └── glucose_labels.csv
│
# Entry Point Scripts (in root)
├── run_web_app.py                    # Start web application
├── train_model.py                    # Train glucose prediction model
├── predict_glucose.py                # Run inference on CSV files
│
# Utilities
├── update_imports.py                 # Import update script
├── generate_html.py                  # Generate HTML from markdown
├── README.md                         # Main README
└── requirements.txt                  # Python dependencies (if created)
```

---

## Module Descriptions

### 1. `src/data_extraction/`

**Purpose**: Extract and preprocess data from VitalDB

**Key Files**:
- `ppg_extractor.py` - Extract PPG signals (SNUADC/PLETH, Solar8000/PLETH, etc.)
- `glucose_extractor.py` - Extract glucose measurements (Lab/GLU, ISTAT/GLU, etc.)
- `ppg_segmentation.py` - Bandpass filtering, signal preprocessing
- `peak_detection.py` - Peak detection with template-based quality filtering
- `ppg_plotter.py` - Plotting functions
- `ppg_visualizer.py` - Advanced visualization

**Typical Workflow**:
```python
from src.data_extraction import (
    PPGExtractor,
    GlucoseExtractor,
    PPGSegmenter,
    ppg_peak_detection_pipeline_with_template
)

# Extract PPG
ppg_extractor = PPGExtractor()
ppg_data = ppg_extractor.extract_ppg(case_id=2, track='SNUADC/PLETH')

# Preprocess
segmenter = PPGSegmenter(sampling_rate=500)
cleansed = segmenter.preprocess_signal(ppg_data['signal'])

# Detect peaks and filter
peaks, filtered_windows, template, all_windows = \
    ppg_peak_detection_pipeline_with_template(cleansed, fs=500)

# Extract glucose
glucose_extractor = GlucoseExtractor()
glucose_df = glucose_extractor.extract_glucose_data(case_id=2)
```

---

### 2. `src/training/`

**Purpose**: Train ResNet34-1D model for glucose prediction

**Key Files**:
- `resnet34_glucose_predictor.py` - Model architecture
  - `ResidualBlock1D` - Basic building block
  - `ResNet34_1D` - 34-layer network
  - `GlucosePredictor` - High-level interface
- `train_glucose_predictor.py` - Complete training script

**Usage**:
```bash
# Via entry point
python train_model.py \
  --data_dir ./training_data \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001

# Direct import
from src.training import ResNet34_1D, GlucosePredictor

model = ResNet34_1D(input_channels=1, num_classes=1)
predictor = GlucosePredictor(input_length=500)
```

**Features**:
- PyTorch-based training
- TensorBoard logging
- Model checkpointing
- Early stopping
- Validation metrics (MAE, RMSE)
- GPU/CPU support

---

### 3. `src/inference/`

**Purpose**: Run predictions on new PPG data

**Key Files**:
- `glucose_from_csv.py` - Predict glucose from CSV files

**Usage**:
```bash
# Via entry point
python predict_glucose.py filtered_windows.csv

# Outputs: glucose_predictions.csv
```

**Input Format** (CSV):
```csv
window_index,sample_index,amplitude
0,0,100.5
0,1,101.2
...
```

**Output Format** (CSV):
```csv
window_index,glucose_mg_dl
0,120.5
1,115.3
...
```

---

### 4. `src/web_app/`

**Purpose**: Interactive web interface for data processing pipeline

**Key Files**:
- `web_app.py` - Flask application with 5-step pipeline
- `templates/index.html` - Frontend UI

**5-Step Pipeline**:
1. **Select Case & Track** - Choose VitalDB case and PPG track
2. **View Raw Data** - Visualize original signal
3. **View Cleansed Data** - See filtered signal (bandpass 0.5-8 Hz)
4. **Peak Detection** - Detect peaks and filter by quality (cosine similarity > 0.85)
5. **Glucose Labels** - Extract from VitalDB or enter manually

**Usage**:
```bash
python run_web_app.py
# Open http://localhost:5000
```

**Features**:
- Real-time visualization
- Interactive parameter adjustment
- CSV download at each step
- Session management
- Automatic glucose extraction

---

### 5. `src/utils/`

**Purpose**: Utility functions and helpers

**Key Files**:
- `vitaldb_utility.py` - VitalDB API wrapper
  - Download cases
  - List tracks
  - Get track data
- `ppg_analysis_pipeline.py` - Batch processing multiple cases
- `ppg_peak_detection_pipeline.py` - Peak detection utilities

**Usage**:
```python
from src.utils import VitalDBUtility

util = VitalDBUtility()
tracks = util.get_available_tracks(case_id=2)
data = util.load_track(case_id=2, track='SNUADC/PLETH')
```

---

## Import Changes

### Old Import Style (Before Reorganization)
```python
from ppg_extractor import PPGExtractor
from glucose_extractor import GlucoseExtractor
from resnet34_glucose_predictor import ResNet34_1D
```

### New Import Style (After Reorganization)
```python
from src.data_extraction import PPGExtractor, GlucoseExtractor
from src.training import ResNet34_1D, GlucosePredictor
```

### Convenient Top-Level Imports
```python
# Main package provides key classes
from src import (
    PPGExtractor,           # from src.data_extraction
    GlucoseExtractor,       # from src.data_extraction
    ResNet34_1D,           # from src.training
    GlucosePredictor,      # from src.training
    predict_glucose_from_csv  # from src.inference
)
```

---

## Entry Point Scripts

These scripts in the root directory provide easy access to key functionality:

### `run_web_app.py`
```python
# Starts Flask web application
python run_web_app.py
```

### `train_model.py`
```python
# Trains ResNet34-1D model
python train_model.py --data_dir ./training_data --epochs 100
```

### `predict_glucose.py`
```python
# Runs inference on CSV files
python predict_glucose.py filtered_windows.csv
```

---

## Example Scripts

The `examples/` folder contains standalone scripts for specific tasks:

| Script | Purpose | Usage |
|--------|---------|-------|
| `example_usage.py` | Basic PPG extraction | `python examples/example_usage.py` |
| `example_download.py` | Download VitalDB case | `python examples/example_download.py --case_id 2` |
| `example_glucose_prediction.py` | Glucose prediction demo | `python examples/example_glucose_prediction.py` |
| `test_glucose_extraction.py` | Test glucose extraction | `python examples/test_glucose_extraction.py --case_id 2` |
| `find_valid_ppg_cases.py` | Find cases with PPG | `python examples/find_valid_ppg_cases.py` |
| `find_valid_snuadc_cases.py` | Find SNUADC cases | `python examples/find_valid_snuadc_cases.py` |
| `list_case_ids.py` | List available cases | `python examples/list_case_ids.py` |
| `download_case.py` | Download specific case | `python examples/download_case.py --case_id 2` |

---

## Documentation Structure

The `docs/` folder contains all documentation (30 files, 6.7 MB):

### Key Documents
- **INDEX.md** - Documentation index (start here)
- **PROJECT_SUMMARY.md** - Complete project overview
- **MODULE_ORGANIZATION.md** - This file

### PDF Documentation (51 pages total)
- **RESNET34_PROCESSING_EXPLAINED.pdf** (16 pages) - How model works
- **TRAINING_HARDWARE_AND_COSTING.pdf** (17 pages) - Hardware & Azure
- **GLUCOSE_PREDICTION_ARCHITECTURE.pdf** (18 pages) - Full API reference

### Markdown Documentation (21 files)
- Quick start guides
- Architecture documentation
- Feature documentation
- Troubleshooting guides

---

## Benefits of New Structure

### 1. Clear Separation of Concerns
- **Data extraction** → `src/data_extraction/`
- **Model training** → `src/training/`
- **Inference** → `src/inference/`
- **Web interface** → `src/web_app/`
- **Utilities** → `src/utils/`

### 2. Easy to Navigate
- Everything in `src/` is code
- Everything in `docs/` is documentation
- Everything in `examples/` is examples
- Entry points in root are obvious

### 3. Modular and Reusable
- Import only what you need
- Each module is self-contained
- Clear dependencies

### 4. Professional Structure
- Standard Python package layout
- `__init__.py` files for clean imports
- Follows best practices

### 5. Scalable
- Easy to add new modules
- Easy to add new features
- Easy to extend

---

## Migration Guide

### For Existing Code

If you have existing code using the old structure:

**Option 1: Update imports**
```python
# Old
from ppg_extractor import PPGExtractor

# New
from src.data_extraction import PPGExtractor
```

**Option 2: Use import update script**
```bash
python update_imports.py
```

### For New Code

Start with entry point scripts:
```bash
# Web application
python run_web_app.py

# Training
python train_model.py --data_dir ./training_data

# Inference
python predict_glucose.py data.csv
```

Or import modules directly:
```python
from src.data_extraction import PPGExtractor
from src.training import GlucosePredictor
```

---

## Development Workflow

### Adding a New Feature

1. **Identify module**: Which module does this belong to?
   - Data processing → `src/data_extraction/`
   - Model changes → `src/training/`
   - Prediction → `src/inference/`
   - Web UI → `src/web_app/`

2. **Add code**: Create or modify files in appropriate module

3. **Update `__init__.py`**: Export new classes/functions
   ```python
   # src/data_extraction/__init__.py
   from .my_new_module import MyNewClass
   __all__ = [..., 'MyNewClass']
   ```

4. **Add tests**: Create test in `examples/` if needed

5. **Update docs**: Add to `docs/`

6. **Create entry point**: If needed, add script to root

---

## Testing the New Structure

### Test 1: Web Application
```bash
python run_web_app.py
# Should start without errors
# Open http://localhost:5000
```

### Test 2: Data Extraction
```python
from src.data_extraction import PPGExtractor
extractor = PPGExtractor()
data = extractor.extract_ppg(2, 'SNUADC/PLETH')
print(f"Extracted {len(data['signal'])} samples")
```

### Test 3: Model Import
```python
from src.training import ResNet34_1D
model = ResNet34_1D(input_channels=1, num_classes=1)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Test 4: Inference
```bash
python predict_glucose.py examples/sample_data.csv
# Should run without errors
```

---

## Summary

The codebase has been reorganized into a clean, modular structure:

✅ **5 main modules**: data_extraction, training, inference, web_app, utils
✅ **30 documentation files**: Complete documentation in `docs/`
✅ **8 example scripts**: Working examples in `examples/`
✅ **3 entry points**: Easy-to-use scripts in root
✅ **Import updates**: All imports updated automatically
✅ **Professional structure**: Follows Python best practices

**Result**: Clean, maintainable, scalable codebase with clear separation of concerns.

---

**Version**: 1.0.0
**Last Updated**: November 2024
**Status**: Migration complete, all imports updated
