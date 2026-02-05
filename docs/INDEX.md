# VitalDB PPG Glucose Prediction - Documentation Index

This folder contains all documentation for the VitalDB PPG analysis and glucose prediction project.

---

## Quick Start Guides

### Getting Started
- **[README.md](README.md)** - Project overview and setup
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide for PPG extraction
- **[PPG_QUICK_START.md](PPG_QUICK_START.md)** - Quick start for PPG analysis
- **[QUICK_START_GLUCOSE.md](QUICK_START_GLUCOSE.md)** - Quick start for glucose prediction

### Web Application
- **[WEB_APP_README.md](WEB_APP_README.md)** - Web interface documentation
- **[START_WEB_APP.md](START_WEB_APP.md)** - How to start the web application

---

## PPG Analysis

### Core Documentation
- **[PPG_README.md](PPG_README.md)** - PPG signal processing documentation
- **[PPG_BATCH_USAGE.md](PPG_BATCH_USAGE.md)** - Batch processing multiple cases
- **[PPG_TROUBLESHOOTING.md](PPG_TROUBLESHOOTING.md)** - Common issues and solutions

### Peak Detection
- **[PEAK_DETECTION_EXTENDED.md](PEAK_DETECTION_EXTENDED.md)** - Peak detection with template filtering
- **[PEAK_DETECTION_CONTROLS_UPDATE.md](PEAK_DETECTION_CONTROLS_UPDATE.md)** - User-adjustable thresholds

### Features
- **[FILTERED_WINDOWS_EXPORT.md](FILTERED_WINDOWS_EXPORT.md)** - Exporting filtered PPG windows
- **[WEB_APP_TEMPLATE_UPDATE.md](WEB_APP_TEMPLATE_UPDATE.md)** - Web app template improvements

---

## Glucose Prediction

### Model Architecture
- **[RESNET34_GLUCOSE_PREDICTION.md](RESNET34_GLUCOSE_PREDICTION.md)** - ResNet34-1D model overview
- **[GLUCOSE_PREDICTION_ARCHITECTURE.md](GLUCOSE_PREDICTION_ARCHITECTURE.md)** - Complete architecture documentation (72 KB)
- **[GLUCOSE_PREDICTION_ARCHITECTURE.pdf](GLUCOSE_PREDICTION_ARCHITECTURE.pdf)** - PDF version (278 KB, 18 pages)

### How It Works
- **[RESNET34_PROCESSING_EXPLAINED.md](RESNET34_PROCESSING_EXPLAINED.md)** - Detailed explanation of PPG to glucose transformation (26 KB)
- **[RESNET34_PROCESSING_EXPLAINED.pdf](RESNET34_PROCESSING_EXPLAINED.pdf)** - PDF version (146 KB, 16 pages)

### Training
- **[TRAINING_QUICK_START.md](TRAINING_QUICK_START.md)** - Quick reference for training commands
- **[TRAINING_HARDWARE_AND_COSTING.md](TRAINING_HARDWARE_AND_COSTING.md)** - Hardware recommendations and Azure pricing (22 KB)
- **[TRAINING_HARDWARE_AND_COSTING.pdf](TRAINING_HARDWARE_AND_COSTING.pdf)** - PDF version (154 KB, 17 pages)

### Data Preparation
- **[GLUCOSE_LABELS_FEATURE.md](GLUCOSE_LABELS_FEATURE.md)** - Manual glucose label entry feature (19 KB)
- **[GLUCOSE_EXTRACTION_FEATURE.md](GLUCOSE_EXTRACTION_FEATURE.md)** - Automatic glucose extraction from VitalDB (15 KB)

---

## Reference Papers

- **[Nature_BGL_from_PPG_2025 (4).pdf](Nature_BGL_from_PPG_2025%20(4).pdf)** - Research paper on blood glucose from PPG (5.8 MB)

---

## Utility Scripts

- **[generate_pdf.py](generate_pdf.py)** - Generate PDF from markdown (generic)
- **[generate_processing_pdf.py](generate_processing_pdf.py)** - Generate ResNet processing PDF
- **[generate_hardware_pdf.py](generate_hardware_pdf.py)** - Generate hardware/costing PDF

---

## Documentation by Topic

### For New Users
1. Start with [README.md](README.md) for project overview
2. Follow [QUICK_START.md](QUICK_START.md) for first steps
3. Use [WEB_APP_README.md](WEB_APP_README.md) to understand the web interface

### For PPG Analysis
1. Read [PPG_README.md](PPG_README.md) for signal processing basics
2. Check [PEAK_DETECTION_EXTENDED.md](PEAK_DETECTION_EXTENDED.md) for peak detection
3. Refer to [PPG_TROUBLESHOOTING.md](PPG_TROUBLESHOOTING.md) for issues

### For Glucose Prediction
1. Understand the model: [RESNET34_GLUCOSE_PREDICTION.md](RESNET34_GLUCOSE_PREDICTION.md)
2. Learn how it works: [RESNET34_PROCESSING_EXPLAINED.md](RESNET34_PROCESSING_EXPLAINED.md) or [PDF version](RESNET34_PROCESSING_EXPLAINED.pdf)
3. Prepare training data: [GLUCOSE_EXTRACTION_FEATURE.md](GLUCOSE_EXTRACTION_FEATURE.md)
4. Train the model: [TRAINING_QUICK_START.md](TRAINING_QUICK_START.md)
5. Check hardware/costs: [TRAINING_HARDWARE_AND_COSTING.md](TRAINING_HARDWARE_AND_COSTING.md) or [PDF version](TRAINING_HARDWARE_AND_COSTING.pdf)

### For Training on Azure
1. Read [TRAINING_HARDWARE_AND_COSTING.pdf](TRAINING_HARDWARE_AND_COSTING.pdf) (17 pages)
   - Hardware recommendations
   - Azure VM options (NC-series, A100)
   - Pricing for different configurations
   - Cost optimization strategies
   - Setup guide

2. Use [TRAINING_QUICK_START.md](TRAINING_QUICK_START.md) for commands

---

## File Size Reference

### Large Files (> 100 KB)
- Nature_BGL_from_PPG_2025 (4).pdf: 5.8 MB
- GLUCOSE_PREDICTION_ARCHITECTURE.pdf: 278 KB
- TRAINING_HARDWARE_AND_COSTING.pdf: 154 KB
- RESNET34_PROCESSING_EXPLAINED.pdf: 146 KB

### Comprehensive Markdown (> 20 KB)
- GLUCOSE_PREDICTION_ARCHITECTURE.md: 72 KB
- RESNET34_PROCESSING_EXPLAINED.md: 26 KB
- TRAINING_HARDWARE_AND_COSTING.md: 22 KB
- GLUCOSE_LABELS_FEATURE.md: 19 KB

---

## Key Features Documented

### Web Application
- **5-step pipeline**: Extract → Cleanse → Detect Peaks → Filter Windows → Glucose Labels
- **Interactive plots**: Visualize signal at each step
- **User controls**: Adjust peak detection thresholds
- **CSV downloads**: Export data at each processing stage

### PPG Processing
- **Signal cleansing**: Bandpass filtering (0.5-8 Hz)
- **Peak detection**: Height and distance thresholds
- **Template filtering**: Cosine similarity (0.85 threshold)
- **Quality metrics**: Pass rate, similarity scores

### Glucose Prediction
- **ResNet34-1D**: 34-layer deep residual network
- **7.2M parameters**: Trained on PPG-glucose pairs
- **Automatic extraction**: Get glucose from VitalDB
- **3 matching methods**: Interpolate, nearest, last_known
- **Training ready**: Complete pipeline from raw data to model

---

## Summary

Total documentation: **21 markdown files** + **4 PDFs** + **3 Python scripts**

**Total size**: ~7.5 MB (mostly from research paper)

**Coverage**:
- ✓ Installation and setup
- ✓ Web application usage
- ✓ PPG signal processing
- ✓ Peak detection and filtering
- ✓ Glucose prediction model
- ✓ Training procedures
- ✓ Hardware requirements
- ✓ Azure cloud setup
- ✓ Cost estimates
- ✓ Troubleshooting

**Formats**:
- Markdown (.md) - Easy to read in editor
- PDF (.pdf) - Professional presentation
- Python (.py) - PDF generation scripts

---

## Contributing

When adding new documentation:
1. Place markdown files in this `docs/` folder
2. Update this INDEX.md with the new file
3. Generate PDF if it's a major document
4. Use consistent formatting and structure

---

**Last Updated**: November 2024
