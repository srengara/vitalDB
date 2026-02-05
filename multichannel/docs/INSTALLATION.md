# Multi-Channel Pipeline - Installation Guide

## 📋 Prerequisites

- **Python:** >= 3.8 (Python 3.10+ recommended)
- **Operating System:** Windows, macOS, or Linux
- **pip:** Latest version recommended

Check your Python version:
```bash
python --version
```

## 🚀 Quick Install

### Option 1: Basic Installation (Recommended)

```bash
cd senzrTech/multichannel
pip install -r requirements.txt
```

This installs:
- ✅ numpy (numerical computing)
- ✅ pandas (data manipulation)
- ✅ scipy (signal processing)
- ✅ Flask (web server)
- ✅ plotly (visualization)
- ✅ pyarrow (Parquet support)

### Option 2: Minimal Installation (Core only)

```bash
pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0 Flask>=2.3.0 plotly>=5.14.0
```

## 🔍 Verify Installation

After installation, test that everything works:

```bash
# Test imports
python -c "import numpy, pandas, scipy, flask, plotly; print('✅ All dependencies installed!')"

# Test processing script
cd senzrTech/multichannel
python generate_multichannel_training_data.py --help
```

Expected output:
```
usage: generate_multichannel_training_data.py [-h] (--input INPUT |
                                              --input_folder INPUT_FOLDER)
                                              --output OUTPUT ...
```

## 📦 Dependency Details

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Array operations, numerical computing |
| pandas | ≥2.0.0 | DataFrames, CSV I/O |
| scipy | ≥1.10.0 | Signal filtering, resampling, peak detection |
| Flask | ≥2.3.0 | Web server for visualization |
| plotly | ≥5.14.0 | Interactive plots |
| pyarrow | ≥12.0.0 | Parquet file support (optional but recommended) |

### Optional Dependencies

```bash
# Parquet support (alternative to pyarrow)
pip install fastparquet>=2023.4.0

# Development tools
pip install pytest>=7.3.0 black>=23.0.0 flake8>=6.0.0
```

## 🐍 Virtual Environment (Recommended)

Using a virtual environment isolates dependencies:

### Create Virtual Environment

**Windows:**
```bash
cd senzrTech/multichannel
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
cd senzrTech/multichannel
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Deactivate Virtual Environment

```bash
deactivate
```

## 🔧 Troubleshooting

### Issue: pip not found

**Solution:**
```bash
# Try python -m pip instead
python -m pip install -r requirements.txt
```

### Issue: Permission denied on Linux/macOS

**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt
```

### Issue: numpy/scipy installation fails on Windows

**Solution:** Install pre-built wheels from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/)

Or use conda:
```bash
conda install numpy pandas scipy
pip install Flask plotly pyarrow
```

### Issue: Old pip version

**Solution:** Upgrade pip first:
```bash
python -m pip install --upgrade pip
```

### Issue: pyarrow installation fails

**Solution:** Skip pyarrow (it's optional):
```bash
pip install numpy pandas scipy Flask plotly
```

The pipeline will still work, but will save CSV instead of Parquet for final output.

## 📁 Shared Dependencies

This project uses modules from the parent `senzrTech/src/` directory:
- `src/data_extraction/ppg_segmentation.py`
- `src/data_extraction/peak_detection.py`

These modules only require numpy, pandas, and scipy (already installed).

**No additional installation needed** - imports are handled automatically.

## ✅ Post-Installation Test

Run a complete test to ensure everything works:

```bash
cd senzrTech/multichannel

# 1. Test processing script
python -c "from generate_multichannel_training_data import extract_metadata_from_filename; print('✅ Processing script OK')"

# 2. Test web app
python -c "from src.multichannel_web_app import create_app; print('✅ Web app OK')"

# 3. Test shared modules
python -c "import sys, os; sys.path.insert(0, '..'); from src.data_extraction.ppg_segmentation import PPGSegmenter; print('✅ Shared modules OK')"
```

All tests should print "✅ OK"

## 🚀 Ready to Go!

Your installation is complete! Try processing a sample file:

```bash
cd senzrTech/multichannel

# Process a single file
python generate_multichannel_training_data.py \
    --input "path/to/force-GLUC123-SYS140-DIA91.csv" \
    --output ./test_output

# Launch web viewer
python run_multichannel_web_app.py --data ./test_output
# Open: http://localhost:5001
```

## 📚 Next Steps

- **User Guide:** [docs/MULTICHANNEL_README.md](docs/MULTICHANNEL_README.md)
- **Quick Reference:** [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
- **Main README:** [README.md](README.md)

---

## 🆘 Still Having Issues?

1. **Check Python version:** Must be ≥ 3.8
2. **Update pip:** `python -m pip install --upgrade pip`
3. **Try virtual environment:** Isolates dependencies
4. **Check error messages:** Often indicate missing system libraries

For detailed error messages, run:
```bash
pip install -r requirements.txt -v
```

The `-v` flag shows verbose output for debugging.
