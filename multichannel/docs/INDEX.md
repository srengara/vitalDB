# Multi-Channel PPG Processing Pipeline - File Index

## 📁 Complete Folder Structure

```
senzrTech/multichannel/
│
├── 📄 README.md                               ← Start here!
├── 📄 INDEX.md                                ← This file
│
├── 🐍 Python Scripts (Main Tools)
│   ├── generate_multichannel_training_data.py  ← Main processing pipeline
│   ├── run_multichannel_web_app.py            ← Web visualization launcher
│   └── combine_multichannel_outputs.py        ← Combine outputs utility
│
├── 📂 src/                                    ← Source code
│   └── multichannel_web_app.py               ← Flask web app backend
│
└── 📂 docs/                                   ← Documentation
    ├── MULTICHANNEL_README.md                ← Complete user guide
    ├── MULTICHANNEL_VALIDATION.md            ← Technical validation
    ├── MULTICHANNEL_SUMMARY.md               ← Overview & features
    └── QUICK_REFERENCE.md                    ← Quick command reference
```

---

## 📖 File Descriptions

### Main Scripts

#### 🐍 `generate_multichannel_training_data.py` (565 lines)
**Purpose:** Core processing pipeline

**What it does:**
- Loads multi-channel CSV files
- Extracts glucose/BP from filename (no labs CSV)
- Processes entire signal (no time windowing)
- Implements 9-step processing pipeline
- Saves 8 intermediate files per case
- Generates final training-ready output

**Usage:**
```bash
# Single file
python generate_multichannel_training_data.py \
    --input force-GLUC123-SYS140-DIA91.csv \
    --output ./output

# Batch process
python generate_multichannel_training_data.py \
    --input_folder ./input \
    --output ./output
```

**Key Features:**
- ✅ Validated against `generate_vitaldb_training_data_d7.py`
- ✅ Batch processing support
- ✅ Automatic metadata extraction
- ✅ Quality tagging (4040/8080)

---

#### 🐍 `run_multichannel_web_app.py` (47 lines)
**Purpose:** Web visualization launcher

**What it does:**
- Launches Flask web server
- Provides interactive dashboard
- Visualizes all processing steps
- Displays extracted features

**Usage:**
```bash
python run_multichannel_web_app.py --data ./output
# Open: http://localhost:5001
```

**Features:**
- Dashboard with all cases
- 5 visualization types per case
- Feature extraction tables
- API endpoints

---

#### 🐍 `combine_multichannel_outputs.py` (234 lines)
**Purpose:** Combine multiple outputs

**What it does:**
- Finds all `*-output.csv` files
- Combines into single training dataset
- Filters by channel or quality
- Generates summary statistics

**Usage:**
```bash
# All channels
python combine_multichannel_outputs.py \
    --input ./output \
    --output training.csv

# Quality filter
python combine_multichannel_outputs.py \
    --input ./output \
    --output high_quality.csv \
    --quality-only
```

---

### Source Code

#### 🐍 `src/multichannel_web_app.py` (795 lines)
**Purpose:** Flask web application backend

**Components:**
- `create_app()` - Flask app factory
- `get_available_cases()` - Scan output folders
- `load_case_data()` - Load all intermediate files
- `create_*_plot()` - Plotly visualization generators
- `extract_features()` - Feature extraction
- Route handlers for web pages and API

**Visualizations Generated:**
1. Signal comparison (4-panel)
2. Peak detection
3. Window extraction & filtering
4. Template matching
5. Feature tables

---

### Documentation

#### 📄 `docs/MULTICHANNEL_README.md` (580 lines)
**Complete user guide with:**
- Quick start examples
- Input/output format specs
- Command-line options
- Processing pipeline details
- Troubleshooting guide
- Integration examples
- Feature extraction details

**Best for:** New users, reference guide

---

#### 📄 `docs/MULTICHANNEL_VALIDATION.md` (460 lines)
**Technical validation document:**
- Line-by-line comparison with `d7.py`
- Validates all 9 processing steps
- Documents intentional modifications
- Confirms code equivalence

**Best for:** Technical review, code verification

---

#### 📄 `docs/MULTICHANNEL_SUMMARY.md` (520 lines)
**Overview document with:**
- Feature summary
- Comparison tables
- Use case examples
- Performance metrics
- Next steps guide

**Best for:** Quick overview, decision making

---

#### 📄 `docs/QUICK_REFERENCE.md` (140 lines)
**One-page quick reference:**
- Common commands
- Input/output formats
- Quality tags
- Quick fixes
- Complete workflow

**Best for:** Quick lookup, cheat sheet

---

## 🎯 Where to Start?

### First Time User?
1. Start with **[README.md](README.md)** for overview
2. Read **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** for commands
3. Process sample data using `generate_multichannel_training_data.py`
4. View results with `run_multichannel_web_app.py`

### Need Detailed Instructions?
→ **[docs/MULTICHANNEL_README.md](docs/MULTICHANNEL_README.md)**

### Want Technical Details?
→ **[docs/MULTICHANNEL_VALIDATION.md](docs/MULTICHANNEL_VALIDATION.md)**

### Need Quick Commands?
→ **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**

---

## 🚀 Quick Start Commands

```bash
# Navigate to multichannel folder
cd senzrTech/multichannel

# 1. Process data
python generate_multichannel_training_data.py \
    --input_folder "C:\senzrtech\Multi-channel\multi-channel-input-files" \
    --output ./output

# 2. Visualize
python run_multichannel_web_app.py --data ./output

# 3. Combine
python combine_multichannel_outputs.py \
    --input ./output \
    --output training_data.csv
```

---

## 📊 Pipeline Overview

```
Input CSV → Time Repair → Cleaning → Downsample (100Hz) →
Bandpass Filter → Peak Detection → Window Extraction →
Template Filtering → Quality Tagging → Output CSV
```

**Intermediate files saved at each step!**

---

## ✅ Validation Status

**All 9 processing steps validated against `generate_vitaldb_training_data_d7.py`**

| Step | Status |
|------|--------|
| Time Repair | ✅ Exact match |
| Signal Cleaning | ✅ Exact match |
| Downsampling | ✅ Exact match |
| Preprocessing | ✅ Exact match |
| Peak Detection | ✅ Exact match |
| Window Extraction | ✅ Exact match |
| Template Filtering | ✅ Exact match |
| Quality Tagging | ✅ Exact match |
| Output Generation | ✅ Exact match |

See [docs/MULTICHANNEL_VALIDATION.md](docs/MULTICHANNEL_VALIDATION.md) for details.

---

## 📦 Dependencies

These scripts use the shared libraries from `senzrTech/src/`:
- `src/data_extraction/ppg_segmentation.py` - PPG segmentation
- `src/data_extraction/peak_detection.py` - Peak detection algorithms

**No installation required** - imports are automatically resolved.

---

## 🔗 Related Files

**Original VitalDB pipeline:**
- `senzrTech/generate_vitaldb_training_data_d7.py` - Original implementation

**Shared libraries:**
- `senzrTech/src/data_extraction/` - Signal processing modules

**Original web app:**
- `senzrTech/run_web_app.py` - VitalDB web app launcher
- `senzrTech/src/web_app/web_app.py` - VitalDB web app

---

## 📞 Need Help?

1. **Check documentation:**
   - User guide: [docs/MULTICHANNEL_README.md](docs/MULTICHANNEL_README.md)
   - Troubleshooting: [docs/MULTICHANNEL_README.md#troubleshooting](docs/MULTICHANNEL_README.md)

2. **Review examples:**
   - Quick reference: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
   - Use cases: [docs/MULTICHANNEL_SUMMARY.md](docs/MULTICHANNEL_SUMMARY.md)

3. **Verify processing:**
   - Check intermediate files in output folder
   - Use web app to visualize each step
   - Review metadata JSON for statistics

---

**Ready to process multi-channel PPG data!** 🚀
