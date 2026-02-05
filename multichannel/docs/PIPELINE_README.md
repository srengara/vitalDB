# Multi-Channel PPG Processing Pipeline

Complete pipeline for processing multi-channel PPG signals with integrated visualization.

## 📁 Folder Structure

```
multichannel/
├── README.md                               # This file
├── INDEX.md                                # File index & descriptions
├── INSTALLATION.md                         # Installation guide
├── requirements.txt                        # Python dependencies
├── generate_multichannel_training_data.py  # Main processing pipeline
├── run_multichannel_web_app.py            # Web visualization launcher
├── combine_multichannel_outputs.py        # Output combination utility
├── src/
│   └── multichannel_web_app.py           # Web app backend
└── docs/
    ├── MULTICHANNEL_README.md            # Complete user guide
    ├── MULTICHANNEL_VALIDATION.md        # Technical validation
    ├── MULTICHANNEL_SUMMARY.md           # Overview document
    └── QUICK_REFERENCE.md                # Quick reference card
```

## 🚀 Quick Start

### 0. Installation

```bash
cd senzrTech/multichannel
pip install -r requirements.txt
```

See [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions.

### 1. Process Data

**Single file:**
```bash
python generate_multichannel_training_data.py \
    --input "C:\senzrtech\Multi-channel\multi-channel-input-files\force-GLUC123-SYS140-DIA91.csv" \
    --output ./output
```

**Batch process folder:**
```bash
python generate_multichannel_training_data.py \
    --input_folder "C:\senzrtech\Multi-channel\multi-channel-input-files" \
    --output ./output
```

### 2. Visualize Results

```bash
python run_multichannel_web_app.py --data ./output
```

Then open: http://localhost:5001

### 3. Combine Outputs (Optional)

```bash
# All channels
python combine_multichannel_outputs.py \
    --input ./output \
    --output training_data.csv

# High-quality windows only
python combine_multichannel_outputs.py \
    --input ./output \
    --output high_quality.csv \
    --quality-only
```

## 📋 Input File Format

### Filename Pattern
```
<channel>-GLUC<glucose>-SYS<systolic>-DIA<diastolic>.csv
```

### Examples
```
force-GLUC123-SYS140-DIA91.csv
Signal1-GLUC123-SYS140-DIA91.csv
Signal2-GLUC123-SYS140-DIA91.csv
Signal3-GLUC123-SYS140-DIA91.csv
```

### CSV Format
Required columns (case-insensitive):
- `time` - Timestamp in seconds
- `ppg` / `signal` / `amplitude` - Signal values

## 📂 Output Structure

For each input file, the pipeline creates:

```
output/
└── <filename>/
    ├── <filename>-01-raw.csv           # Raw data
    ├── <filename>-02-cleaned.csv       # Time repaired
    ├── <filename>-03-downsampled.csv   # 100Hz resampled
    ├── <filename>-04-preprocessed.csv  # Bandpass filtered
    ├── <filename>-05-peaks.csv         # Detected peaks
    ├── <filename>-06-windows.csv       # All windows
    ├── <filename>-07-filtered.csv      # Quality-filtered windows
    ├── <filename>-08-template.csv      # Template signal
    ├── <filename>-output.csv           # ⭐ FINAL OUTPUT
    └── <filename>-metadata.json        # Processing metadata
```

## 🔬 Processing Pipeline

The pipeline implements 9 processing steps (validated against `generate_vitaldb_training_data_d7.py`):

1. **Time Repair** - Reconstructs time axis if broken/missing
2. **Signal Cleaning** - Forward fill + leading NaN trim
3. **Downsampling** - Resamples to 100Hz
4. **Preprocessing** - Bandpass filter (0.5-8Hz, 3rd order)
5. **Peak Detection** - Height + distance thresholds
6. **Window Extraction** - 1-second windows (100 samples)
7. **Template Filtering** - Cosine similarity ≥0.85
8. **Quality Tagging** - 4040 (repaired) or 8080 (pure)
9. **Output Generation** - Wide format with glucose filtering

## 🎯 Key Differences from VitalDB Pipeline

| Feature | VitalDB (d7.py) | Multi-Channel |
|---------|----------------|---------------|
| Input | VitalDB case + labs CSV | CSV with glucose in filename |
| Glucose source | External labs CSV | Filename parsing (GLUC123) |
| Time windowing | ±8 minutes | Entire signal |
| Intermediate files | None | 8 steps saved |
| Visualization | Separate tool | Integrated web app |

**✅ All core signal processing is identical!**

## 📊 Final Output Format

**File:** `*-output.csv` (Wide format, training-ready)

Columns:
- `channel` - Signal channel name
- `window_index` - Unique ID with quality tag (4040/8080)
- `glucose_mg_dl` - Glucose value from filename
- `systolic_mmhg` - Systolic BP (if available)
- `diastolic_mmhg` - Diastolic BP (if available)
- `amplitude_sample_0` to `amplitude_sample_99` - 100 PPG samples

## 🌐 Web Visualization Features

The web app provides:

1. **Dashboard** - Grid view of all processed cases
2. **Signal Pipeline** - 4-panel transformation view
3. **Peak Detection** - Signal with detected peaks
4. **Window Comparison** - Before/after filtering
5. **Template Matching** - Template + sample windows
6. **Feature Tables** - Extracted features for training

## 🔧 Command-Line Options

### Processing Script

```bash
# Required (one of):
--input FILE            # Process single file
--input_folder FOLDER   # Process all CSV files in folder
--output DIR            # Output directory

# Optional:
--sampling_rate HZ      # Override sampling rate (default: auto-detect)
--height FLOAT          # Peak height multiplier (default: 0.3)
--distance FLOAT        # Peak distance multiplier (default: 0.8)
--similarity FLOAT      # Template similarity threshold (default: 0.85)
```

### Web App

```bash
--data DIR              # Output directory (default: ./output)
--port PORT             # Server port (default: 5001)
--host HOST             # Server host (default: 0.0.0.0)
```

### Combine Script

```bash
--input DIR             # Input directory with processed folders
--output FILE           # Output CSV file
--channels LIST         # Specific channels to include
--quality-only          # Only include 8080 (pure) windows
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No 'time' column found" | CSV must have 'time' column |
| "Could not extract glucose" | Filename must contain `GLUC<number>` |
| "No valid windows" | Try: `--height 0.2 --distance 0.6 --similarity 0.80` |
| "Non-recoverable NaNs" | Check raw data quality (has leading NaNs) |
| "No cases in web app" | Verify `--data` path points to output directory |

## 📚 Documentation

- **[Complete User Guide](docs/MULTICHANNEL_README.md)** - Detailed usage instructions
- **[Technical Validation](docs/MULTICHANNEL_VALIDATION.md)** - Line-by-line comparison with d7.py
- **[Overview & Summary](docs/MULTICHANNEL_SUMMARY.md)** - Feature summary and use cases
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - One-page command reference

## 🎯 Example Workflow

```bash
# 1. Process all channels
cd multichannel
python generate_multichannel_training_data.py \
    --input_folder "C:\senzrtech\Multi-channel\multi-channel-input-files" \
    --output ./output

# 2. Visualize results
python run_multichannel_web_app.py --data ./output
# Open: http://localhost:5001

# 3. Review quality in web app
# - Check filtering rates
# - Verify glucose values
# - Inspect intermediate files

# 4. Combine for training
python combine_multichannel_outputs.py \
    --input ./output \
    --output training_data.csv \
    --quality-only

# 5. Train your model
# Load training_data.csv in your ML pipeline
```

## ✅ Validation Status

**All 9 core processing steps validated against `generate_vitaldb_training_data_d7.py`:**

✅ Time repair (exact match)
✅ Signal cleaning (exact match)
✅ Downsampling (exact match)
✅ Preprocessing (exact match)
✅ Peak detection (exact match)
✅ Window extraction (exact match)
✅ Template filtering (exact match)
✅ Quality tagging (exact match)
✅ Output generation (exact match)

See [docs/MULTICHANNEL_VALIDATION.md](docs/MULTICHANNEL_VALIDATION.md) for detailed validation.

## 📞 Support

For detailed information, see the documentation in the `docs/` folder:
- User questions → [MULTICHANNEL_README.md](docs/MULTICHANNEL_README.md)
- Technical details → [MULTICHANNEL_VALIDATION.md](docs/MULTICHANNEL_VALIDATION.md)
- Quick commands → [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

---

**Ready to process your multi-channel PPG data!** 🚀
