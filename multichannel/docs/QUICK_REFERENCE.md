# Multi-Channel Pipeline - Quick Reference Card

## 🎯 One-Line Commands

### Process Single File
```bash
python generate_multichannel_training_data.py --input force-GLUC123-SYS140-DIA91.csv --output ./output
```

### Process Entire Folder
```bash
python generate_multichannel_training_data.py --input_folder "C:\senzrtech\Multi-channel\multi-channel-input-files" --output ./output
```

### Launch Web Viewer
```bash
python run_multichannel_web_app.py --data ./output
# Open: http://localhost:5001
```

### Combine Outputs
```bash
python combine_multichannel_outputs.py --input ./output --output training_data.csv
```

### High-Quality Only
```bash
python combine_multichannel_outputs.py --input ./output --output high_quality.csv --quality-only
```

---

## 📁 Input Format

### Filename
```
<channel>-GLUC<glucose>-SYS<systolic>-DIA<diastolic>.csv
```

### CSV Columns
- `time` - Timestamp (seconds)
- `ppg` / `signal` / `amplitude` - Signal values

---

## 📊 Output Files (Per Input)

```
output/
└── <filename>/
    ├── <filename>-01-raw.csv           # Raw
    ├── <filename>-02-cleaned.csv       # Time repaired
    ├── <filename>-03-downsampled.csv   # 100Hz
    ├── <filename>-04-preprocessed.csv  # Filtered
    ├── <filename>-05-peaks.csv         # Peaks
    ├── <filename>-06-windows.csv       # All windows
    ├── <filename>-07-filtered.csv      # Quality filtered
    ├── <filename>-08-template.csv      # Template
    ├── <filename>-output.csv           # ⭐ FINAL OUTPUT
    └── <filename>-metadata.json        # Stats
```

---

## 🔍 Quality Tags

- `8080xxxxxx` = Pure data (no repairs) ✅
- `4040xxxxxx` = Repaired data (interpolated) ⚠️

---

## ⚙️ Common Parameters

```bash
--height 0.3        # Peak height multiplier
--distance 0.8      # Peak distance multiplier
--similarity 0.85   # Template similarity threshold
--sampling_rate 100 # Override sampling rate
```

---

## 🌐 Web App URLs

- Dashboard: `http://localhost:5001/`
- Case view: `http://localhost:5001/case/<name>`
- API cases: `http://localhost:5001/api/cases`
- API features: `http://localhost:5001/api/case/<name>/features`

---

## 🐛 Quick Fixes

| Error | Solution |
|-------|----------|
| No 'time' column | CSV must have 'time' column |
| Can't extract glucose | Filename needs `GLUC<number>` |
| No valid windows | Lower thresholds: `--height 0.2 --distance 0.6` |
| Non-recoverable NaNs | Check raw data, has leading NaNs |
| No cases in web app | Check `--data` path is correct |

---

## 📚 Documentation

- **User Guide:** [`MULTICHANNEL_README.md`](MULTICHANNEL_README.md)
- **Validation:** [`MULTICHANNEL_VALIDATION.md`](MULTICHANNEL_VALIDATION.md)
- **Summary:** [`MULTICHANNEL_SUMMARY.md`](MULTICHANNEL_SUMMARY.md)

---

## 🚀 Complete Workflow

```bash
# 1. Process data
python generate_multichannel_training_data.py \
    --input_folder ./input \
    --output ./output

# 2. Visualize
python run_multichannel_web_app.py --data ./output

# 3. Combine
python combine_multichannel_outputs.py \
    --input ./output \
    --output training.csv \
    --quality-only

# 4. Train model (your code)
python train_model.py --data training.csv
```

---

## 📦 Files Created

1. **`generate_multichannel_training_data.py`** - Main processing pipeline
2. **`run_multichannel_web_app.py`** - Web app launcher
3. **`src/web_app/multichannel_web_app.py`** - Web app backend
4. **`combine_multichannel_outputs.py`** - Combine utility
5. **`MULTICHANNEL_README.md`** - Complete user guide
6. **`MULTICHANNEL_VALIDATION.md`** - Technical validation
7. **`MULTICHANNEL_SUMMARY.md`** - Overview document

---

## ✅ Processing Steps

1. Time Repair → 2. Signal Cleaning → 3. Downsample (100Hz) → 4. Bandpass Filter (0.5-8Hz) → 5. Peak Detection → 6. Window Extraction (1-sec) → 7. Template Filtering → 8. Quality Tagging → 9. Output

**All steps match VitalDB pipeline (d7.py) exactly!** ✅
