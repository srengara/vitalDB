# Multi-Channel PPG Processing Pipeline

## 📍 Location

The multi-channel processing pipeline has been moved to:

```
senzrTech/multichannel/
```

## 📚 Documentation

All documentation and scripts are now organized in the multichannel folder:

### Start Here
- **[multichannel/README.md](multichannel/README.md)** - Main documentation and quick start
- **[multichannel/INDEX.md](multichannel/INDEX.md)** - Complete file index and descriptions

### Main Scripts
- **[multichannel/generate_multichannel_training_data.py](multichannel/generate_multichannel_training_data.py)** - Processing pipeline
- **[multichannel/run_multichannel_web_app.py](multichannel/run_multichannel_web_app.py)** - Web visualization
- **[multichannel/combine_multichannel_outputs.py](multichannel/combine_multichannel_outputs.py)** - Combine outputs

### Detailed Documentation
- **[multichannel/docs/MULTICHANNEL_README.md](multichannel/docs/MULTICHANNEL_README.md)** - Complete user guide
- **[multichannel/docs/MULTICHANNEL_VALIDATION.md](multichannel/docs/MULTICHANNEL_VALIDATION.md)** - Technical validation
- **[multichannel/docs/MULTICHANNEL_SUMMARY.md](multichannel/docs/MULTICHANNEL_SUMMARY.md)** - Overview & summary
- **[multichannel/docs/QUICK_REFERENCE.md](multichannel/docs/QUICK_REFERENCE.md)** - Quick reference

## 🚀 Quick Start

```bash
# Navigate to multichannel folder
cd senzrTech/multichannel

# Process your data
python generate_multichannel_training_data.py \
    --input_folder "C:\senzrtech\Multi-channel\multi-channel-input-files" \
    --output ./output

# Visualize results
python run_multichannel_web_app.py --data ./output

# Open browser: http://localhost:5001
```

## 📂 Folder Structure

```
senzrTech/
├── multichannel/                          ← Multi-channel pipeline (NEW!)
│   ├── README.md                         ← Start here
│   ├── INDEX.md                          ← File index
│   ├── generate_multichannel_training_data.py
│   ├── run_multichannel_web_app.py
│   ├── combine_multichannel_outputs.py
│   ├── src/
│   │   └── multichannel_web_app.py
│   └── docs/
│       ├── MULTICHANNEL_README.md
│       ├── MULTICHANNEL_VALIDATION.md
│       ├── MULTICHANNEL_SUMMARY.md
│       └── QUICK_REFERENCE.md
│
├── generate_vitaldb_training_data_d7.py  ← Original VitalDB pipeline
├── run_web_app.py                        ← Original web app
└── src/                                  ← Shared libraries
    ├── data_extraction/
    │   ├── ppg_extractor.py
    │   ├── ppg_segmentation.py
    │   └── peak_detection.py
    └── web_app/
        └── web_app.py
```

## 🔗 Key Differences

| Feature | VitalDB Pipeline | Multi-Channel Pipeline |
|---------|------------------|------------------------|
| Location | `senzrTech/` | `senzrTech/multichannel/` |
| Input | VitalDB case + labs CSV | CSV with glucose in filename |
| Windowing | ±8 minutes | Entire signal |
| Intermediate files | None | 8 steps saved |
| Web app | Separate | Integrated |

**✅ All core signal processing is identical!**

---

**Navigate to [multichannel/](multichannel/) to get started!** 🚀
