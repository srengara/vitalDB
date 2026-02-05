# PPG Analysis - Quick Start Guide

## One-Command Analysis

```bash
python ppg_analysis_pipeline.py --case-id 1
```

That's it! This will:
1. ✅ Extract PPG data
2. ✅ Segment into pulses
3. ✅ Calculate heart rate
4. ✅ Generate plots
5. ✅ Create HTML report

## View Results

Open the HTML report in your browser:
```
ppg_analysis/case_1_report.html
```

## Common Commands

### List available PPG tracks
```bash
python ppg_analysis_pipeline.py --case-id 1 --list-tracks
```

### Use different track
```bash
python ppg_analysis_pipeline.py --case-id 1 --track "Solar8000/PLETH"
```

### Custom output directory
```bash
python ppg_analysis_pipeline.py --case-id 1 --output ./my_results
```

## Python API (One Function)

```python
from ppg_analysis_pipeline import run_pipeline

run_pipeline(case_id=1, track_name='SNUADC/PLETH')
```

## Output Files

```
ppg_analysis/
├── case_1_SNUADC_PLETH.csv          # Raw data
├── case_1_segmentation.json         # Pulse data
└── case_1_report.html               # 🌟 Open this!
```

## What You Get

📊 **In the HTML Report:**
- Summary statistics (HR, pulse count, validity rate)
- PPG signal with detected pulses
- Heart rate variability plot
- Detailed pulse table with quality scores

## Next Steps

- Read [PPG_README.md](PPG_README.md) for detailed documentation
- Check example scripts for advanced usage
- Customize segmentation parameters as needed

## Requirements

```bash
pip install -r requirements.txt
```

## Help

```bash
python ppg_analysis_pipeline.py --help
```

---

**That's all you need to get started! 🚀**
