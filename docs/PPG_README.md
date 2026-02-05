# PPG Analysis Toolkit for VitalDB

Complete toolkit for extracting, segmenting, and visualizing PPG (Photoplethysmography) signals from the VitalDB dataset.

## Features

✅ **PPG Extraction** - Download PPG signals from VitalDB cases
✅ **Signal Processing** - Preprocessing with filtering and smoothing
✅ **Pulse Segmentation** - Precision interval detection algorithm
✅ **Quality Assessment** - Automatic quality scoring for each pulse
✅ **Visualization** - Matplotlib plots and interactive HTML reports
✅ **Heart Rate Analysis** - HR variability and interval analysis

## PPG Tracks in VitalDB

| Track Name | Sampling Rate | Description |
|------------|---------------|-------------|
| SNUADC/PLETH | 500 Hz | Most common, high resolution |
| Solar8000/PLETH | 62.5 Hz | Solar monitor |
| Primus/PLETH | 100 Hz | Primus anesthesia monitor |
| BIS/PLETH | Variable | BIS monitor |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Option 1: Complete Pipeline (Recommended)

Run the complete analysis pipeline with one command:

```bash
# Analyze case 1 with default SNUADC/PLETH track
python ppg_analysis_pipeline.py --case-id 1

# Analyze with specific track
python ppg_analysis_pipeline.py --case-id 1 --track "Solar8000/PLETH"

# List available tracks
python ppg_analysis_pipeline.py --case-id 1 --list-tracks

# Custom output directory
python ppg_analysis_pipeline.py --case-id 1 --output ./my_analysis
```

### Option 2: Step-by-Step

#### Step 1: Extract PPG Data

```python
from ppg_extractor import PPGExtractor

extractor = PPGExtractor()

# Check available PPG tracks for a case
tracks = extractor.get_available_ppg_tracks(case_id=1)
for track in tracks:
    print(f"- {track['tname']}")

# Extract specific track
result = extractor.extract_ppg(
    case_id=1,
    track_name='SNUADC/PLETH',
    output_dir='./ppg_data'
)

print(f"Saved to: {result['csv_file']}")
print(f"Samples: {result['num_samples']:,}")
print(f"Duration: {result['duration_seconds']:.2f} seconds")
```

#### Step 2: Segment PPG Signal

```python
from ppg_segmentation import PPGSegmenter

segmenter = PPGSegmenter(
    sampling_rate=500.0,  # Hz
    min_heart_rate=40.0,   # bpm
    max_heart_rate=180.0   # bpm
)

# Segment from CSV file
result = segmenter.segment_from_file(
    csv_file='./ppg_data/case_1_SNUADC_PLETH.csv',
    output_file='./ppg_data/case_1_segmentation.json'
)

# Print summary
print(f"Total pulses: {result['summary']['total_pulses']}")
print(f"Valid pulses: {result['summary']['valid_pulses']}")
print(f"Mean HR: {result['summary']['mean_heart_rate']:.1f} bpm")
```

#### Step 3: Create Plots

```python
from ppg_plotter import PPGPlotter

plotter = PPGPlotter()

# Plot signal
plotter.plot_ppg_signal(
    csv_file='./ppg_data/case_1_SNUADC_PLETH.csv',
    end_time=30,  # First 30 seconds
    title='PPG Signal - Case 1'
)

# Plot multiple segments
plotter.plot_ppg_segments(
    csv_file='./ppg_data/case_1_SNUADC_PLETH.csv',
    start_time=0,
    duration=10,
    num_segments=5
)
```

#### Step 4: Generate HTML Report

```python
from ppg_visualizer import PPGHTMLVisualizer

visualizer = PPGHTMLVisualizer()

visualizer.generate_html(
    csv_file='./ppg_data/case_1_SNUADC_PLETH.csv',
    segmentation_file='./ppg_data/case_1_segmentation.json',
    output_file='./ppg_data/case_1_report.html'
)
```

## Output Files

After running the pipeline, you'll get:

```
ppg_analysis/
├── case_1_SNUADC_PLETH.csv              # Raw PPG data
├── case_1_SNUADC_PLETH_metadata.json     # Extraction metadata
├── case_1_initial_plot.png               # Initial signal plot
├── case_1_segmentation.json              # Segmentation results
├── case_1_segmentation_pulses.csv        # Pulse data (CSV)
├── case_1_segment_1.png                  # Segment plots
├── case_1_segment_2.png
├── case_1_segment_3.png
└── case_1_report.html                    # 🌟 Interactive HTML report
```

## HTML Report Features

The generated HTML report includes:

- 📊 **Summary Statistics** - Total pulses, validity rate, mean HR, etc.
- 📈 **Signal Visualization** - PPG waveform with segmentation overlay
- 💓 **Heart Rate Plot** - HR variability over time
- 📋 **Pulse Details Table** - Individual pulse metrics with quality scores
- ⚙️ **Parameters** - Segmentation settings and metadata

## Segmentation Algorithm

The precision interval segmentation algorithm:

### 1. Preprocessing
- DC component removal
- Bandpass filtering (0.5-10 Hz)
- Savitzky-Golay smoothing

### 2. Peak Detection
- Detects systolic peaks using `scipy.signal.find_peaks`
- Parameters: minimum distance, height threshold, prominence

### 3. Valley Detection
- Detects diastolic valleys (pulse onsets)
- Searches between consecutive peaks

### 4. Quality Assessment

Each pulse is scored based on:
- **Amplitude** (40%) - Peak-to-valley height
- **Shape** (40%) - Monotonic rise and fall
- **Duration** (20%) - Valid heart rate range

Quality score: 0-1 (higher is better)

### 5. Interval Extraction

For each valid pulse:
- Onset time
- Peak time
- Pulse interval
- Heart rate
- Amplitude
- Quality score

## Customization

### Adjust Segmentation Parameters

```python
segmenter = PPGSegmenter(
    sampling_rate=500.0,     # Your signal's sampling rate
    min_heart_rate=40.0,     # Minimum expected HR (bpm)
    max_heart_rate=180.0,    # Maximum expected HR (bpm)
)

# Change quality threshold
segmenter.quality_threshold = 0.6  # Default: 0.5
```

### Filter Specific Heart Rate Range

```python
# After segmentation
valid_pulses = [
    p for p in result['pulses']
    if p['valid'] and 60 <= p['heart_rate'] <= 100
]

print(f"Pulses in 60-100 bpm range: {len(valid_pulses)}")
```

## Finding Cases with PPG Data

```python
from ppg_extractor import PPGExtractor

extractor = PPGExtractor()

# Find all cases with SNUADC/PLETH
cases = extractor.find_cases_with_ppg('SNUADC/PLETH')
print(f"Found {len(cases)} cases with SNUADC/PLETH")

# Find cases with any PPG track
all_ppg_cases = extractor.find_cases_with_ppg()
print(f"Found {len(all_ppg_cases)} cases with PPG data")
```

## Module Reference

### ppg_extractor.py

**Class:** `PPGExtractor`

**Methods:**
- `get_available_ppg_tracks(case_id)` - List PPG tracks for a case
- `extract_ppg(case_id, track_name, output_dir)` - Extract PPG data
- `extract_all_ppg_tracks(case_id, output_dir)` - Extract all PPG tracks
- `find_cases_with_ppg(track_name)` - Find cases with PPG data

### ppg_segmentation.py

**Class:** `PPGSegmenter`

**Methods:**
- `load_ppg_data(csv_file)` - Load PPG from CSV
- `preprocess_signal(signal)` - Filter and smooth signal
- `detect_peaks(signal)` - Find systolic peaks
- `detect_valleys(signal, peaks)` - Find diastolic valleys
- `calculate_pulse_quality(...)` - Score pulse quality
- `segment_pulses(time, signal)` - Complete segmentation
- `segment_from_file(csv_file, output_file)` - Segment from file

### ppg_plotter.py

**Class:** `PPGPlotter`

**Methods:**
- `load_ppg_data(csv_file)` - Load data
- `plot_ppg_signal(csv_file, ...)` - Plot signal
- `plot_ppg_segments(csv_file, ...)` - Plot multiple segments
- `plot_ppg_with_overlay(csv_file, overlay_data, ...)` - Overlay plot
- `plot_ppg_comparison(csv_files, ...)` - Compare multiple signals

### ppg_visualizer.py

**Class:** `PPGHTMLVisualizer`

**Methods:**
- `create_signal_plot(csv_file, segmentation_data)` - Create signal plot
- `create_heart_rate_plot(segmentation_data)` - Create HR plot
- `generate_html(csv_file, segmentation_file, output_file)` - Generate report

### ppg_analysis_pipeline.py

**Function:** `run_pipeline(case_id, track_name, sampling_rate, output_dir)`

Runs complete analysis pipeline.

## Examples

### Example 1: Batch Analysis

```python
from ppg_extractor import PPGExtractor
from ppg_analysis_pipeline import run_pipeline

extractor = PPGExtractor()

# Find cases with PPG
cases = extractor.find_cases_with_ppg('SNUADC/PLETH')[:10]  # First 10

# Analyze each case
for case_id in cases:
    print(f"\nAnalyzing case {case_id}...")
    try:
        run_pipeline(
            case_id=case_id,
            track_name='SNUADC/PLETH',
            output_dir=f'./batch_analysis/case_{case_id}'
        )
    except Exception as e:
        print(f"Failed: {e}")
```

### Example 2: Heart Rate Statistics

```python
import json
import numpy as np

# Load segmentation results
with open('case_1_segmentation.json', 'r') as f:
    data = json.load(f)

# Extract heart rates from valid pulses
hrs = [p['heart_rate'] for p in data['pulses'] if p['valid']]

print(f"Heart Rate Statistics:")
print(f"  Mean: {np.mean(hrs):.1f} bpm")
print(f"  Std: {np.std(hrs):.1f} bpm")
print(f"  Min: {np.min(hrs):.1f} bpm")
print(f"  Max: {np.max(hrs):.1f} bpm")
print(f"  Median: {np.median(hrs):.1f} bpm")
```

### Example 3: Export to Different Formats

```python
import pandas as pd

# Load pulse data
df = pd.read_csv('case_1_segmentation_pulses.csv')

# Filter valid pulses
valid_df = df[df['valid'] == True]

# Export to Excel
valid_df.to_excel('case_1_valid_pulses.xlsx', index=False)

# Export specific columns
hr_df = valid_df[['pulse_number', 'onset_time', 'heart_rate', 'quality']]
hr_df.to_csv('case_1_heart_rates.csv', index=False)
```

## Troubleshooting

### Issue: "No PPG tracks found"
**Solution:** The case may not have PPG data. Use `find_cases_with_ppg()` to find suitable cases.

### Issue: "Insufficient peaks detected"
**Solution:**
- Check signal quality
- Adjust `min_heart_rate` and `max_heart_rate` parameters
- Try different sampling rate

### Issue: "Low validity rate"
**Solution:**
- Lower `quality_threshold` (default: 0.5)
- Check if sampling rate is correct
- Signal may be noisy - try different track

### Issue: Import errors
**Solution:** Install all dependencies:
```bash
pip install -r requirements.txt
```

## Performance Notes

- **Extraction**: ~10-30 seconds per track (depends on duration)
- **Segmentation**: ~5-10 seconds for 10 minutes of 500Hz data
- **HTML Generation**: ~5-10 seconds

## Tips

1. **Start with short durations** - Test with 30-60 seconds before full analysis
2. **Check sampling rate** - Use metadata to verify correct rate
3. **Visualize first** - Plot signal before segmentation to check quality
4. **Batch processing** - Use pipeline for multiple cases
5. **Quality filtering** - Adjust threshold based on your needs

## Citation

If you use VitalDB dataset, please cite:

Lee, H.C., Jung, C.W. Vital Recorder—a free research tool for automatic recording of high-resolution time-synchronised physiological data from multiple anaesthesia devices. Sci Data 5, 180305 (2018). https://doi.org/10.1038/sdata.2018.305

## License

This toolkit is provided for research and educational purposes. Please refer to VitalDB's Data Use Agreement for dataset usage terms.

## Support

For issues or questions about:
- **VitalDB dataset**: https://vitaldb.net/dataset/
- **This toolkit**: Check the code documentation and examples

---

**Version**: 1.0
**Last Updated**: January 2025
