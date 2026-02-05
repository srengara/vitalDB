# VitalDB Utility - Summary

## Overview

A comprehensive Python utility for accessing and downloading case data from the VitalDB surgical patient biosignal dataset (6,388 cases).

## Files Created

### Core Files
- **[vitaldb_utility.py](vitaldb_utility.py)** - Main utility class with all functionality
- **[requirements.txt](requirements.txt)** - Python dependencies (requests)

### Command-Line Tools
- **[download_case.py](download_case.py)** - CLI tool for downloading case data
- **[list_case_ids.py](list_case_ids.py)** - Simple script to list all case IDs

### Example Scripts
- **[example_usage.py](example_usage.py)** - Examples for listing and querying cases
- **[example_download.py](example_download.py)** - Examples for downloading case data

### Documentation
- **[README.md](README.md)** - Complete documentation
- **[QUICK_START.md](QUICK_START.md)** - Quick reference guide
- **[SUMMARY.md](SUMMARY.md)** - This file

## Key Features

### 1. Query Capabilities
- List all 6,388 case IDs
- Get 196 available biosignal tracks
- Filter cases by required tracks (AND logic)
- Filter cases by optional tracks (OR logic)
- Get cases organized by track name

### 2. Download Capabilities
- Download clinical information for any case
- Download biosignal data (waveforms and parameters)
- Download specific tracks or all tracks
- Automatic file organization
- Progress logging

### 3. Data Access
- RESTful API integration
- Automatic gzip/plain text handling
- CSV parsing and export
- JSON metadata export
- Caching for performance

## Quick Usage

### List all case IDs
```bash
python list_case_ids.py
```

### List tracks for a case
```bash
python download_case.py 1 --list-tracks
```

### Download case data
```bash
python download_case.py 1 SNUADC/ECG_II SNUADC/ART
```

### Python API
```python
from vitaldb_utility import VitalDBUtility

util = VitalDBUtility()

# Get all case IDs
case_ids = util.get_all_case_ids()  # Returns 6,388 IDs

# Find cases with ECG and arterial pressure
cases = util.get_cases_with_tracks(['SNUADC/ECG_II', 'SNUADC/ART'])  # Returns 3,644 cases

# Download case data
result = util.download_case_data(1, ['SNUADC/ECG_II', 'SNUADC/ART'], './data')
```

## Test Results

### Successfully Tested
✓ Listed all 6,388 case IDs
✓ Retrieved 196 unique track types
✓ Found 3,644 cases with ECG_II and ART
✓ Listed 76 tracks available for case 1
✓ Downloaded clinical information (77 fields)
✓ Downloaded biosignal data (5.7M data points per track)
✓ Created organized directory structure
✓ Exported JSON metadata
✓ Exported CSV waveform data

### Example Download Output
```
test_downloads/case_1/
├── case_info.json          # Clinical info (age, sex, operation, labs, etc.)
├── tracks.json             # Track metadata
├── SNUADC_ECG_II.csv      # 5,770,575 ECG data points
└── SNUADC_ART.csv         # 5,770,575 arterial pressure data points
```

## API Endpoints Used

- `https://api.vitaldb.net/cases` - Clinical information
- `https://api.vitaldb.net/trks` - Track metadata
- `https://api.vitaldb.net/{tid}` - Track data

## Data Statistics

- **Total cases**: 6,388
- **Total tracks**: 196 unique types
- **Case ID range**: 1 to 6,388
- **Cases with ECG_II + ART**: 3,644
- **Average tracks per case**: ~50-100
- **Data points per waveform**: Millions (depends on duration and sampling rate)

## Common Track Types

**Vital Signs**: ECG, ART, PLETH, HR, SpO2
**Blood Pressure**: SBP, DBP, MBP, NIBP
**Anesthesia**: BIS, EMG, MAC, ETCO2, FIO2
**Ventilation**: TV, RR, PIP, PEEP, MV

## Performance

- **Caching**: Automatic caching of API responses
- **API calls**: ~2-3 seconds per request
- **Download time**: 10-60 seconds per track (depends on size)
- **Data size**: ~50-100 MB per high-frequency waveform

## Requirements

- Python 3.6+
- requests library
- Internet connection (for API access)

## Use Cases

1. **Research**: Download specific biosignals for analysis
2. **Machine Learning**: Build datasets filtered by specific criteria
3. **Clinical Studies**: Access pre-operative and intra-operative data
4. **Algorithm Development**: Test on real surgical patient data
5. **Education**: Learn from real-world biosignal data

## Important Notes

1. **Data Use Agreement**: All users must agree to VitalDB's Data Use Agreement
2. **Network Required**: Internet connection needed for API access
3. **Large Downloads**: Complete case data can be several hundred MB
4. **Time Format**: All times are in seconds from case start
5. **Missing Values**: Waveforms may have gaps (empty values in CSV)

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Read the quick start guide: [QUICK_START.md](QUICK_START.md)
3. Run examples: `python example_usage.py`
4. Download test case: `python download_case.py 1 SNUADC/ECG_II`

## Support

- Full documentation: [README.md](README.md)
- Quick reference: [QUICK_START.md](QUICK_START.md)
- VitalDB website: https://vitaldb.net/dataset/
- VitalDB documentation: https://vitaldb.net/dataset/?query=overview
