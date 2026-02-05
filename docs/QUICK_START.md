# VitalDB Utility - Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Common Tasks

### 1. List All Case IDs

```bash
python list_case_ids.py
```

Or in Python:

```python
from vitaldb_utility import VitalDBUtility

util = VitalDBUtility()
case_ids = util.get_all_case_ids()
print(f"Total cases: {len(case_ids)}")
print(case_ids)
```

### 2. List Available Tracks for a Case

```bash
python download_case.py 1 --list-tracks
```

Or in Python:

```python
util = VitalDBUtility()
tracks = util.get_case_tracks(1)
for track in tracks:
    print(f"{track['tname']}")
```

### 3. Download Case Data

**Download all tracks for a case:**

```bash
python download_case.py 1
```

**Download specific tracks:**

```bash
python download_case.py 1 SNUADC/ECG_II SNUADC/ART
```

**Download to specific directory:**

```bash
python download_case.py 1 SNUADC/ECG_II --output ./my_data
```

### 4. Find Cases with Specific Biosignals

**Find cases with ECG and arterial pressure:**

```python
util = VitalDBUtility()
cases = util.get_cases_with_tracks(['SNUADC/ECG_II', 'SNUADC/ART'])
print(f"Found {len(cases)} cases with both ECG and ART")
```

**Find cases with any of several signals:**

```python
cases = util.get_cases_with_any_track(['SNUADC/ECG_II', 'SNUADC/ART', 'SNUADC/PLETH'])
print(f"Found {len(cases)} cases")
```

### 5. Get Case Information

```python
util = VitalDBUtility()
case_info = util.get_case_info(1)

print(f"Age: {case_info['age']}")
print(f"Sex: {case_info['sex']}")
print(f"Operation: {case_info['opname']}")
print(f"ASA: {case_info['asa']}")
```

### 6. Download and Analyze Data

```python
import pandas as pd
from vitaldb_utility import VitalDBUtility

# Download data
util = VitalDBUtility()
result = util.download_case_data(
    case_id=1,
    track_names=['SNUADC/ECG_II', 'SNUADC/ART'],
    output_dir='./data'
)

# Load downloaded data
ecg_file = [f for f in result['data_files'] if 'ECG' in f][0]
ecg_data = pd.read_csv(ecg_file)

print(f"ECG data shape: {ecg_data.shape}")
print(ecg_data.head())
```

## Data Structure

### Downloaded Files

When you download case data, you get:

```
case_<id>/
├── case_info.json        # Clinical information
├── tracks.json           # Track metadata
├── SNUADC_ECG_II.csv    # ECG data
├── SNUADC_ART.csv       # Arterial pressure data
└── ...                   # Other tracks
```

### Case Info Fields

Key fields in `case_info.json`:
- `age`, `sex`, `height`, `weight`, `bmi`
- `opname` - Operation name
- `optype` - Operation type
- `asa` - ASA classification
- `preop_*` - Pre-operative lab values
- `intraop_*` - Intra-operative data

### Track Data Format

CSV files with columns:
- `Time` - Time in seconds from case start
- `<track_name>` - Signal values

## Common Track Names

**Vital Signs:**
- `SNUADC/ECG_II` - ECG lead II
- `SNUADC/ECG_V5` - ECG lead V5
- `SNUADC/ART` - Arterial pressure waveform
- `SNUADC/PLETH` - Plethysmography waveform
- `Solar8000/HR` - Heart rate
- `Solar8000/PLETH_SPO2` - SpO2

**Blood Pressure:**
- `Solar8000/ART_SBP` - Systolic BP
- `Solar8000/ART_DBP` - Diastolic BP
- `Solar8000/ART_MBP` - Mean BP
- `Solar8000/NIBP_*` - Non-invasive BP

**Anesthesia:**
- `BIS/BIS` - Bispectral index
- `BIS/EMG` - EMG
- `Primus/ETCO2` - End-tidal CO2
- `Primus/MAC` - MAC
- `Primus/FIO2` - FiO2

**Ventilation:**
- `Primus/TV` - Tidal volume
- `Primus/RR_CO2` - Respiratory rate
- `Primus/PIP_MBAR` - Peak inspiratory pressure
- `Primus/PEEP_MBAR` - PEEP

## Tips

1. **Use caching**: The utility caches results by default for faster repeated queries
2. **Start small**: Download a few tracks first to test before downloading all data
3. **Check available tracks**: Always use `--list-tracks` first to see what's available
4. **Large downloads**: Downloading all tracks for a case can take several minutes
5. **Data size**: Each high-frequency waveform can have millions of data points

## Examples

Run the example scripts to see the utility in action:

```bash
# List all case IDs and available tracks
python example_usage.py

# Interactive download examples
python example_download.py
```

## Need Help?

- Check [README.md](README.md) for full API documentation
- Run `python download_case.py --help` for CLI help
- View example scripts for common use cases
