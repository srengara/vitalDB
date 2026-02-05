# PPG Toolkit - Troubleshooting Guide

## Problem: Empty PPG Data Files

### Symptoms
- CSV files are created but contain no data rows
- Files like `case_1_SNUADC_PLETH.csv` and `case_2_SNUADC_PLETH.csv` are empty
- Error: "No valid signal data found"

### Cause
Many VitalDB cases have PPG tracks listed in metadata, but the actual signal data may be:
- Completely empty (all NULL/NaN values)
- Too sparse (< 100 valid samples)
- Not recorded for that particular case

This is normal - not all surgical cases have all monitoring equipment connected.

### Solution 1: Use the Improved Pipeline (Recommended)

The updated pipeline automatically tries alternative tracks:

```bash
# This will automatically find the best available PPG track
python ppg_analysis_pipeline.py --case-id 1
```

The pipeline now:
1. ✅ Checks for empty data before saving
2. ✅ Tries alternative tracks automatically
3. ✅ Reports which tracks failed and why
4. ✅ Only proceeds if valid data is found

### Solution 2: Find Valid Cases First

Use the case finder to identify cases with usable PPG data:

```bash
# Find first 10 cases with valid PPG
python find_valid_ppg_cases.py

# This will test cases and report which ones have valid data
# Output example:
# [1/100] Testing case 1... ✗ Invalid
# [2/100] Testing case 2... ✗ Invalid
# [3/100] Testing case 3... ✓ VALID
#   Track: SNUADC/PLETH
#   Samples: 2,500,000
#   Duration: 5000.0s
```

Then analyze the valid cases:
```bash
python ppg_analysis_pipeline.py --case-id <valid_case_id>
```

### Solution 3: Check Specific Cases

Find which tracks are available and have data:

```python
from ppg_extractor import PPGExtractor

extractor = PPGExtractor()

# Check case 100
case_id = 100

# List available PPG tracks
tracks = extractor.get_available_ppg_tracks(case_id)
print(f"Case {case_id} has {len(tracks)} PPG tracks:")
for track in tracks:
    print(f"  - {track['tname']}")

# Try to extract the best one
try:
    result = extractor.extract_best_ppg_track(case_id, './test_data')
    print(f"\nSuccess! Found valid data in: {result['track_name']}")
    print(f"Samples: {result['num_samples']:,}")
except ValueError as e:
    print(f"\nNo valid PPG data: {e}")
```

## Problem: "Insufficient data: only X valid samples found"

### Cause
The PPG signal exists but has too few valid (non-NaN) values to be useful.

### Solution
- Try a different case
- Use the case finder to locate cases with substantial data
- Some surgical procedures may have shorter monitoring periods

## Problem: All Cases Failing

### Check 1: Verify VitalDB Access
```python
from vitaldb_utility import VitalDBUtility

util = VitalDBUtility()
cases = util.get_all_case_ids()
print(f"Total cases: {len(cases)}")  # Should be 6388

# Check specific case info
info = util.get_case_info(1)
print(info)  # Should return case details
```

### Check 2: Network Connection
The toolkit requires internet access to download VitalDB data.

```bash
# Test API access
curl https://api.vitaldb.net/cases
```

### Check 3: Dependencies
```bash
pip install -r requirements.txt --upgrade
```

## Problem: "Track 'SNUADC/PLETH' not found"

### Cause
The specified track is not available for that case.

### Solution
```bash
# List available tracks first
python ppg_analysis_pipeline.py --case-id 1 --list-tracks

# Then use an available track
python ppg_analysis_pipeline.py --case-id 1 --track "Solar8000/PLETH"
```

Or let the pipeline choose automatically:
```bash
# Don't specify --track, let it find the best one
python ppg_analysis_pipeline.py --case-id 1
```

## Problem: Segmentation Fails

### Error: "Insufficient peaks detected"

**Cause:** Signal quality is poor or parameters don't match the data.

**Solutions:**
1. Adjust heart rate range:
```python
from ppg_segmentation import PPGSegmenter

segmenter = PPGSegmenter(
    sampling_rate=500.0,
    min_heart_rate=30.0,   # Lower for bradycardia
    max_heart_rate=200.0   # Higher for tachycardia
)
```

2. Check sampling rate:
```bash
# Let pipeline auto-detect
python ppg_analysis_pipeline.py --case-id <id>

# Or specify manually
python ppg_analysis_pipeline.py --case-id <id> --sampling-rate 500
```

### Error: "Low validity rate"

**Cause:** Many detected pulses failed quality checks.

**Solution:** Lower the quality threshold:
```python
segmenter = PPGSegmenter(sampling_rate=500.0)
segmenter.quality_threshold = 0.3  # Default is 0.5
```

## Known Issues & Limitations

### 1. Not All Cases Have PPG Data
- **Expected:** ~60-70% of cases may not have usable PPG signals
- **Reason:** Different monitoring equipment, surgical types
- **Solution:** Use `find_valid_ppg_cases.py` to identify good cases

### 2. Varying Data Quality
- **Issue:** Signal quality varies by case and monitoring device
- **Solution:** Check quality scores in the HTML report

### 3. Different Sampling Rates
- **Issue:** Different tracks have different sampling rates
- **Solution:** Pipeline auto-detects or specify with `--sampling-rate`

### 4. Large Files
- **Issue:** High-frequency PPG data can be large (100+ MB)
- **Solution:** Use shorter duration segments for testing

## Quick Diagnostic Script

Run this to diagnose issues:

```python
# diagnose.py
from ppg_extractor import PPGExtractor
from vitaldb_utility import VitalDBUtility

print("VitalDB PPG Diagnostic")
print("=" * 50)

# Test 1: VitalDB access
print("\n[1] Testing VitalDB API access...")
try:
    util = VitalDBUtility()
    cases = util.get_all_case_ids()
    print(f"  ✓ Connected. {len(cases)} cases available")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 2: Find PPG cases
print("\n[2] Finding cases with PPG tracks...")
try:
    extractor = PPGExtractor()
    ppg_cases = extractor.find_cases_with_ppg()
    print(f"  ✓ Found {len(ppg_cases)} cases with PPG tracks")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 3: Check first few cases for valid data
print("\n[3] Checking for valid PPG data...")
valid_found = 0
for case_id in ppg_cases[:20]:  # Test first 20
    try:
        result = extractor.extract_best_ppg_track(case_id, './test_ppg')
        print(f"  ✓ Case {case_id}: {result['track_name']} "
              f"({result['num_samples']:,} samples)")
        valid_found += 1
        if valid_found >= 3:
            break
    except:
        continue

if valid_found > 0:
    print(f"\n✓ SUCCESS! Found {valid_found} cases with valid PPG data")
    print(f"  You can analyze these cases with the pipeline")
else:
    print(f"\n✗ No valid PPG data found in tested cases")
    print(f"  Try running: python find_valid_ppg_cases.py")
```

## Getting Help

1. **Check case validity first:**
   ```bash
   python find_valid_ppg_cases.py --max-valid 5
   ```

2. **Use automatic track selection:**
   ```bash
   python ppg_analysis_pipeline.py --case-id <valid_id>
   ```

3. **Review the logs:**
   - The pipeline prints detailed information about what it's trying
   - Look for "Valid samples after cleaning" count
   - Check which tracks are being tested

4. **Start with known good cases:**
   - Use the case finder to build a list of validated cases
   - Save the results for future reference

## Best Practices

✅ **Do:**
- Use `find_valid_ppg_cases.py` to identify good cases first
- Let the pipeline auto-select the best track
- Check the HTML report for data quality
- Start with shorter time segments for testing

❌ **Don't:**
- Assume all cases have PPG data
- Manually specify tracks without checking availability
- Skip data validation
- Process large batches without testing first

## Performance Tips

1. **For batch processing:**
   - First run case finder to get valid cases list
   - Then process only the valid cases
   - Use parallel processing for multiple cases

2. **For large datasets:**
   - Extract and validate first
   - Process segmentation separately
   - Generate HTML reports in batches

3. **For testing:**
   - Use `end_time=30` to test with first 30 seconds
   - Check one case completely before batch processing
   - Validate sampling rate detection

---

**Still having issues?** Check:
- [PPG_README.md](PPG_README.md) for detailed documentation
- [PPG_QUICK_START.md](PPG_QUICK_START.md) for basic usage
- Python error messages for specific issues
