# PPG Pipeline - Batch Processing Guide

## Overview

The PPG analysis pipeline now supports batch processing of multiple cases with interactive track selection.

## Modes

### 1. Single Case Mode

Analyze one case at a time:

```bash
python ppg_analysis_pipeline.py --case-id 1
```

### 2. Batch Mode

Analyze multiple cases in a range:

```bash
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10
```

## Interactive Track Selection

When running in batch mode without specifying a track, the pipeline will:

1. **Analyze available tracks** across your case range (checks first 10 cases)
2. **Display availability statistics** for each track type
3. **Prompt you to choose** which track to use for all cases

### Example Interactive Session

```
======================================================================
PPG TRACK SELECTION
======================================================================

Analyzing PPG tracks across 10 case(s)...

Available PPG tracks (checked 10 cases):
----------------------------------------------------------------------
1. SNUADC/PLETH
   Available in: 8/10 cases (80%)
   Expected rate: 500 Hz
   Example cases: [1, 2, 3, 4, 5]

2. Solar8000/PLETH
   Available in: 10/10 cases (100%)
   Expected rate: 62.5 Hz
   Example cases: [1, 2, 3, 4, 5]

3. Primus/PLETH
   Available in: 5/10 cases (50%)
   Expected rate: 100 Hz
   Example cases: [2, 4, 6, 8, 10]

======================================================================
Select PPG track to use:

  0 - Auto-select best available track for each case (RECOMMENDED)
  1 - SNUADC/PLETH
  2 - Solar8000/PLETH
  3 - Primus/PLETH

Enter your choice (0-3):
```

**Choose:**
- **0** (Recommended): Let the pipeline automatically find the best track for each case
- **1-3**: Use that specific track for all cases (may fail if not available)

## Command Reference

### Single Case Examples

```bash
# Basic single case
python ppg_analysis_pipeline.py --case-id 100

# With specific track
python ppg_analysis_pipeline.py --case-id 100 --track "SNUADC/PLETH"

# List available tracks first
python ppg_analysis_pipeline.py --case-id 100 --list-tracks

# Custom output directory
python ppg_analysis_pipeline.py --case-id 100 --output ./my_analysis

# Skip plots for speed
python ppg_analysis_pipeline.py --case-id 100 --skip-plots
```

### Batch Processing Examples

```bash
# Interactive batch (will prompt for track selection)
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10

# Non-interactive with auto-selection
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10 --non-interactive

# Batch with specific track (no prompt)
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10 \
    --track "SNUADC/PLETH" --non-interactive

# Fast batch (skip plots)
python ppg_analysis_pipeline.py --start-case-id 100 --end-case-id 200 \
    --skip-plots --non-interactive

# Custom output directory
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10 \
    --output ./batch_results

# Specific sampling rate
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10 \
    --sampling-rate 500
```

## Command-Line Options

### Required (one of):
- `--case-id <id>` - Single case mode
- `--start-case-id <id>` - Batch mode starting case

### Batch mode required:
- `--end-case-id <id>` - Ending case (inclusive)

### Optional:
- `--track <name>` - Specific PPG track (default: auto-select)
- `--sampling-rate <hz>` - Sampling rate in Hz (default: auto-detect)
- `--output <dir>` - Output directory
- `--skip-plots` - Skip plot generation for faster processing
- `--non-interactive` - No prompts (batch mode only)
- `--list-tracks` - List tracks and exit (single mode only)

## Output Structure

### Single Case Output

```
ppg_analysis/
├── case_100_SNUADC_PLETH.csv
├── case_100_SNUADC_PLETH_metadata.json
├── case_100_initial_plot.png
├── case_100_segmentation.json
├── case_100_segmentation_pulses.csv
├── case_100_segment_1.png
├── case_100_segment_2.png
├── case_100_segment_3.png
└── case_100_report.html
```

### Batch Output

```
ppg_batch_analysis/
├── batch_results.json          # Summary of all cases
├── case_1/
│   ├── case_1_*.csv
│   ├── case_1_*.json
│   └── case_1_report.html
├── case_2/
│   └── ...
├── case_3/
│   └── ...
└── case_10/
    └── ...
```

### Batch Results JSON

```json
{
  "total": 10,
  "successful": 7,
  "failed": 3,
  "track_used": null,
  "successful": [
    {
      "case_id": 1,
      "output_dir": "./ppg_batch_analysis/case_1"
    },
    ...
  ],
  "failed": [
    {
      "case_id": 2,
      "reason": "No valid PPG data found in any available track"
    },
    ...
  ]
}
```

## Performance Tips

### 1. Use --skip-plots for Speed

Plot generation takes time. For large batches:

```bash
# Much faster without plots
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 100 --skip-plots
```

You can still view the data in the HTML reports and CSV files.

### 2. Use --non-interactive for Automation

For scripts and automated processing:

```bash
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 100 \
    --non-interactive --skip-plots
```

### 3. Process in Chunks

For very large ranges, process in chunks:

```bash
# Process 100-199
python ppg_analysis_pipeline.py --start-case-id 100 --end-case-id 199 --non-interactive

# Then 200-299
python ppg_analysis_pipeline.py --start-case-id 200 --end-case-id 299 --non-interactive
```

### 4. Find Valid Cases First

Use the case finder to identify good cases, then batch process only those:

```bash
# Step 1: Find valid cases
python find_valid_ppg_cases.py --start 1 --num-cases 500 --max-valid 50

# Step 2: Check valid_ppg_cases.json for case IDs
# Step 3: Process those specific ranges
```

## Track Selection Strategies

### Option 1: Auto-Select (Recommended)

```bash
# Each case gets its best available track
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10
# Choose: 0 (Auto-select)
```

**Pros:**
- ✓ Maximizes success rate
- ✓ Handles cases with different available tracks
- ✓ Uses highest quality track available

**Cons:**
- Different cases may use different tracks
- Sampling rates may vary

### Option 2: Specific Track

```bash
# All cases use SNUADC/PLETH
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10 \
    --track "SNUADC/PLETH" --non-interactive
```

**Pros:**
- ✓ Consistent track across all cases
- ✓ Same sampling rate
- ✓ No prompts

**Cons:**
- May fail for cases without that track
- Lower success rate

### Option 3: Interactive Selection

```bash
# See availability, then choose
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10
# Review statistics, then choose track
```

**Pros:**
- ✓ Informed decision based on availability
- ✓ Can see which track is most common
- ✓ Balance consistency vs success rate

## Real-World Examples

### Example 1: Quick Test

```bash
# Test with 5 cases, auto-select, no plots
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 5 \
    --non-interactive --skip-plots
```

### Example 2: High-Quality Dataset

```bash
# Interactive selection, all plots, specific track
python ppg_analysis_pipeline.py --start-case-id 100 --end-case-id 120

# When prompted, choose SNUADC/PLETH for highest sampling rate
```

### Example 3: Large-Scale Batch

```bash
# Process 100 cases fast
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 100 \
    --non-interactive --skip-plots --output ./large_batch

# Check results
cat ./large_batch/batch_results.json
```

### Example 4: Mixed Quality Cases

```bash
# Let auto-select handle different case types
python ppg_analysis_pipeline.py --start-case-id 500 --end-case-id 550
# Choose: 0 (Auto-select)
```

## Troubleshooting Batch Processing

### Issue: Many cases failing

**Solution:** Use auto-selection instead of specific track

```bash
# Instead of this (may fail)
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10 \
    --track "SNUADC/PLETH"

# Use this
python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10
# Choose: 0 (Auto-select)
```

### Issue: Different sampling rates in batch

**Expected:** When using auto-select, different tracks = different rates

**Solution:** If you need consistent sampling rate:
1. Choose a specific track, OR
2. Pre-filter cases using `find_valid_ppg_cases.py`

### Issue: Batch processing is slow

**Solutions:**
```bash
# 1. Skip plots
--skip-plots

# 2. Use non-interactive mode
--non-interactive

# 3. Process smaller chunks
```

### Issue: Want to resume failed cases

Check `batch_results.json` for failed case IDs, then:

```bash
# Re-run only failed cases
python ppg_analysis_pipeline.py --start-case-id 2 --end-case-id 2
python ppg_analysis_pipeline.py --start-case-id 5 --end-case-id 5
# etc.
```

## Best Practices

✅ **Do:**
- Use `--non-interactive` for automation
- Use `--skip-plots` for large batches
- Review track availability before choosing
- Check `batch_results.json` after completion
- Use auto-selection for maximum success rate

❌ **Don't:**
- Process thousands of cases without testing first
- Specify tracks without checking availability
- Ignore failed cases in results
- Skip the initial track selection prompt blindly

## Exit Codes

- **0** - Success (>50% success rate in batch)
- **1** - Failure or <50% success rate

Use in scripts:
```bash
if python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10 --non-interactive; then
    echo "Batch completed successfully"
else
    echo "Batch had high failure rate"
fi
```

---

**Need help?** Check the main [PPG_README.md](PPG_README.md) or run:
```bash
python ppg_analysis_pipeline.py --help
```
