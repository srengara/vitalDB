# Quick Start Guide - VitalDB PPG Web Interface

## Start the Web Application

Open a terminal/command prompt in the `c:\IITM\vitalDB` directory and run:

```bash
python web_app.py
```

You should see output like:
```
======================================================================
VitalDB PPG Analysis Web Interface
======================================================================

Starting server...
Open your browser and navigate to: http://localhost:5000

Press Ctrl+C to stop the server
======================================================================
 * Serving Flask app 'web_app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.x:5000
```

## Access the Web Interface

Open your web browser and go to:
```
http://localhost:5000
```

## Step-by-Step Usage

### Step 1: Select Case and Track

1. **Enter Case ID**: Type a VitalDB case ID (e.g., `1` or `2`)
2. **Click "Load Available Tracks"**: Wait for tracks to load
3. **Select Track**: Choose a PPG track from dropdown (e.g., "SNUADC/PLETH (500 Hz)")
4. **Click "Download Data"**: Wait for data download (may take 10-20 seconds)

### Step 2: View Raw Data

You'll see:
- **Statistics**: Total rows, NaN counts, sampling rate, duration
- **Plot**: First 30 seconds of the signal
- **Data Table**: Preview of first 100 rows
- **Download Link**: Download raw CSV

If data has NaN values, you'll see a warning message and a "Cleanse Data" button.

### Step 3: Cleanse Data (if needed)

1. **Click "Cleanse Data"**: Processes NaN values
2. View cleansing results:
   - Before/after row counts
   - Cleansed signal plot
   - Data preview
3. **Download cleansed CSV** if needed

### Step 4: Detect Peaks

1. **Click "Detect Peaks"**: Runs peak detection algorithm
2. View results:
   - **Total peaks detected**
   - **Mean heart rate** (bpm)
   - **Mean interval** between peaks
3. See two plots:
   - Original signal with peaks marked in red
   - Preprocessed signal with peaks marked in red
4. View peaks data table
5. **Download peaks CSV** for further analysis

## Example Test Cases

### Case 1 - Has NaN time values
```
Case ID: 1
Track: SNUADC/PLETH (500 Hz)
Expected: ~5.7M samples, requires cleansing
```

### Case 2 - Has NaN time values
```
Case ID: 2
Track: SNUADC/PLETH (500 Hz)
Expected: ~7.8M samples, requires cleansing
```

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the Flask server.

## Troubleshooting

### "Port 5000 already in use"
Another application is using port 5000. Either:
1. Stop the other application
2. Or edit `web_app.py` and change the port:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

### "Module not found" errors
Install required packages:
```bash
pip install -r requirements.txt
```

### Web page not loading
1. Check if server is running (terminal should show "Running on...")
2. Try `http://127.0.0.1:5000` instead of `localhost`
3. Check firewall settings

### "Session not found" errors
Restart the server - sessions are stored in memory.

## Data Storage

Downloaded data is stored in:
```
c:\IITM\vitalDB\web_app_data\
```

Each case/track combination gets its own folder with:
- Raw CSV
- Cleansed CSV
- Peaks CSV
- Metadata JSON

## Tips

- **Use valid case IDs**: Start with 1, 2, 100, etc.
- **Wait for completion**: Data downloads can take 10-20 seconds
- **Check statistics**: Look at NaN counts to know if cleansing is needed
- **Download CSVs**: Save data for later analysis
- **Try different tracks**: Some tracks may have better data quality

---

**Ready to start? Run:** `python web_app.py`
