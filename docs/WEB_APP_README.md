# VitalDB PPG Analysis Web Interface

Interactive web application for PPG data extraction, cleansing, and peak detection from VitalDB dataset.

## Features

### 4-Step Interactive Pipeline:

1. **Case & Track Selection**
   - Enter VitalDB case ID
   - Load available PPG tracks
   - Select track from dropdown (shows sampling rate)

2. **Raw Data View**
   - Download and display raw data
   - Show statistics (total rows, NaN counts, duration)
   - Plot first 30 seconds of signal
   - Preview data table (first 100 rows)
   - Download raw CSV
   - Automatic detection of data quality issues

3. **Data Cleansing**
   - Handle NaN values in time and signal columns
   - Create synthetic time values if needed (based on sampling rate)
   - Show before/after statistics
   - Plot cleansed signal
   - Preview cleansed data
   - Download cleansed CSV

4. **Peak Detection**
   - Detect systolic peaks using preprocessing and scipy peak detection
   - Display peak statistics (count, mean HR, intervals)
   - Show two plots:
     - Original signal with peaks marked
     - Preprocessed signal with peaks marked
   - Preview peaks data table
   - Download peaks CSV

## Installation

1. Install required packages:
```bash
pip install flask requests pandas numpy matplotlib scipy
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Start the Web Server:

```bash
python web_app.py
```

### Access the Application:

Open your web browser and navigate to:
```
http://localhost:5000
```

### Using the Interface:

1. **Enter Case ID** (e.g., 1, 2, 100)
2. Click **Load Available Tracks**
3. **Select a track** from the dropdown
4. Click **Download Data**
5. Review the raw data and statistics
6. If data has issues, click **Cleanse Data**
7. Click **Detect Peaks**
8. View results and download CSV files

## API Endpoints

The web app provides RESTful API endpoints:

### `POST /api/get_tracks`
Get available PPG tracks for a case.

**Request:**
```json
{
  "case_id": 1
}
```

**Response:**
```json
{
  "tracks": [
    {"name": "SNUADC/PLETH", "rate": 500},
    {"name": "Solar8000/PLETH", "rate": 62.5}
  ]
}
```

### `POST /api/download_data`
Download raw PPG data from VitalDB.

**Request:**
```json
{
  "case_id": 1,
  "track_name": "SNUADC/PLETH"
}
```

**Response:**
```json
{
  "session_id": "case_1_SNUADC_PLETH",
  "total_rows": 5770154,
  "nan_time": 5770154,
  "nan_signal": 0,
  "has_issues": true,
  "sampling_rate": 500.0,
  "duration": 11540.31,
  "preview_data": [...],
  "plot_image": "data:image/png;base64,..."
}
```

### `POST /api/cleanse_data`
Cleanse PPG data (handle NaN values).

**Request:**
```json
{
  "session_id": "case_1_SNUADC_PLETH"
}
```

**Response:**
```json
{
  "original_rows": 5770575,
  "cleansed_rows": 5770154,
  "removed_rows": 421,
  "preview_data": [...],
  "plot_image": "data:image/png;base64,..."
}
```

### `POST /api/detect_peaks`
Detect peaks in cleansed PPG data.

**Request:**
```json
{
  "session_id": "case_1_SNUADC_PLETH"
}
```

**Response:**
```json
{
  "total_peaks": 12450,
  "mean_hr": 65.2,
  "std_hr": 3.5,
  "mean_interval": 0.92,
  "std_interval": 0.05,
  "preview_data": [...],
  "plot_original": "data:image/png;base64,...",
  "plot_preprocessed": "data:image/png;base64,..."
}
```

### `GET /api/download_csv/<session_id>/<file_type>`
Download CSV file.

**Parameters:**
- `session_id`: Session identifier
- `file_type`: `raw`, `cleansed`, or `peaks`

## Data Storage

All session data is stored in `./web_app_data/` directory:

```
web_app_data/
├── case_1_SNUADC_PLETH/
│   ├── case_1_SNUADC_PLETH.csv              # Raw data
│   ├── case_1_SNUADC_PLETH_metadata.json     # Metadata
│   ├── case_1_SNUADC_PLETH_cleansed.csv     # Cleansed data
│   └── case_1_SNUADC_PLETH_cleansed_peaks.csv  # Peaks
└── case_2_Solar8000_PLETH/
    └── ...
```

## Architecture

### Backend (Flask):
- **web_app.py** - Main Flask application
- **ppg_extractor.py** - VitalDB data extraction (reused)
- **ppg_segmentation.py** - Preprocessing and peak detection (reused)

### Frontend (HTML/CSS/JavaScript):
- **templates/index.html** - Single-page application
- Pure JavaScript (no frameworks)
- RESTful API calls
- Responsive design with gradient UI

### Key Features:
- **Session Management**: Each case/track combination gets a unique session
- **In-memory Cache**: Fast access to session data
- **Progressive Pipeline**: Steps are enabled as prerequisites complete
- **Visual Feedback**: Real-time status updates and loading indicators
- **Data Preview**: First 100 rows shown in tables
- **Plot Generation**: Matplotlib plots converted to base64 images
- **Download Support**: Export CSV files at each stage

## Code Organization

The web app reuses existing project code:

```
vitalDB/
├── web_app.py                    # NEW: Flask web server
├── templates/
│   └── index.html               # NEW: Web interface
├── web_app_data/                # NEW: Session storage
├── ppg_extractor.py             # REUSED: Data extraction
├── ppg_segmentation.py          # REUSED: Peak detection
├── ppg_peak_detection_pipeline.py  # Command-line version
├── ppg_analysis_pipeline.py     # Full analysis pipeline
└── requirements.txt             # UPDATED: Added flask
```

## Troubleshooting

### Port Already in Use:
```bash
# Change port in web_app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Session Not Found:
- Restart the server to clear session cache
- Session data is in-memory, not persistent

### Large Files:
- Plots show only first 30 seconds for performance
- Data tables show only first 100 rows
- Full data available in CSV downloads

## Example Workflow

1. Start server: `python web_app.py`
2. Open browser: `http://localhost:5000`
3. Enter case ID: `1`
4. Click "Load Available Tracks"
5. Select: "SNUADC/PLETH (500 Hz)"
6. Click "Download Data"
7. See warning about NaN values
8. Click "Cleanse Data"
9. Review cleansed statistics
10. Click "Detect Peaks"
11. See 12,450 peaks detected with HR 65 bpm
12. Download peaks CSV for further analysis

## Performance

- **Data Download**: ~5-10 seconds (depending on file size)
- **Cleansing**: ~1-2 seconds
- **Peak Detection**: ~3-5 seconds
- **Plot Generation**: ~1 second per plot

## Browser Compatibility

Tested on:
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

## Security Notes

- This is a local development server
- **Do not expose to internet** without proper security measures
- Session data stored in plain text
- No authentication/authorization implemented

## Future Enhancements

Potential additions:
- [ ] Persistent session storage (SQLite/Redis)
- [ ] User authentication
- [ ] Multiple case comparison
- [ ] Export to PDF reports
- [ ] Real-time progress bars for long operations
- [ ] Configurable peak detection parameters
- [ ] Batch processing interface
- [ ] WebSocket for real-time updates

---

**Developed for VitalDB PPG Analysis Project**
