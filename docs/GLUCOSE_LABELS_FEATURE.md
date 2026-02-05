# Glucose Labels Feature - Web App Enhancement

## Overview

Added a new **Step 5: Glucose Labels** to the VitalDB PPG Analysis web application. This feature allows users to input glucose measurements corresponding to filtered PPG windows and download the `glucose_labels.csv` file needed for training the ResNet34-1D glucose prediction model.

---

## What Was Changed

### 1. Backend (web_app.py)

#### New Route: `/api/save_glucose_labels`
**Purpose**: Save glucose measurements for each filtered PPG window

**Request Format**:
```json
POST /api/save_glucose_labels
{
  "session_id": "abc123",
  "glucose_labels": [120.5, 115.3, 130.2, ...]
}
```

**Response Format**:
```json
{
  "success": true,
  "message": "Glucose labels saved successfully",
  "num_labels": 4605,
  "file_path": "web_app_data/.../glucose_labels.csv",
  "stats": {
    "mean": 125.4,
    "std": 18.2,
    "min": 70.0,
    "max": 180.0
  }
}
```

**File Created**: `<case_id>_<track>_glucose_labels.csv`

**Format**:
```csv
window_index,glucose_mg_dl
0,120.5
1,115.3
2,130.2
...
```

#### Updated Route: `/api/download_csv/<session_id>/<file_type>`
**Added Support For**: `glucose_labels` file type

**Usage**:
```
GET /api/download_csv/<session_id>/glucose_labels
```

Downloads the glucose labels CSV file.

---

### 2. Frontend (templates/index.html)

#### New Step 5 Section
Added a complete UI section for glucose label input:

**Components**:
1. **Info Box**: Explains the purpose of glucose labels
2. **Window Count Display**: Shows how many glucose values are needed
3. **Text Area Input**: For entering glucose values (newline or comma-separated)
4. **Save Button**: Validates and saves glucose labels
5. **Generate Random Button**: Creates random test data (70-180 mg/dL)
6. **Statistics Display**: Shows mean, std, min, max after saving
7. **Download Link**: Direct download of glucose_labels.csv

#### New JavaScript Functions

**`generateRandomGlucose()`**
- Generates random glucose values for testing
- Range: 70-180 mg/dL (normal to elevated)
- Populates text area automatically

**`saveGlucoseLabels()`**
- Parses glucose values from text input
- Validates count matches number of windows
- Validates numeric values
- Warns if values outside normal range (40-400 mg/dL)
- Sends to backend API
- Displays statistics and download link

**Integration**:
- Step 5 is automatically enabled after Step 4 (peak detection) completes
- Window count is automatically populated from filtered windows

---

## How to Use

### Step-by-Step Workflow

1. **Complete Steps 1-4** (Extract → Cleanse → Detect Peaks)
   - Step 4 must complete successfully to enable Step 5
   - Note the number of filtered windows

2. **Enter Glucose Labels** (Step 5)
   - Enter glucose values corresponding to each window
   - Format options:
     - One value per line
     - Comma-separated
     - Mix of both

3. **Choose Input Method**:

   **Option A: Manual Entry**
   ```
   120.5
   115.3
   130.2
   125.8
   ...
   ```

   **Option B: Comma-Separated**
   ```
   120.5, 115.3, 130.2, 125.8, ...
   ```

   **Option C: Generate Random (for testing)**
   - Click "Generate Random (for testing)"
   - Creates random values between 70-180 mg/dL

4. **Save and Download**
   - Click "Save Glucose Labels"
   - View statistics (mean, std, min, max)
   - Click download link to get CSV file

---

## File Output

### glucose_labels.csv Format

```csv
window_index,glucose_mg_dl
0,120.5
1,115.3
2,130.2
3,125.8
4,118.7
...
```

**Columns**:
- `window_index`: Index of the filtered PPG window (0-based)
- `glucose_mg_dl`: Glucose measurement in mg/dL

### Example File Location

```
web_app_data/
└── case_2_SNUADC_PLETH/
    ├── case_2_SNUADC_PLETH_raw.csv
    ├── case_2_SNUADC_PLETH_raw_cleansed.csv
    ├── case_2_SNUADC_PLETH_raw_cleansed_peaks.csv
    ├── case_2_SNUADC_PLETH_raw_cleansed_filtered_windows.csv
    ├── case_2_SNUADC_PLETH_raw_cleansed_filtered_windows_detailed.csv
    └── case_2_SNUADC_PLETH_raw_cleansed_glucose_labels.csv  ← NEW
```

---

## Training Data Preparation

### Required Files for Training

After using the web app, you'll have two files needed for training:

1. **ppg_windows.csv** (from `filtered_windows_detailed.csv`)
   ```csv
   window_index,sample_index,amplitude
   0,0,100.5
   0,1,101.2
   ...
   ```

2. **glucose_labels.csv** (from this feature)
   ```csv
   window_index,glucose_mg_dl
   0,120.5
   1,115.3
   ...
   ```

### Prepare for Training

**Option 1: Use glucose_from_csv.py** (already handles filtered_windows_detailed.csv)

**Option 2: Prepare training data**:
```bash
# Copy and rename files
mkdir training_data
cp case_X_filtered_windows_detailed.csv training_data/ppg_windows.csv
cp case_X_glucose_labels.csv training_data/glucose_labels.csv

# Train model
python train_glucose_predictor.py --data_dir ./training_data
```

---

## Validation and Error Handling

### Input Validation

**Count Validation**:
- Number of glucose values must exactly match number of filtered windows
- Error message shows expected vs actual count

**Value Validation**:
- All values must be numeric (floats)
- Non-numeric values are rejected with error message

**Range Warning**:
- Values outside 40-400 mg/dL trigger a warning
- User can choose to proceed or cancel
- Typical normal range: 70-100 mg/dL (fasting)
- Typical elevated range: 100-180 mg/dL (post-meal or diabetic)

### Error Messages

**No Filtered Windows**:
```
Error: No filtered windows available. Run peak detection first.
```

**Count Mismatch**:
```
Error: Expected 4605 values, but got 100
```

**Invalid Values**:
```
Error: Invalid glucose values. Please enter numbers only.
```

**Session Not Found**:
```
Error: Invalid session
```

---

## API Endpoints Summary

### POST /api/save_glucose_labels

**Request**:
```json
{
  "session_id": "abc123",
  "glucose_labels": [120.5, 115.3, 130.2, ...]
}
```

**Success Response** (200):
```json
{
  "success": true,
  "message": "Glucose labels saved successfully",
  "num_labels": 4605,
  "file_path": "web_app_data/.../glucose_labels.csv",
  "stats": {
    "mean": 125.4,
    "std": 18.2,
    "min": 70.0,
    "max": 180.0
  }
}
```

**Error Response** (400):
```json
{
  "error": "Number of glucose labels (100) must match number of filtered windows (4605)"
}
```

### GET /api/download_csv/<session_id>/glucose_labels

**Response**: CSV file download

**Filename**: `<case_id>_<track>_glucose_labels.csv`

---

## UI Components

### Step 5 Layout

```
┌─────────────────────────────────────────────────────────┐
│ [5] Glucose Labels (for Training)                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ ⓘ Training Data Preparation:                          │
│   Enter glucose measurements (in mg/dL) corresponding  │
│   to each filtered PPG window.                         │
│                                                         │
│ Number of filtered windows: 4605                       │
│ You need to provide glucose values (mg/dL) for each.  │
│                                                         │
│ Glucose Values (mg/dL):                                │
│ ┌─────────────────────────────────────────────────┐   │
│ │ 120.5                                           │   │
│ │ 115.3                                           │   │
│ │ 130.2                                           │   │
│ │ ...                                             │   │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
│ [Save Glucose Labels] [Generate Random (for testing)] │
│                                                         │
│ ┌─────────────────────────────────────────────────┐   │
│ │ Statistics                                      │   │
│ │ Number of Labels: 4605                          │   │
│ │ Mean: 125.4 mg/dL                              │   │
│ │ Std: 18.2 mg/dL                                │   │
│ │ Min: 70.0 mg/dL                                │   │
│ │ Max: 180.0 mg/dL                               │   │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
│ 📥 Download Glucose Labels CSV                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Example Usage Scenarios

### Scenario 1: Real Clinical Data

**Context**: You have actual glucose measurements from patients

**Steps**:
1. Run Steps 1-4 to get filtered PPG windows
2. Match glucose measurements to window timestamps
3. Enter real glucose values in Step 5
4. Save and download for training

**Use Case**: Production model training with real data

### Scenario 2: Testing the Pipeline

**Context**: You want to test the complete workflow

**Steps**:
1. Run Steps 1-4 with any VitalDB case
2. Click "Generate Random (for testing)"
3. Save glucose labels
4. Download both filtered_windows_detailed.csv and glucose_labels.csv
5. Test training script

**Use Case**: Development, testing, demonstrations

### Scenario 3: Multiple Cases for Large Dataset

**Context**: Preparing training data from multiple patients

**Steps**:
1. Process Case 1 → Download glucose_labels.csv as `case1_glucose_labels.csv`
2. Process Case 2 → Download glucose_labels.csv as `case2_glucose_labels.csv`
3. Process Case N → Download glucose_labels.csv as `caseN_glucose_labels.csv`
4. Concatenate all files for training:
   ```bash
   # Combine all glucose labels
   cat case*_glucose_labels.csv | grep -v "window_index" > combined_glucose_labels.csv
   # Add header
   echo "window_index,glucose_mg_dl" | cat - combined_glucose_labels.csv > training_data/glucose_labels.csv
   ```

**Use Case**: Large-scale training dataset preparation

---

## Integration with Training Pipeline

### Complete Workflow

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Web App (Data Preparation)                             │
│  ┌────────────────────────────────────────────┐         │
│  │ Step 1: Select Case & Track                │         │
│  │ Step 2: View Raw Data                      │         │
│  │ Step 3: View Cleansed Data                 │         │
│  │ Step 4: Peak Detection & Filtering         │         │
│  │ Step 5: Enter Glucose Labels (NEW)         │         │
│  └────────────────────────────────────────────┘         │
│           ↓                        ↓                     │
│    filtered_windows.csv    glucose_labels.csv           │
│                                                          │
└──────────────────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Training (train_glucose_predictor.py)                  │
│  ┌────────────────────────────────────────────┐         │
│  │ Load PPG windows and glucose labels        │         │
│  │ Create PyTorch datasets                    │         │
│  │ Train ResNet34-1D model                    │         │
│  │ Save trained model                         │         │
│  └────────────────────────────────────────────┘         │
│           ↓                                              │
│    best_model.pth                                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Inference (glucose_from_csv.py)                        │
│  ┌────────────────────────────────────────────┐         │
│  │ Load trained model                         │         │
│  │ Load new filtered windows                  │         │
│  │ Predict glucose values                     │         │
│  └────────────────────────────────────────────┘         │
│           ↓                                              │
│    glucose_predictions.csv                              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Testing

### Manual Test

1. **Start web app**:
   ```bash
   cd /c/IITM/vitalDB
   python web_app.py
   ```

2. **Open browser**: http://localhost:5000

3. **Run through steps**:
   - Step 1: Enter case ID (e.g., 2) and select track
   - Step 2: View raw data
   - Step 3: Cleanse data
   - Step 4: Detect peaks (use default thresholds or adjust)
   - Step 5: Click "Generate Random" then "Save Glucose Labels"

4. **Verify files created** in `web_app_data/<case>_<track>/`:
   - Check `glucose_labels.csv` exists
   - Verify CSV format is correct
   - Check number of rows matches filtered windows

### Automated Test

```python
import requests

# Assuming web app is running on localhost:5000
BASE_URL = "http://localhost:5000"

# Assume session_id and glucose_labels are available
session_id = "your_session_id"
num_windows = 100
glucose_labels = [120.0 + i * 0.1 for i in range(num_windows)]

# Test save glucose labels
response = requests.post(
    f"{BASE_URL}/api/save_glucose_labels",
    json={
        "session_id": session_id,
        "glucose_labels": glucose_labels
    }
)

assert response.status_code == 200
data = response.json()
assert data["success"] == True
assert data["num_labels"] == num_windows

# Test download
response = requests.get(f"{BASE_URL}/api/download_csv/{session_id}/glucose_labels")
assert response.status_code == 200
assert "glucose_labels.csv" in response.headers["Content-Disposition"]
```

---

## Benefits

1. **Complete Training Pipeline**: Now users can generate both required files (PPG windows + glucose labels) from the web UI

2. **User-Friendly**: Simple textarea input with clear instructions and validation

3. **Flexible Input**: Supports multiple formats (newline, comma-separated, or mixed)

4. **Validation**: Prevents common errors (wrong count, non-numeric values, unrealistic ranges)

5. **Testing Support**: Random generation allows testing without real glucose data

6. **Statistics**: Immediate feedback on glucose distribution (mean, std, min, max)

7. **One-Click Download**: Direct download link for training data

8. **Session Management**: All files from one session are kept together

---

## Future Enhancements

### Possible Improvements

1. **Bulk Upload**: Allow uploading glucose values from CSV file

2. **Glucose Timeline Plot**: Visualize glucose values over time

3. **Clarke Error Grid**: If predictions are available, show clinical accuracy grid

4. **Auto-match Timestamps**: If glucose measurements have timestamps, auto-align with PPG windows

5. **Multiple Glucose Sources**: Support CGM data, manual measurements, lab results

6. **Data Validation**: More sophisticated range checks based on clinical guidelines

7. **Export to Training Format**: One-click export of both files in correct format for training script

8. **Real-time Prediction**: After model is trained, allow upload of model and predict on current data

---

## Files Modified

1. **web_app.py** (lines 528-583, 547-548)
   - Added `/api/save_glucose_labels` route
   - Updated `/api/download_csv` to support `glucose_labels` file type

2. **templates/index.html** (lines 412-446, 789-913)
   - Added Step 5 HTML section
   - Added JavaScript functions: `generateRandomGlucose()`, `saveGlucoseLabels()`
   - Integrated Step 5 activation in peak detection completion

---

## Summary

This feature completes the data preparation pipeline by allowing users to create the `glucose_labels.csv` file directly from the web interface. Combined with the filtered PPG windows from Step 4, users now have everything needed to train the ResNet34-1D glucose prediction model using `train_glucose_predictor.py`.

**Key Achievement**: End-to-end workflow from raw VitalDB data to training-ready datasets, all through an intuitive web interface.
