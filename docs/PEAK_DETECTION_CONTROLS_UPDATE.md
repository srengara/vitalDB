# Peak Detection Controls Update

## Summary

Updated the VitalDB PPG Analysis web application to give users control over peak detection parameters through the UI.

## Changes Made

### 1. Frontend (templates/index.html)

#### Added Input Controls (Step 3)
Two new input fields added before the "Detect Peaks" button:

1. **Height Threshold Multiplier**
   - Default value: 0.3
   - Range: 0 to 2
   - Step: 0.1
   - Formula: `threshold = mean + multiplier × std`

2. **Distance Threshold Multiplier**
   - Default value: 0.8
   - Range: 0.1 to 2
   - Step: 0.1
   - Formula: `min_distance = multiplier × sampling_rate`

#### Updated JavaScript (detectPeaks function)
- Reads values from the input fields
- Sends parameters to backend via POST request:
  ```javascript
  {
    session_id: sessionId,
    height_multiplier: heightMultiplier,
    distance_multiplier: distanceMultiplier
  }
  ```

#### Updated Results Display (Step 4)
Added two new stat cards to show the actual thresholds used:
- **Height Threshold**: Shows calculated threshold value
- **Distance Threshold**: Shows calculated distance in samples

### 2. Backend (web_app.py)

#### Updated `/api/detect_peaks` Endpoint

**Input Parameters:**
- `session_id` (required): Session identifier
- `height_multiplier` (optional, default: 0.3): Multiplier for height threshold
- `distance_multiplier` (optional, default: 0.8): Multiplier for distance threshold

**Threshold Calculations:**
```python
# Height threshold
signal_mean = np.mean(preprocessed_signal)
signal_std = np.std(preprocessed_signal)
height_threshold = signal_mean + float(height_multiplier) * signal_std

# Distance threshold
distance_threshold = float(distance_multiplier) * float(sampling_rate)
```

**Response (added fields):**
```json
{
  "total_peaks": 12450,
  "mean_hr": 65.2,
  "std_hr": 3.5,
  "mean_interval": 0.92,
  "std_interval": 0.05,
  "height_threshold": 45.67,           // NEW
  "distance_threshold": 400.0,         // NEW
  "height_multiplier": 0.3,            // NEW
  "distance_multiplier": 0.8,          // NEW
  "preview_data": [...],
  "plot_original": "data:image/png;base64,...",
  "plot_preprocessed": "data:image/png;base64,..."
}
```

## Usage

### Starting the Web Application
```bash
python web_app.py
```

Then open: http://localhost:5000

### Using the Controls

1. Complete Steps 1-3 (Download data → Cleanse data)
2. Before clicking "Detect Peaks", adjust the threshold multipliers:
   - **Increase height multiplier** (e.g., 0.5, 0.7) → Detect only higher peaks (fewer peaks)
   - **Decrease height multiplier** (e.g., 0.1, 0.2) → Detect lower peaks (more peaks)
   - **Increase distance multiplier** (e.g., 1.0, 1.5) → Require more space between peaks (fewer peaks)
   - **Decrease distance multiplier** (e.g., 0.5, 0.6) → Allow closer peaks (more peaks)
3. Click "Detect Peaks"
4. View results with the threshold values displayed

### Examples

**Conservative Detection (fewer peaks):**
- Height Multiplier: 0.5
- Distance Multiplier: 1.2

**Aggressive Detection (more peaks):**
- Height Multiplier: 0.1
- Distance Multiplier: 0.5

**Default (balanced):**
- Height Multiplier: 0.3
- Distance Multiplier: 0.8

## Technical Details

### Peak Detection Algorithm (peak_detection.py)

The algorithm uses two main constraints:

1. **Height Constraint**: A peak must exceed `mean + multiplier × std`
   - Higher multiplier = Only prominent peaks detected
   - Lower multiplier = More subtle peaks detected

2. **Distance Constraint**: Peaks must be separated by at least `multiplier × sampling_rate` samples
   - For 500 Hz sampling rate and 0.8 multiplier: minimum 400 samples apart
   - This prevents detecting multiple peaks within a single heartbeat

### Integration with Preprocessing

The thresholds are applied to the **preprocessed signal**, which includes:
1. DC component removal
2. Bandpass filtering (0.5-10 Hz)
3. Savitzky-Golay smoothing

This ensures cleaner peak detection compared to raw signal.

## Benefits

1. **User Control**: Users can tune detection sensitivity for different signal qualities
2. **Transparency**: Actual threshold values are displayed in results
3. **Flexibility**: Can adjust parameters without restarting the application
4. **Repeatability**: Same multipliers can be used across different cases for consistency
5. **Experimentation**: Easy to compare results with different parameter settings

## Notes

- Default values (0.3 and 0.8) work well for most VitalDB PPG signals
- For noisy signals, increase height multiplier to avoid false peaks
- For signals with irregular rhythms, adjust distance multiplier accordingly
- All parameters are saved in session for potential future reference
