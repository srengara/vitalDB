"""
Example: Glucose Prediction from PPG Signals using ResNet34-1D

This script demonstrates the complete pipeline:
1. Load PPG data from VitalDB
2. Extract and cleanse PPG signal
3. Detect peaks and filter windows
4. Predict glucose values using ResNet34-1D

Usage:
    python example_glucose_prediction.py
"""

import numpy as np
import pandas as pd
from ppg_extractor import PPGExtractor
from ppg_segmentation import PPGSegmenter
from peak_detection import ppg_peak_detection_pipeline_with_template
from resnet34_glucose_predictor import GlucosePredictor


def main():
    print("=" * 80)
    print("PPG Signal to Glucose Prediction Pipeline")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Extract PPG Data from VitalDB
    # =========================================================================
    print("\n[STEP 1] Extracting PPG data from VitalDB...")

    case_id = 2  # Example case ID
    track_name = 'SNUADC/ART'  # Arterial blood pressure waveform
    output_dir = './example_glucose_output'

    extractor = PPGExtractor()

    try:
        result = extractor.extract_ppg_cleansed(case_id, track_name, output_dir)
        print(f"[SUCCESS] Extracted {result['total_samples']} samples")
        print(f"   Sampling Rate: {result['estimated_sampling_rate']} Hz")
        print(f"   Duration: {result['duration_seconds']:.2f} seconds")

        # Load cleansed data
        df = pd.read_csv(result['csv_file'])
        time = df['time'].values
        signal = df['ppg'].values
        sampling_rate = result['estimated_sampling_rate']

    except Exception as e:
        print(f"[ERROR] Error extracting data: {e}")
        print("\n[INFO] Using synthetic PPG signal for demonstration...")

        # Generate synthetic PPG signal
        sampling_rate = 500
        duration = 30  # 30 seconds
        t = np.linspace(0, duration, int(sampling_rate * duration))

        # Simulate PPG: combination of sine waves
        heart_rate = 75  # bpm
        frequency = heart_rate / 60  # Hz

        signal = (
            100 * np.sin(2 * np.pi * frequency * t) +  # Fundamental
            30 * np.sin(4 * np.pi * frequency * t) +   # Harmonic
            10 * np.random.randn(len(t))               # Noise
        )
        signal = signal + 200  # Offset
        time = t

        print(f"[SUCCESS] Generated synthetic signal: {len(signal)} samples")
        print(f"   Sampling Rate: {sampling_rate} Hz")
        print(f"   Duration: {duration} seconds")

    # =========================================================================
    # STEP 2: Preprocess Signal
    # =========================================================================
    print("\n[STEP 2] Preprocessing PPG signal...")

    segmenter = PPGSegmenter(sampling_rate=sampling_rate)
    preprocessed_signal = segmenter.preprocess_signal(signal)

    print(f"[SUCCESS] Signal preprocessed")
    print(f"   DC removed: Yes")
    print(f"   Bandpass filtered (0.5-10 Hz): Yes")
    print(f"   Savitzky-Golay smoothed: Yes")

    # =========================================================================
    # STEP 3: Peak Detection and Window Filtering
    # =========================================================================
    print("\n[STEP 3] Detecting peaks and extracting windows...")

    # Calculate adaptive thresholds
    signal_mean = np.mean(preprocessed_signal)
    signal_std = np.std(preprocessed_signal)
    height_threshold = signal_mean + 0.3 * signal_std
    distance_threshold = 0.8 * sampling_rate

    # Run template-based peak detection pipeline
    peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
        ppg_signal=preprocessed_signal,
        fs=sampling_rate,
        window_duration=1.0,
        height_threshold=height_threshold,
        distance_threshold=distance_threshold,
        similarity_threshold=0.85
    )

    print(f"[SUCCESS] Peak detection completed")
    print(f"   Total peaks detected: {len(peaks)}")
    print(f"   Windows extracted: {len(all_windows)}")
    print(f"   Filtered windows (high similarity): {len(filtered_windows)}")
    print(f"   Filtering rate: {len(filtered_windows)/len(all_windows)*100:.1f}%")
    print(f"   Template length: {len(template)} samples")

    if len(filtered_windows) == 0:
        print("\n[ERROR] No windows passed filtering. Cannot proceed with glucose prediction.")
        return

    # =========================================================================
    # STEP 4: Glucose Prediction using ResNet34-1D
    # =========================================================================
    print("\n[STEP 4] Predicting glucose values using ResNet34-1D...")

    # Initialize glucose predictor
    # Note: For demonstration, we're using an untrained model
    # In production, you would load a pre-trained model
    window_length = len(filtered_windows[0]) if len(filtered_windows) > 0 else 100

    predictor = GlucosePredictor(
        input_length=window_length,
        device='cpu'  # Use 'cuda' if GPU available
    )

    # Print model summary
    print(predictor.get_model_summary())

    # Predict glucose values
    print(f"\n[INFO] Running inference on {len(filtered_windows)} windows...")

    glucose_results = predictor.predict_with_stats(filtered_windows, batch_size=32)

    # =========================================================================
    # STEP 5: Display Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("GLUCOSE PREDICTION RESULTS")
    print("=" * 80)

    print(f"\nGlucose Statistics:")
    print(f"   Mean Glucose:  {glucose_results['mean_glucose']:.2f} mg/dL")
    print(f"   Std Deviation: {glucose_results['std_glucose']:.2f} mg/dL")
    print(f"   Min Glucose:   {glucose_results['min_glucose']:.2f} mg/dL")
    print(f"   Max Glucose:   {glucose_results['max_glucose']:.2f} mg/dL")
    print(f"   Num Windows:   {glucose_results['num_windows']}")

    # Show first 10 predictions
    predictions = glucose_results['predictions']
    print(f"\nFirst 10 Glucose Predictions:")
    print(f"{'Window':<10} {'Glucose (mg/dL)':<20}")
    print("-" * 30)
    for i in range(min(10, len(predictions))):
        print(f"{i:<10} {predictions[i]:<20.2f}")

    # =========================================================================
    # STEP 6: Save Results
    # =========================================================================
    print(f"\n[STEP 6] Saving results...")

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save glucose predictions to CSV
    glucose_df = pd.DataFrame({
        'window_index': range(len(predictions)),
        'peak_index': peaks[:len(predictions)],
        'glucose_mg_dl': predictions,
        'time_seconds': [time[peak] for peak in peaks[:len(predictions)]]
    })

    glucose_csv = os.path.join(output_dir, 'glucose_predictions.csv')
    glucose_df.to_csv(glucose_csv, index=False)

    print(f"[SUCCESS] Glucose predictions saved to: {glucose_csv}")

    # =========================================================================
    # NOTES
    # =========================================================================
    print("\n" + "=" * 80)
    print("IMPORTANT NOTES:")
    print("=" * 80)
    print("""
1. The model used here is UNTRAINED. Predictions are random.

2. To use in production:
   - Train the ResNet34 model on labeled PPG-glucose paired data
   - Save the trained model: predictor.save_model('glucose_model.pth')
   - Load for inference: predictor = GlucosePredictor(model_path='glucose_model.pth')

3. Training requirements:
   - Paired dataset: PPG windows + corresponding glucose measurements
   - Typical dataset size: 10,000+ samples
   - Training time: Several hours on GPU

4. Model performance metrics:
   - MAE (Mean Absolute Error): Typical target < 15 mg/dL
   - RMSE (Root Mean Squared Error): Typical target < 20 mg/dL

5. Clinical validation required before medical use.
""")

    print("=" * 80)
    print("[SUCCESS] Pipeline completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
