#!/usr/bin/env python
"""
Multi-Channel Training Data Generator
======================================
Processes multi-channel PPG signals (Force, Signal1, Signal2, Signal3) with glucose values in filename.

Key Differences from VitalDB Pipeline (d7.py):
- No labs_data.csv loading (glucose value extracted from filename)
- No 16-minute window cutting (processes entire signal)
- Generates intermediate files for each processing step
- Supports batch processing of folder with multiple files

Input File Format:
    force-GLUC123-SYS140-DIA91.csv
    Signal1-GLUC123-SYS140-DIA91.csv
    Signal2-GLUC123-SYS140-DIA91.csv
    Signal3-GLUC123-SYS140-DIA91.csv

Output Structure:
    output_folder/
        <filename>-01-raw.csv              # Raw data after loading
        <filename>-02-cleaned.csv          # After time repair and filling
        <filename>-03-downsampled.csv      # After 100Hz downsampling
        <filename>-04-preprocessed.csv     # After bandpass filtering
        <filename>-05-peaks.csv            # Detected peaks
        <filename>-06-windows.csv          # All extracted windows
        <filename>-07-filtered.csv         # Quality-filtered windows
        <filename>-08-template.csv         # Computed template
        <filename>-output.csv              # Final output (wide format)
        <filename>-metadata.json           # Processing metadata

Usage:
    # Single file
    python generate_multichannel_training_data.py --input force-GLUC123-SYS140-DIA91.csv --output ./output

    # Batch process entire folder
    python generate_multichannel_training_data.py --input_folder C:\\senzrtech\\Multi-channel\\multi-channel-input-files --output ./output
"""

import os
import sys
import re
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

# Add parent directory to path to import from senzrTech/src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.data_extraction.ppg_segmentation import PPGSegmenter
from src.data_extraction.peak_detection import ppg_peak_detection_pipeline_with_template

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_metadata_from_filename(filename: str) -> Dict:
    """
    Extract glucose, systolic BP, diastolic BP from filename.

    Expected format: <channel>-GLUC<value>-SYS<value>-DIA<value>.csv
    Example: force-GLUC123-SYS140-DIA91.csv

    Returns:
        Dict with keys: channel, glucose, systolic, diastolic
    """
    basename = os.path.basename(filename)

    # Extract channel name (everything before first hyphen)
    parts = basename.split('-')
    if len(parts) < 4:
        raise ValueError(f"Filename format invalid: {basename}. Expected: <channel>-GLUC<val>-SYS<val>-DIA<val>.csv")

    channel = parts[0]

    # Extract glucose value
    gluc_match = re.search(r'GLUC(\d+)', basename, re.IGNORECASE)
    if not gluc_match:
        raise ValueError(f"Could not extract glucose value from filename: {basename}")
    glucose = int(gluc_match.group(1))

    # Extract systolic BP (optional)
    sys_match = re.search(r'SYS(\d+)', basename, re.IGNORECASE)
    systolic = int(sys_match.group(1)) if sys_match else None

    # Extract diastolic BP (optional)
    dia_match = re.search(r'DIA(\d+)', basename, re.IGNORECASE)
    diastolic = int(dia_match.group(1)) if dia_match else None

    return {
        'channel': channel,
        'glucose': glucose,
        'systolic': systolic,
        'diastolic': diastolic,
        'original_filename': basename
    }


def load_multichannel_data(csv_file: str, expected_sampling_rate: float = 100.0) -> Tuple[pd.DataFrame, Dict]:
    """
    Load multi-channel PPG data from CSV.

    Expected columns: time, ppg (or signal)

    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    logger.info(f"Loading data from: {csv_file}")

    df = pd.read_csv(csv_file)

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Check for required columns (flexible naming)
    time_col = None
    signal_col = None

    for col in df.columns:
        if 'time' in col:
            time_col = col
        if 'ppg' in col or 'signal' in col or 'amplitude' in col:
            signal_col = col

    if time_col is None:
        raise ValueError(f"No 'time' column found in {csv_file}")
    if signal_col is None:
        raise ValueError(f"No signal column (ppg/signal/amplitude) found in {csv_file}")

    # Rename to standard names
    df = df.rename(columns={time_col: 'time', signal_col: 'ppg'})

    # Keep only time and ppg columns
    df = df[['time', 'ppg']].copy()

    # Convert to numeric
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['ppg'] = pd.to_numeric(df['ppg'], errors='coerce')

    # Extract metadata from filename
    file_metadata = extract_metadata_from_filename(csv_file)

    # Infer sampling rate if possible
    if df['time'].notna().sum() > 10:
        time_diffs = np.diff(df['time'].dropna().values)
        if len(time_diffs) > 0:
            median_diff = np.nanmedian(time_diffs)
            if median_diff > 0:
                inferred_sr = 1.0 / median_diff
                file_metadata['inferred_sampling_rate'] = float(inferred_sr)

    file_metadata['expected_sampling_rate'] = expected_sampling_rate
    file_metadata['num_samples'] = len(df)
    file_metadata['duration_seconds'] = float(df['time'].max() - df['time'].min()) if df['time'].notna().any() else 0.0

    logger.info(f"  Loaded {len(df)} samples")
    logger.info(f"  Channel: {file_metadata['channel']}")
    logger.info(f"  Glucose: {file_metadata['glucose']} mg/dL")
    if file_metadata.get('systolic'):
        logger.info(f"  Systolic BP: {file_metadata['systolic']} mmHg")
    if file_metadata.get('diastolic'):
        logger.info(f"  Diastolic BP: {file_metadata['diastolic']} mmHg")

    return df, file_metadata


def save_intermediate_file(df: pd.DataFrame, output_path: str, description: str):
    """Save intermediate processing step."""
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved {description}: {output_path}")


def repair_time_axis(df: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
    """
    STEP 1: Global Time Repair

    Reconstructs time axis based on sample index if time is missing/broken.
    Matches d7.py lines 300-318.
    """
    logger.info("Step 1: Global Time Repair")

    time_nans = df['time'].isna().sum()

    if time_nans > 0 or df['time'].isna().all():
        logger.warning(f"  Found {time_nans} missing timestamps. Reconstructing time axis...")

        # Reconstruct time: t = index / fs
        synthetic_time = np.arange(len(df)) / sampling_rate

        # Align to valid start time if available
        start_t = df['time'].min()
        if pd.isna(start_t):
            start_t = 0.0

        df['time'] = synthetic_time + start_t
        logger.info("  Time axis reconstructed successfully.")
    else:
        logger.info("  Time axis is valid. No repair needed.")

    return df


def clean_signal_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    STEP 2: Global Fill & Mask

    Tracks missing signal data and applies forward fill.
    Matches d7.py lines 320-344.
    """
    logger.info("Step 2: Signal Cleaning (Forward Fill)")

    # Create mask to track bad data
    bad_data_mask = df['ppg'].isna().astype(float).values
    initial_nans = bad_data_mask.sum()

    if initial_nans > 0:
        logger.info(f"  Found {initial_nans} NaN values ({initial_nans/len(df)*100:.2f}%)")
        logger.info("  Applying forward fill...")
        df['ppg'] = df['ppg'].ffill()

        # Handle leading NaNs (unfixable by ffill)
        if df['ppg'].isna().any():
            logger.warning("  Found leading NaNs. Trimming start of data...")
            valid_idx = df['ppg'].notna()
            df = df[valid_idx].copy()
            bad_data_mask = bad_data_mask[valid_idx]

        remaining_nans = df['ppg'].isna().sum()
        if remaining_nans > 0:
            logger.error(f"  ERROR: {remaining_nans} NaNs remain after cleaning!")
            raise ValueError("Data contains non-recoverable NaNs")

        logger.info("  Signal cleaning complete.")
    else:
        logger.info("  No missing values found.")

    return df, bad_data_mask


def downsample_to_100hz(df: pd.DataFrame, bad_data_mask: np.ndarray,
                        current_sr: float, target_sr: float = 100.0) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    STEP 3: Global Downsampling to 100Hz

    Resamples signal and time to 100Hz.
    Matches d7.py lines 346-380.
    """
    logger.info(f"Step 3: Global Downsampling ({current_sr:.1f}Hz -> {target_sr}Hz)")

    if current_sr > (target_sr + 1.0):
        num_samples = int(len(df) * (target_sr / current_sr))

        logger.info(f"  Resampling from {len(df)} to {num_samples} samples...")

        # Resample signal and time
        new_ppg, new_time = scipy_signal.resample(df['ppg'].values, num_samples, t=df['time'].values)

        # Resample bad data mask
        old_time = df['time'].values
        new_mask = np.interp(new_time, old_time, bad_data_mask)

        # Reconstruct DataFrame
        df = pd.DataFrame({'time': new_time, 'ppg': new_ppg})
        df['is_bad'] = (new_mask > 0.01)

        logger.info(f"  Downsampling complete: {len(df)} samples at {target_sr}Hz")
    else:
        logger.info("  Sampling rate already <= 100Hz. Skipping downsample.")
        df['is_bad'] = (bad_data_mask > 0.5)

    # Final forward fill (in case resampling introduced edge artifacts)
    df['ppg'] = df['ppg'].ffill()

    return df, target_sr


def preprocess_signal(df: pd.DataFrame, sampling_rate: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    STEP 4: Signal Preprocessing (Bandpass Filter)

    Uses PPGSegmenter to apply bandpass filter (0.5-8 Hz).
    Matches d7.py lines 439-441.
    """
    logger.info("Step 4: Signal Preprocessing (Bandpass Filter 0.5-8Hz)")

    segmenter = PPGSegmenter(sampling_rate=sampling_rate)
    preprocessed_signal = segmenter.preprocess_signal(df['ppg'].values)

    df_preprocessed = df.copy()
    df_preprocessed['ppg_preprocessed'] = preprocessed_signal

    logger.info("  Preprocessing complete.")

    return df_preprocessed, preprocessed_signal


def detect_peaks_and_extract_windows(preprocessed_signal: np.ndarray,
                                      sampling_rate: float,
                                      height_multiplier: float = 0.3,
                                      distance_multiplier: float = 0.8,
                                      similarity_threshold: float = 0.85) -> Tuple[List[int], List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    STEP 5: Peak Detection and Window Extraction

    Detects peaks and extracts 1-second windows.
    Matches d7.py lines 442-458.
    """
    logger.info("Step 5: Peak Detection and Window Extraction")

    # Calculate thresholds
    signal_mean = np.mean(preprocessed_signal)
    signal_std = np.std(preprocessed_signal)
    height_threshold = signal_mean + height_multiplier * signal_std
    distance_threshold = distance_multiplier * sampling_rate

    logger.info(f"  Height threshold: {height_threshold:.2f}")
    logger.info(f"  Distance threshold: {distance_threshold:.2f} samples")

    # Run peak detection pipeline with template matching
    peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
        ppg_signal=preprocessed_signal,
        fs=float(sampling_rate),
        window_duration=1.0,  # 1-second windows (100 samples at 100Hz)
        height_threshold=float(height_threshold),
        distance_threshold=distance_threshold,
        similarity_threshold=similarity_threshold
    )

    logger.info(f"  Detected {len(peaks)} peaks")
    logger.info(f"  Extracted {len(all_windows)} windows")
    logger.info(f"  Filtered to {len(filtered_windows)} high-quality windows")
    if len(all_windows) > 0:
        logger.info(f"  Filtering rate: {len(filtered_windows)/len(all_windows)*100:.1f}%")

    return peaks, filtered_windows, template, all_windows


def save_peaks_data(peaks: List[int], df_preprocessed: pd.DataFrame, output_path: str):
    """Save peak detection results."""
    peak_data = []
    for i, peak_idx in enumerate(peaks):
        if peak_idx < len(df_preprocessed):
            peak_data.append({
                'peak_number': i + 1,
                'peak_index': peak_idx,
                'peak_time': df_preprocessed.iloc[peak_idx]['time'],
                'peak_amplitude': df_preprocessed.iloc[peak_idx]['ppg_preprocessed']
            })

    if peak_data:
        pd.DataFrame(peak_data).to_csv(output_path, index=False)
        logger.info(f"  Saved peak data: {output_path}")


def save_windows_data(windows: List[np.ndarray], output_path: str, prefix: str = "window"):
    """Save window data in long format."""
    window_rows = []
    for i, window in enumerate(windows):
        for j, value in enumerate(window):
            window_rows.append({
                'window_number': i + 1,
                'sample_index': j,
                'amplitude': value
            })

    if window_rows:
        pd.DataFrame(window_rows).to_csv(output_path, index=False)
        logger.info(f"  Saved {len(windows)} windows: {output_path}")


def save_template_data(template: np.ndarray, output_path: str):
    """Save template signal."""
    if len(template) > 0:
        template_df = pd.DataFrame({
            'sample_index': range(len(template)),
            'amplitude': template
        })
        template_df.to_csv(output_path, index=False)
        logger.info(f"  Saved template: {output_path}")


def create_final_output(filtered_windows: List[np.ndarray], metadata: Dict,
                       segment_has_bad_data: bool) -> pd.DataFrame:
    """
    STEP 6: Create Final Wide-Format Output

    Matches d7.py lines 475-495 (windowing and tagging logic).
    """
    logger.info("Step 6: Creating Final Output (Wide Format)")

    if not filtered_windows:
        logger.warning("  No filtered windows to save!")
        return pd.DataFrame()

    # Validate window length
    window_length = len(filtered_windows[0])
    if window_length != 100:
        logger.warning(f"  Window length is {window_length}, expected 100!")

    # Pre-compute column names
    sample_col_names = [f"amplitude_sample_{i}" for i in range(100)]

    final_rows = []
    glucose_value = metadata['glucose']
    channel = metadata['channel']

    for window_idx, window in enumerate(filtered_windows):
        if len(window) != 100:
            continue

        # Tagging Logic (matches d7.py)
        # 4040 = Contains Repaired/Fake Data
        # 8080 = Pure, Strict Data
        code_prefix = 4040 if segment_has_bad_data else 8080

        # Unique window ID
        unique_window_id = int(f"{code_prefix}{window_idx:06d}")

        row_dict = {
            'channel': channel,
            'window_index': unique_window_id,
            'glucose_mg_dl': float(glucose_value)
        }

        # Add BP data if available
        if metadata.get('systolic'):
            row_dict['systolic_mmhg'] = float(metadata['systolic'])
        if metadata.get('diastolic'):
            row_dict['diastolic_mmhg'] = float(metadata['diastolic'])

        # Add window samples
        row_dict.update(dict(zip(sample_col_names, window)))
        final_rows.append(row_dict)

    final_df = pd.DataFrame(final_rows)

    logger.info(f"  Created {len(final_df)} rows in wide format")

    # Apply glucose range filter (12-483 mg/dL) - matches d7.py lines 524-530
    before_filter = len(final_df)
    final_df = final_df[
        (final_df['glucose_mg_dl'] >= 12) &
        (final_df['glucose_mg_dl'] <= 483)
    ]
    after_filter = len(final_df)

    if after_filter < before_filter:
        logger.warning(f"  Glucose range filter (12-483): {before_filter} -> {after_filter} windows")

    # Final NaN check (matches d7.py lines 537-543)
    if final_df.isnull().values.any():
        nan_cols = final_df.columns[final_df.isna().any()].tolist()
        logger.error(f"  ERROR: Final dataset contains NaN values in columns: {nan_cols}")
        raise ValueError("Final dataset contains NaN values!")

    return final_df


def process_single_file(input_file: str, output_dir: str,
                       sampling_rate_override: Optional[float] = None,
                       height_multiplier: float = 0.3,
                       distance_multiplier: float = 0.8,
                       similarity_threshold: float = 0.85) -> Dict:
    """
    Process a single multi-channel file through the complete pipeline.

    Returns:
        Dict containing processing statistics and status
    """
    logger.info("=" * 70)
    logger.info(f"Processing: {os.path.basename(input_file)}")
    logger.info("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename for outputs
    base_filename = Path(input_file).stem

    try:
        # Load data
        df_raw, metadata = load_multichannel_data(input_file, sampling_rate_override or 100.0)

        # Save Step 0: Raw data
        save_intermediate_file(df_raw,
                             os.path.join(output_dir, f"{base_filename}-01-raw.csv"),
                             "raw data")

        # Determine sampling rate
        sampling_rate = sampling_rate_override or metadata.get('inferred_sampling_rate') or metadata['expected_sampling_rate']

        # Step 1: Time repair
        df_repaired = repair_time_axis(df_raw, sampling_rate)
        save_intermediate_file(df_repaired,
                             os.path.join(output_dir, f"{base_filename}-02-cleaned.csv"),
                             "time-repaired data")

        # Step 2: Signal cleaning
        df_cleaned, bad_data_mask = clean_signal_data(df_repaired)

        # Step 3: Downsample to 100Hz
        df_downsampled, actual_sr = downsample_to_100hz(df_cleaned, bad_data_mask, sampling_rate)
        save_intermediate_file(df_downsampled[['time', 'ppg']],
                             os.path.join(output_dir, f"{base_filename}-03-downsampled.csv"),
                             "downsampled data (100Hz)")

        # Track if data has been repaired (for tagging logic)
        segment_has_bad_data = df_downsampled['is_bad'].any()

        # Step 4: Preprocess (bandpass filter)
        df_preprocessed, preprocessed_signal = preprocess_signal(df_downsampled, actual_sr)
        save_intermediate_file(df_preprocessed[['time', 'ppg', 'ppg_preprocessed']],
                             os.path.join(output_dir, f"{base_filename}-04-preprocessed.csv"),
                             "preprocessed data (bandpass filtered)")

        # Step 5: Peak detection and window extraction
        peaks, filtered_windows, template, all_windows = detect_peaks_and_extract_windows(
            preprocessed_signal,
            actual_sr,
            height_multiplier,
            distance_multiplier,
            similarity_threshold
        )

        # Save peak data
        save_peaks_data(peaks, df_preprocessed,
                       os.path.join(output_dir, f"{base_filename}-05-peaks.csv"))

        # Save all windows
        save_windows_data(all_windows,
                         os.path.join(output_dir, f"{base_filename}-06-windows.csv"),
                         "all_windows")

        # Save filtered windows
        save_windows_data(filtered_windows,
                         os.path.join(output_dir, f"{base_filename}-07-filtered.csv"),
                         "filtered_windows")

        # Save template
        save_template_data(template,
                          os.path.join(output_dir, f"{base_filename}-08-template.csv"))

        # Step 6: Create final output
        final_df = create_final_output(filtered_windows, metadata, segment_has_bad_data)

        if final_df.empty:
            logger.warning("No valid windows in final output!")
            stats = {
                'status': 'FAILED_NO_WINDOWS',
                'input_file': input_file,
                'metadata': metadata
            }
            return stats

        # Save final output
        output_file = os.path.join(output_dir, f"{base_filename}-output.csv")
        final_df.to_csv(output_file, index=False)
        logger.info(f"  Saved final output: {output_file}")

        # Create statistics
        stats = {
            'status': 'SUCCESS',
            'input_file': input_file,
            'output_file': output_file,
            'metadata': metadata,
            'processing': {
                'sampling_rate': actual_sr,
                'total_samples': len(df_raw),
                'duration_seconds': metadata['duration_seconds'],
                'peaks_detected': len(peaks),
                'windows_extracted': len(all_windows),
                'windows_filtered': len(filtered_windows),
                'filtering_rate': len(filtered_windows) / len(all_windows) * 100 if all_windows else 0,
                'final_output_rows': len(final_df),
                'data_quality': 'repaired' if segment_has_bad_data else 'pure'
            }
        }

        # Save metadata
        metadata_file = os.path.join(output_dir, f"{base_filename}-metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Saved metadata: {metadata_file}")

        logger.info("=" * 70)
        logger.info(f"SUCCESS: Processed {os.path.basename(input_file)}")
        logger.info(f"  Final output: {len(final_df)} rows")
        logger.info(f"  Quality: {stats['processing']['data_quality']}")
        logger.info("=" * 70)

        return stats

    except Exception as e:
        logger.error(f"ERROR processing {input_file}: {e}", exc_info=True)
        stats = {
            'status': 'FAILED',
            'input_file': input_file,
            'error': str(e)
        }
        return stats


def process_folder(input_folder: str, output_root: str, **kwargs) -> Dict:
    """
    Process all CSV files in a folder.

    Creates separate output folders for each input file.
    """
    logger.info("=" * 70)
    logger.info("BATCH PROCESSING MODE")
    logger.info("=" * 70)

    input_path = Path(input_folder)
    if not input_path.is_dir():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    # Find all CSV files
    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {input_folder}")

    logger.info(f"Found {len(csv_files)} CSV files to process")

    results = []
    success_count = 0

    for csv_file in csv_files:
        # Create output directory for this file
        file_output_dir = os.path.join(output_root, csv_file.stem)

        # Process file
        stats = process_single_file(str(csv_file), file_output_dir, **kwargs)
        results.append(stats)

        if stats['status'] == 'SUCCESS':
            success_count += 1

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total files: {len(csv_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(csv_files) - success_count}")
    logger.info("=" * 70)

    # Save batch summary
    summary_file = os.path.join(output_root, "batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'total_files': len(csv_files),
            'successful': success_count,
            'failed': len(csv_files) - success_count,
            'results': results
        }, f, indent=2)
    logger.info(f"Saved batch summary: {summary_file}")

    return {
        'total': len(csv_files),
        'successful': success_count,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Channel Training Data Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python generate_multichannel_training_data.py --input force-GLUC123-SYS140-DIA91.csv --output ./output

  # Process entire folder
  python generate_multichannel_training_data.py --input_folder C:\\senzrtech\\Multi-channel\\multi-channel-input-files --output ./output

  # Custom parameters
  python generate_multichannel_training_data.py --input file.csv --output ./out --height 0.3 --distance 0.8 --similarity 0.85
        """
    )

    # Input/Output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Single input CSV file')
    input_group.add_argument('--input_folder', type=str, help='Folder containing multiple CSV files')
    parser.add_argument('--output', type=str, required=True, help='Output directory')

    # Processing parameters
    parser.add_argument('--sampling_rate', type=float, default=None,
                       help='Override sampling rate (default: auto-detect or 100Hz)')
    parser.add_argument('--height', type=float, default=0.3,
                       help='Peak height threshold multiplier (default: 0.3)')
    parser.add_argument('--distance', type=float, default=0.8,
                       help='Peak distance threshold multiplier (default: 0.8)')
    parser.add_argument('--similarity', type=float, default=0.85,
                       help='Template similarity threshold (default: 0.85)')

    args = parser.parse_args()

    try:
        if args.input:
            # Single file mode
            stats = process_single_file(
                args.input,
                args.output,
                sampling_rate_override=args.sampling_rate,
                height_multiplier=args.height,
                distance_multiplier=args.distance,
                similarity_threshold=args.similarity
            )

            if stats['status'] != 'SUCCESS':
                logger.error(f"Processing failed with status: {stats['status']}")
                return 1

        else:
            # Batch mode
            batch_stats = process_folder(
                args.input_folder,
                args.output,
                sampling_rate_override=args.sampling_rate,
                height_multiplier=args.height,
                distance_multiplier=args.distance,
                similarity_threshold=args.similarity
            )

            if batch_stats['successful'] == 0:
                logger.error("No files were processed successfully!")
                return 1

        logger.info("\nProcessing complete!")
        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
