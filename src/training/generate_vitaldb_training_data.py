#!/usr/bin/env python
"""
VitalDB Training Data Generator
===============================
Standalone app to generate training files (ppg_windows.csv and glucose_labels.csv)
from VitalDB case data.

Usage:
1) Ensure labs CSV exists (default: ./labs_data/lab_data.csv).
2) Run a single case:
   python generate_training_data.py --case_id 1 --track SNUADC/PLETH --labs_csv ./labs_data/lab_data.csv
3) Or run a local case folder:
   python generate_training_data.py --case_dir ./data/case_16_SNUADC_PLETH --labs_csv ./labs_data/lab_data.csv
4) Outputs are written under --output (default: ./training_data).

Notes:
- Prompts for how many glucose measurements to use unless --glucose_limit is set.
- For batch runs, use --process_all with --data_root pointing to case folders.

Features:
- Extracts PPG data from VitalDB
- Detects peaks and filters windows
- Generates glucose labels from labs CSV (per measurement)
- Outputs training-ready CSV files
"""

import os
import sys
import argparse
import json
import logging
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_extraction.ppg_extractor import PPGExtractor
from src.data_extraction.ppg_segmentation import PPGSegmenter
from src.data_extraction.peak_detection import ppg_peak_detection_pipeline_with_template

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_local_ppg_data(case_dir, track_name=None, sampling_rate_override=None):
    """
    Load a locally downloaded VitalDB case folder (already contains *_raw.csv and metadata).

    Args:
        case_dir: Path to the case directory.
        track_name: Optional override for track name.
        sampling_rate_override: Optional override for sampling rate when metadata is missing.

    Returns:
        Dictionary shaped like PPGExtractor.extract_ppg_raw output.
    """
    case_path = Path(case_dir)
    if not case_path.is_dir():
        raise ValueError(f"Local case directory not found: {case_dir}")

    # Pick the first *_raw.csv (fallback to any csv)
    csv_candidates = sorted(case_path.glob("*_raw.csv"))
    if not csv_candidates:
        csv_candidates = sorted(case_path.glob("*.csv"))
    if not csv_candidates:
        raise ValueError(f"No CSV file found in {case_dir}")
    csv_file = csv_candidates[0]

    # Metadata is optional but preferred
    metadata = {}
    metadata_file = None
    metadata_candidates = sorted(case_path.glob("*metadata*.json"))
    if metadata_candidates:
        metadata_file = metadata_candidates[0]
        try:
            metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        except Exception as exc:
            logger.warning(f"Could not read metadata from {metadata_file}: {exc}")

    meta_case_id = metadata.get('case_id')
    meta_track_name = metadata.get('track_name')
    case_info = metadata.get('case_info')

    chosen_track = track_name or meta_track_name
    expected_sampling_rate = metadata.get('expected_sampling_rate') or sampling_rate_override
    if expected_sampling_rate is None and chosen_track:
        expected_sampling_rate = PPGExtractor.PPG_TRACKS.get(chosen_track)

    return {
        'case_id': meta_case_id,
        'track_name': chosen_track,
        'expected_sampling_rate': expected_sampling_rate,
        'csv_file': str(csv_file),
        'metadata_file': str(metadata_file) if metadata_file else None,
        'case_info': case_info,
    }


def _load_glucose_measurements_from_labs(case_id, labs_csv_path):
    """
    Load all glucose measurements for a case from a labs CSV.

    The labs file must have columns: caseid, dt, name, result
    Glucose rows are detected when `name` contains 'glu' (case-insensitive).
    """
    if not labs_csv_path:
        raise ValueError("labs_csv path is required to load glucose measurements.")

    labs_path = Path(labs_csv_path)
    if not labs_path.is_file():
        raise ValueError(f"Labs CSV not found: {labs_path}")

    try:
        df = pd.read_csv(labs_path)
    except Exception as exc:
        raise ValueError(f"Could not read labs CSV {labs_path}: {exc}") from exc

    df.columns = [c.lower() for c in df.columns]
    expected_cols = {'caseid', 'dt', 'name', 'result'}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Labs CSV missing required columns {expected_cols}: {labs_path}")

    # Only keep glucose-related rows for this case
    df = df[(df['caseid'] == case_id) & (df['name'].str.contains('glu', case=False, na=False))]

    if df.empty:
        raise ValueError(f"No glucose rows found in labs CSV for case {case_id}")

    df = df.copy()
    df['dt'] = pd.to_numeric(df['dt'], errors='coerce')
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df = df.dropna(subset=['dt', 'result'])
    df = df[(df['result'] > 0) & (df['result'] < 1000)]
    df = df.sort_values('dt')

    if df.empty:
        raise ValueError(f"No valid glucose measurements after cleaning labs CSV for case {case_id}")

    measurements = []
    for _, row in df.iterrows():
        measurements.append({
            'glucose': float(row['result']),
            'source': 'labs_csv',
            'time': float(row['dt'])
        })

    logger.info(f"Loaded {len(measurements)} glucose measurements from labs CSV for case {case_id}")
    return measurements, 'labs_csv'


def _select_glucose_measurements(measurements, glucose_limit):
    total_measurements = len(measurements)
    logger.info(f"Total glucose measurements available: {total_measurements}")

    if total_measurements == 0:
        return [], total_measurements, 0

    selected_count = glucose_limit
    if glucose_limit is None:
        if sys.stdin is not None and sys.stdin.isatty():
            while True:
                resp = input(
                    f"How many glucose measurements to import? (1-{total_measurements}, or 'all') [all]: "
                ).strip().lower()
                if resp in ("", "all"):
                    selected_count = total_measurements
                    break
                try:
                    selected_count = int(resp)
                except ValueError:
                    logger.warning("Invalid entry. Enter an integer or 'all'.")
                    continue
                if 1 <= selected_count <= total_measurements:
                    break
                logger.warning(f"Please enter a value between 1 and {total_measurements}.")
        else:
            logger.info("Non-interactive session detected; using all glucose measurements.")
            selected_count = total_measurements
    else:
        if glucose_limit < 1:
            raise ValueError("--glucose_limit must be >= 1")
        if glucose_limit > total_measurements:
            logger.warning(
                f"Requested {glucose_limit} glucose measurements, but only {total_measurements} are available. Using all."
            )
            selected_count = total_measurements

    if selected_count == total_measurements:
        logger.info("Using all glucose measurements.")
    else:
        logger.info(f"Using first {selected_count} glucose measurements (sorted by time).")

    return measurements[:selected_count], total_measurements, selected_count


def _copy_input_files_to_output(ppg_result, output_dir):
    """
    Copy the raw input artifacts (CSV + JSON metadata) into the output directory.
    This keeps the exact inputs bundled with the generated training assets.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    candidates = []
    for key in ('csv_file', 'metadata_file'):
        if ppg_result.get(key):
            candidates.append(Path(ppg_result[key]))

    # Add any JSON files that live next to the CSV (e.g., metadata variants)
    csv_parent = None
    if ppg_result.get('csv_file'):
        csv_parent = Path(ppg_result['csv_file']).parent
        candidates.extend(csv_parent.glob("*.json"))

    copied = []
    seen = set()
    for src in candidates:
        try:
            src_path = src.resolve()
        except Exception:
            continue

        if src_path in seen or not src_path.is_file():
            continue
        seen.add(src_path)

        dest = output_path / src_path.name
        if dest.resolve() == src_path:
            copied.append(str(dest))
            continue

        try:
            shutil.copy2(src_path, dest)
            copied.append(str(dest))
            logger.info(f"Copied input file to output: {dest}")
        except Exception as exc:
            logger.warning(f"Could not copy input file {src_path} -> {dest}: {exc}")

    if not copied:
        logger.info("No input artifacts were copied into the output directory.")
    return copied


def _select_time_windows(time_arr, duration_sec=20.0, num_windows=3):
    """Pick evenly spaced time windows for zoomed visualization (currently unused; kept for reference)."""
    if time_arr is None or len(time_arr) == 0:
        return []
    t_min, t_max = float(np.nanmin(time_arr)), float(np.nanmax(time_arr))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return []
    margins = duration_sec / 2.0
    centers = np.linspace(t_min + margins, t_max - margins, num_windows)
    return [(c - margins, c + margins) for c in centers if (c - margins) < (c + margins)]


def generate_training_data(case_id, track_name, output_dir,
                          height_multiplier=0.3, distance_multiplier=0.8,
                          similarity_threshold=0.85, sampling_rate_override=None,
                          case_dir=None, labs_csv=None, glucose_limit=None):
    """
    Generate training data files from VitalDB case.

    Args:
        case_id: VitalDB case ID
        track_name: PPG track name (e.g., 'SNUADC/PLETH')
        output_dir: Output directory for CSV files
        height_multiplier: Peak detection height threshold multiplier
        distance_multiplier: Peak detection distance threshold multiplier
        similarity_threshold: Template similarity threshold for filtering windows
        sampling_rate_override: Override sampling rate when metadata is missing
        case_dir: Optional path to local case folder containing *_raw.csv
        labs_csv: Labs CSV path (caseid, dt, name, result) used to resolve glucose

    Returns:
        Tuple of (ppg_file_path, glucose_file_path, stats_dict)
    """
    logger.info("=" * 70)
    logger.info("VitalDB Training Data Generator")
    logger.info("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract PPG data
    if case_dir:
        logger.info("Step 1: Loading PPG data from local folder...")
        ppg_result = load_local_ppg_data(case_dir, track_name, sampling_rate_override)
        case_id = case_id or ppg_result.get('case_id')
        track_name = track_name or ppg_result.get('track_name')
        if case_id is None:
            raise ValueError("case_id is required when using local data (provide via --case_id or metadata).")
        if track_name is None:
            raise ValueError("track is required when using local data (provide via --track or metadata).")
        logger.info(f"Using local file: {ppg_result['csv_file']}")
    else:
        logger.info("Step 1: Extracting PPG data from VitalDB...")
        ppg_extractor = PPGExtractor()

        try:
            ppg_result = ppg_extractor.extract_ppg_raw(case_id, track_name, output_dir)
            logger.info(f"Extracted PPG data: {ppg_result['num_samples']} samples")
        except Exception as e:
            logger.error(f"Failed to extract PPG data: {e}")
            raise

    logger.info(f"Case ID: {case_id}")
    logger.info(f"Track: {track_name}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    copied_inputs = _copy_input_files_to_output(ppg_result, output_dir)

    # Step 2: Load raw PPG data (cleansing happens per measurement window)
    logger.info("\nStep 2: Loading PPG data (raw; cleanse per measurement window)...")
    df = pd.read_csv(ppg_result['csv_file'])
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['ppg'] = pd.to_numeric(df['ppg'], errors='coerce')
    ppg_result.setdefault('num_samples', len(df))
    sampling_rate = ppg_result.get('expected_sampling_rate') or sampling_rate_override
    if df['time'].isna().all():
        if sampling_rate is None:
            raise ValueError("Cannot determine sampling rate for track with all NaN time values")
        df['time'] = np.arange(len(df)) / sampling_rate
    elif df['time'].isna().any() and sampling_rate is not None:
        # Fill missing times using index/sampling rate for robustness
        missing_mask = df['time'].isna()
        df.loc[missing_mask, 'time'] = np.arange(len(df))[missing_mask] / sampling_rate

    # Step 3: Get glucose values
    logger.info("\nStep 3: Generating glucose labels with per-measurement windows from labs CSV...")
    measurements, glucose_source = _load_glucose_measurements_from_labs(case_id, labs_csv or "./labs_data/lab_data.csv")
    measurements, total_glucose_count, used_glucose_count = _select_glucose_measurements(
        measurements,
        glucose_limit
    )
    measurement_windows_16min = len(measurements)

    # Prepare base dataframe and time column
    df = pd.read_csv(ppg_result['csv_file'])
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['ppg'] = pd.to_numeric(df['ppg'], errors='coerce')
    ppg_result.setdefault('num_samples', len(df))

    sampling_rate = ppg_result.get('expected_sampling_rate') or sampling_rate_override
    if df['time'].isna().all():
        if sampling_rate is None:
            raise ValueError("Cannot determine sampling rate for track with all NaN time values")
        df['time'] = np.arange(len(df)) / sampling_rate
    elif df['time'].isna().any() and sampling_rate is not None:
        missing_mask = df['time'].isna()
        df.loc[missing_mask, 'time'] = np.arange(len(df))[missing_mask] / sampling_rate

    # Process per measurement window (16 minutes centered on glucose timestamp when available)
    window_minutes = 8
    total_peaks = 0
    total_all_windows = 0
    total_filtered_windows = 0
    rows_before_downsampling = 0
    rows_after_downsampling = 0
    unique_glucose_values = set()
    ppg_rows = []
    glucose_rows = []
    window_index = 0
    window_length = None
    data_size_before_window = len(df)
    data_size_after_window = 0

    for meas_idx, meas in enumerate(measurements):
        meas_time = meas.get('time')
        meas_source = meas.get('source')
        measurement_dt = float(meas_time) if meas_time is not None else np.nan

        seg_df = df.copy()
        label_start_time = None
        label_end_time = None
        if meas_time is not None:
            start_t = meas_time - window_minutes * 60
            end_t = meas_time + window_minutes * 60
            seg_df = seg_df[(seg_df['time'] >= start_t) & (seg_df['time'] <= end_t)]
            logger.info(f"  Glucose measurement #{meas_idx} time={meas_time} -> window [{start_t}, {end_t}] sec; samples: {len(seg_df)}")
            if seg_df.empty:
                logger.warning(f"  No samples in time window for measurement #{meas_idx}; falling back to full signal")
                seg_df = df.copy()
            # Use this measurement's glucose value
            label_glucose = float(meas['glucose'])
        else:
            logger.info(f"  Glucose measurement #{meas_idx} without timestamp -> using full signal")
            label_glucose = float(meas['glucose'])

        data_size_after_window += len(seg_df)
        unique_glucose_values.add(float(label_glucose))

        # Cleanse segment
        raw_seg_df = seg_df.copy()
        seg_df = seg_df.dropna(subset=['ppg', 'time'])
        if seg_df.empty:
            logger.warning(f"No PPG samples after cleansing for measurement #{meas_idx}")
            continue

        time = seg_df['time'].values
        signal = seg_df['ppg'].values
        logger.info(f"Cleansed data (measurement #{meas_idx}): {len(signal)} samples")

        # Optional downsample to 100 Hz to standardize processing
        seg_sampling_rate = sampling_rate
        if seg_sampling_rate is None:
            raise ValueError("Cannot preprocess signal without a known sampling rate. Provide metadata or --sampling_rate.")
        pre_downsample_len = len(signal)
        if seg_sampling_rate > 100:
            target_sr = 100.0
            start_t, end_t = time.min(), time.max()
            if end_t <= start_t:
                logger.warning(f"Invalid time range for measurement #{meas_idx}")
                continue
            new_time = np.arange(start_t, end_t, 1.0 / target_sr)
            if len(new_time) < 2:
                logger.warning(f"Insufficient samples after downsampling for measurement #{meas_idx}")
                continue
            signal = np.interp(new_time, time, signal)
            time = new_time
            seg_sampling_rate = target_sr
            logger.info(f"Downsampled to {target_sr} Hz (measurement #{meas_idx})")
        post_downsample_len = len(signal)
        rows_before_downsampling += pre_downsample_len
        rows_after_downsampling += post_downsample_len

        # Preprocess signal
        segmenter = PPGSegmenter(sampling_rate=seg_sampling_rate)
        preprocessed_signal = segmenter.preprocess_signal(signal)
        logger.info(f"Signal preprocessed (measurement #{meas_idx})")

        # Plotting disabled (waveform, overlay, PSD)

        # Peak detection and windowing
        signal_mean = np.mean(preprocessed_signal)
        signal_std = np.std(preprocessed_signal)
        height_threshold = signal_mean + height_multiplier * signal_std
        distance_threshold = distance_multiplier * seg_sampling_rate

        peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
            ppg_signal=preprocessed_signal,
            fs=float(seg_sampling_rate),
            window_duration=1.0,
            height_threshold=float(height_threshold),
            distance_threshold=distance_threshold,
            similarity_threshold=similarity_threshold
        )

        logger.info(f"Detected {len(peaks)} peaks (measurement #{meas_idx})")
        logger.info(f"Extracted {len(all_windows)} windows (measurement #{meas_idx})")
        logger.info(f"Filtered to {len(filtered_windows)} high-quality windows (measurement #{meas_idx})")

        if len(filtered_windows) == 0:
            if len(all_windows) > 0:
                logger.warning(f"No windows passed quality filters for measurement #{meas_idx}; using all windows instead")
                filtered_windows = all_windows
            else:
                logger.warning(f"No valid windows extracted for measurement #{meas_idx}")
                continue

        if window_length is None and filtered_windows:
            window_length = len(filtered_windows[0])

        total_peaks += len(peaks)
        total_all_windows += len(all_windows)
        total_filtered_windows += len(filtered_windows)

        # Save PPG rows with metadata
        for local_window_idx, window in enumerate(filtered_windows):
            for sample_idx, amplitude in enumerate(window):
                ppg_rows.append({
                    'case_id': case_id,
                    'window_index': window_index,
                    'sample_index': sample_idx,
                    'amplitude': float(amplitude),
                    'glucose_dt': measurement_dt
                })
            glucose_rows.append({
                'case_id': case_id,
                'window_index': window_index,
                'glucose_mg_dl': float(label_glucose),
                'glucose_dt': measurement_dt
            })
            window_index += 1

    # Fallback: if no windows at all, build naive 1-second windows over the full signal
    if not glucose_rows:
        logger.warning("No valid windows extracted for any glucose measurement. Falling back to naive fixed windows.")
        if sampling_rate is None:
            raise ValueError("Cannot fall back to fixed windows without a sampling rate.")
        window_len = int(sampling_rate)
        if window_len <= 0:
            raise ValueError("Invalid sampling rate for fallback windowing.")
        pre_df = df.dropna(subset=['ppg', 'time'])
        if pre_df.empty:
            if df['ppg'].notna().any():
                # Use available PPG with synthetic time
                pre_df = pd.DataFrame({
                    'time': np.arange(df['ppg'].notna().sum()) / sampling_rate,
                    'ppg': df['ppg'].dropna().values
                })
                logger.warning("No aligned PPG/time rows; using synthetic time for fallback.")
            else:
                raise ValueError("No PPG samples available after cleansing; cannot build fallback windows.")
        full_signal = pre_df['ppg'].values
        num_full_windows = len(full_signal) // window_len
        if num_full_windows == 0:
            raise ValueError("Signal too short to create even one window in fallback mode.")

        # Use first measurement glucose for fallback
        meas_glucose = float(measurements[0]['glucose'])
        meas_source = measurements[0].get('source')
        meas_time = measurements[0].get('time')
        fallback_dt = float(meas_time) if meas_time is not None else np.nan
        window_length = window_len
        if not unique_glucose_values:
            unique_glucose_values.add(float(meas_glucose))
        if rows_before_downsampling == 0:
            rows_before_downsampling = len(pre_df)
        if rows_after_downsampling == 0:
            rows_after_downsampling = len(pre_df)
        for w_idx in range(num_full_windows):
            start = w_idx * window_len
            window = full_signal[start:start + window_len]
            for sample_idx, amplitude in enumerate(window):
                ppg_rows.append({
                    'case_id': case_id,
                    'window_index': window_index,
                    'sample_index': sample_idx,
                    'amplitude': float(amplitude),
                    'glucose_dt': fallback_dt
                })
            glucose_rows.append({
                'case_id': case_id,
                'window_index': window_index,
                'glucose_mg_dl': float(meas_glucose),
                'glucose_dt': fallback_dt
            })
            window_index += 1
        total_all_windows = num_full_windows
        total_filtered_windows = num_full_windows

    # Ensure every window has a glucose label (align counts before saving)
    unique_windows = pd.DataFrame(ppg_rows)['window_index'].nunique() if ppg_rows else 0
    if unique_windows != len(glucose_rows):
        logger.warning(f"Mismatched window counts: {unique_windows} windows vs {len(glucose_rows)} glucose labels. Aligning counts.")
        # Build a mapping of window_index present in PPG rows
        existing_indices = sorted(set([row['window_index'] for row in ppg_rows]))
        # Rebuild glucose rows to match existing window indices; use nearest available glucose label if missing
        rebuilt_glucose_rows = []
        label_df = pd.DataFrame(glucose_rows) if glucose_rows else pd.DataFrame(columns=['glucose_mg_dl', 'glucose_dt'])
        label_glucose_array = label_df['glucose_mg_dl'].to_numpy() if 'glucose_mg_dl' in label_df else np.array([])
        label_dt_array = label_df['glucose_dt'].to_numpy() if 'glucose_dt' in label_df else np.array([])
        for idx_pos, win_idx in enumerate(existing_indices):
            if idx_pos < len(label_glucose_array):
                glu_val = float(label_glucose_array[idx_pos])
            elif len(label_glucose_array) > 0:
                glu_val = float(label_glucose_array[-1])
            else:
                raise ValueError("No glucose labels available to align with windows.")
            if idx_pos < len(label_dt_array):
                glu_dt_val = float(label_dt_array[idx_pos])
            elif len(label_dt_array) > 0:
                glu_dt_val = float(label_dt_array[-1])
            else:
                glu_dt_val = np.nan
            rebuilt_glucose_rows.append({
                'case_id': case_id,
                'window_index': win_idx,
                'glucose_mg_dl': glu_val,
                'glucose_dt': glu_dt_val
            })
        glucose_rows = rebuilt_glucose_rows

    num_windows = len(glucose_rows)
    glucose_labels = np.array([row['glucose_mg_dl'] for row in glucose_rows])
    logger.info(f"Generated {num_windows} glucose labels across {len(measurements)} measurement(s)")

    # Step 6: Save PPG windows to CSV
    logger.info("\nStep 6: Saving PPG windows...")
    ppg_windows_file = os.path.join(output_dir, 'ppg_windows.csv')

    ppg_df = pd.DataFrame(ppg_rows)
    ppg_df.to_csv(ppg_windows_file, index=False)

    logger.info(f"Saved PPG windows: {ppg_windows_file}")
    logger.info(f"  Format: {num_windows} windows x {window_length if window_length else 'N/A'} samples")

    # Step 7: Save glucose labels to CSV
    logger.info("\nStep 7: Saving glucose labels...")
    glucose_file = os.path.join(output_dir, 'glucose_labels.csv')

    glucose_df = pd.DataFrame(glucose_rows)
    glucose_df.to_csv(glucose_file, index=False)

    logger.info(f"Saved glucose labels: {glucose_file}")

    # Step 8: Generate statistics
    logger.info("\nStep 8: Summary Statistics")
    logger.info("=" * 70)

    stats = {
        'case_id': case_id,
        'track': track_name,
        'glucose_source': glucose_source,
        'glucose_value': float(glucose_labels[0]),
        'glucose_measurements_total': total_glucose_count,
        'glucose_measurements_used': used_glucose_count,
        'measurement_windows_16min': measurement_windows_16min,
        'data_size_before_window': data_size_before_window,
        'data_size_after_window': data_size_after_window,
        'total_available_windows': total_all_windows,
        'windows_with_glucose': num_windows,
        'num_windows': num_windows,
        'window_length': window_length,
        'sampling_rate': sampling_rate,
        'total_peaks': total_peaks,
        'filtered_windows': num_windows,
        'filtering_rate': float(num_windows / total_all_windows * 100) if total_all_windows else 0,
        'ppg_file': ppg_windows_file,
        'glucose_file': glucose_file,
        'input_files': copied_inputs
    }

    logger.info(f"  Case ID: {stats['case_id']}")
    logger.info(f"  Track: {stats['track']}")
    logger.info(f"  Glucose source: {stats['glucose_source']}")
    logger.info(f"  Glucose value: {stats['glucose_value']} mg/dL")
    logger.info(f"  Number of windows: {stats['num_windows']}")
    logger.info(f"  Window length: {stats['window_length']} samples")
    logger.info(f"  Sampling rate: {stats['sampling_rate']} Hz")
    logger.info(f"  Total peaks detected: {stats['total_peaks']}")
    logger.info(f"  Windows after filtering: {stats['filtered_windows']}")
    logger.info(f"  Filtering rate: {stats['filtering_rate']:.1f}%")
    logger.info("")

    logger.info(f"PPG windows file: {stats['ppg_file']}")
    logger.info(f"Glucose labels file: {stats['glucose_file']}")
    logger.info("=" * 70)
    logger.info("Training data generation complete!")
    logger.info("=" * 70)

    # Save per-case analysis summary
    analysis_dir = Path("data_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / "ppg_glucose_analysis.csv"
    analysis_row = {
        'case_id': case_id,
        'track': track_name,
        'glucose_source': glucose_source,
        'glucose_measurements_total': total_glucose_count,
        'glucose_measurements_used': used_glucose_count,
        'glucose_measurement_count': len(measurements),
        'unique_glucose_values_count': len(unique_glucose_values),
        'measurement_windows_16min': measurement_windows_16min,
        'data_size_before_window': data_size_before_window,
        'data_size_after_window': data_size_after_window,
        'total_available_windows': total_all_windows,
        'windows_with_glucose': num_windows,
        'rows_before_downsampling': rows_before_downsampling,
        'rows_after_downsampling': rows_after_downsampling,
        'num_windows': num_windows,
        'window_length': window_length,
        'sampling_rate': sampling_rate,
        'total_peaks': total_peaks,
        'filtered_windows': total_filtered_windows
    }
    analysis_df = pd.DataFrame([analysis_row])
    if analysis_path.exists():
        try:
            existing_df = pd.read_csv(analysis_path)
            analysis_df = pd.concat([existing_df, analysis_df], ignore_index=True)
        except Exception as exc:
            logger.warning(f"Could not read existing analysis file {analysis_path}: {exc}")
    analysis_df.to_csv(analysis_path, index=False)
    logger.info(f"Saved PPG/glucose analysis summary to {analysis_path}")

    return ppg_windows_file, glucose_file, stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data from VitalDB case',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use manual glucose value
  python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0

  # Auto-extract glucose from clinical data
  python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose auto

  # Process a locally downloaded case folder (no API download)
  python generate_training_data.py --case_dir ./data/case_16_SNUADC_PLETH --glucose auto

  # Specify output directory
  python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0 --output ./my_data

  # Adjust peak detection parameters
  python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0 \\
      --height 0.4 --distance 0.9 --similarity 0.9
        """
    )

    # Required arguments
    parser.add_argument('--case_id', type=int, required=False,
                        help='VitalDB case ID (e.g., 1, 2, 3, ...)')
    parser.add_argument('--track', type=str, required=False,
                        help='PPG track name (e.g., SNUADC/PLETH, Primus/PLETH)')

    # Optional arguments
    parser.add_argument('--output', type=str, default='./training_data',
                        help='Output directory for CSV files (default: ./training_data)')
    parser.add_argument('--case_dir', type=str, default=None,
                        help='Path to a local case folder containing *_raw.csv and optional metadata JSON')
    parser.add_argument('--process_all', action='store_true',
                        help='Process every case folder found under --data_root (uses local files)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root folder containing local case subdirectories (default: ./data)')
    parser.add_argument('--sampling_rate', type=float, default=None,
                        help='Override sampling rate when metadata is missing')
    parser.add_argument('--labs_csv', type=str, default='./labs_data/lab_data.csv',
                        help='Path to labs CSV (caseid, dt, name, result) to source glucose labels')
    parser.add_argument('--glucose_limit', type=int, default=None,
                        help='Number of glucose measurements to import (default: prompt; all if non-interactive)')

    # Peak detection parameters
    parser.add_argument('--height', type=float, default=0.3,
                        help='Peak height threshold multiplier (default: 0.3)')
    parser.add_argument('--distance', type=float, default=0.8,
                        help='Peak distance threshold multiplier (default: 0.8)')
    parser.add_argument('--similarity', type=float, default=0.85,
                        help='Template similarity threshold (default: 0.85)')

    args = parser.parse_args()

    try:
        # Batch processing mode: iterate over local case folders
        if args.process_all:
            root = Path(args.data_root)
            if not root.is_dir():
                logger.error(f"--data_root does not exist: {root}")
                return 1

            exit_code = 0
            for case_path in sorted(p for p in root.iterdir() if p.is_dir()):
                logger.info("\n" + "=" * 70)
                logger.info(f"Processing case folder: {case_path.name}")
                logger.info("=" * 70)
                try:
                    meta = load_local_ppg_data(case_path, args.track, args.sampling_rate)
                    case_id = args.case_id or meta.get('case_id')
                    track_name = args.track or meta.get('track_name')
                    if case_id is None or track_name is None:
                        logger.error(f"Missing case_id/track for {case_path}. Add metadata or pass --case_id/--track.")
                        exit_code = 1
                        continue

                    output_dir = os.path.join(args.output, case_path.name)
                    generate_training_data(
                        case_id=case_id,
                        track_name=track_name,
                        output_dir=output_dir,
                        height_multiplier=args.height,
                        distance_multiplier=args.distance,
                        similarity_threshold=args.similarity,
                        sampling_rate_override=args.sampling_rate,
                        case_dir=str(case_path),
                        labs_csv=args.labs_csv,
                        glucose_limit=args.glucose_limit
                    )
                except Exception as exc:
                    logger.error(f"Failed to process {case_path}: {exc}")
                    exit_code = 1
            return exit_code

        # Single-case mode
        local_case_dir = args.case_dir
        if local_case_dir and (args.case_id is None or args.track is None):
            try:
                meta = load_local_ppg_data(local_case_dir, args.track, args.sampling_rate)
                args.case_id = args.case_id or meta.get('case_id')
                args.track = args.track or meta.get('track_name')
            except Exception as exc:
                logger.error(f"Could not infer case_id/track from {local_case_dir}: {exc}")
                return 1

        if args.case_id is None or args.track is None:
            logger.error("case_id and track are required unless they can be inferred from --case_dir/metadata.")
            return 1

        ppg_file, glucose_file, stats = generate_training_data(
            case_id=args.case_id,
            track_name=args.track,
            output_dir=args.output,
            height_multiplier=args.height,
            distance_multiplier=args.distance,
            similarity_threshold=args.similarity,
            sampling_rate_override=args.sampling_rate,
            case_dir=local_case_dir,
            labs_csv=args.labs_csv,
            glucose_limit=args.glucose_limit
        )

        logger.info("\nSUCCESS! Training data files are ready.")
        logger.info(f"\nTo train the model, run:")
        logger.info(f"  python -m src.training.train_glucose_predictor --data_dir {args.output}")

        return 0

    except Exception as e:
        logger.error(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
