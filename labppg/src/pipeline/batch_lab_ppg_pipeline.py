#!/usr/bin/env python3
"""
Batch Lab PPG Pipeline
======================
Processes all PPG CSV files from an input folder through the full pipeline:
  Load -> Cleanse -> Downsample -> Preprocess -> Peak Detection ->
  Window Extraction -> Template -> Filtering -> Save ppg_windows.csv ->
  Inference -> glucose_predictions.csv

Inputs:
  --input_dir   : Folder containing raw PPG CSV files (time, ppg columns)
  --model_path  : Path to the trained model checkpoint (.pth)
  --output_dir  : Root output folder (subfolders created per person/date/time)

Filename convention:
  {Name}_{DD}_{MM}_{YYYY}_{HH}_{MM}_Gluc_{G}_Sys_{S}_Dia_{D}_HR_{H}[_extras]_col2.csv

Usage:
  python batch_lab_ppg_pipeline.py ^
    --input_dir  "C:\\IITM\\vitalDB\\data\\VanillaPPG_Batch1_500Hz\\red_PPG" ^
    --model_path "C:\\IITM\\model\\latest_model-143853\\best_model.pth" ^
    --output_dir "C:\\IITM\\vitalDB\\data\\VanillaPPG_Batch1_500Hz\\results"

  python batch_lab_ppg_pipeline.py ^
    --input_dir  "C:\\IITM\\vitalDB\\data\\VanillaPPG_Batch1_500Hz\\red_PPG" ^
    --model_path "C:\\IITM\\model\\latest_model-143853\\best_model.pth" ^
    --output_dir "C:\\IITM\\vitalDB\\data\\VanillaPPG_Batch1_500Hz\\results" ^
    --sampling_rate 500 --target_sr 100 --distance_multiplier 0.8
"""

import os
import sys
import re
import argparse
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from scipy.signal import butter, filtfilt, savgol_filter, resample

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from training.resnet34_glucose_predictor import ResidualBlock1D


# ============================================================
# Filename parser
# ============================================================
def parse_filename(filename):
    """
    Parse lab PPG filename into metadata dict.

    Expected: Name_DD_MM_YYYY_HH_MM_Gluc_G_Sys_S_Dia_D_HR_H[...].csv
    Returns dict with keys: name, date, time, glucose, systolic, diastolic, hr, folder_name
    """
    base = filename
    # Strip trailing _col2.csv or _col2 and .csv
    base = re.sub(r'_col2\.csv$', '', base, flags=re.IGNORECASE)
    base = re.sub(r'\.csv$', '', base, flags=re.IGNORECASE)

    # Normalize separators: replace '-' with '_' so both batch formats work
    # Batch 1: Name_DD_MM_YYYY_HH_MM_Gluc_G_Sys_S_Dia_D_HR_H
    # Batch 2: Name-DD-MM-YYYY-HH-MM-Gluc-G-Sys-S-Dia-D-HR-H
    base_normalized = base.replace('-', '_')

    # Extract known fields using regex
    # Name is everything before the first date-like pattern DD_MM_YYYY
    m = re.match(
        r'^(?P<name>[A-Za-z]+)_'
        r'(?P<day>\d{1,2})_(?P<month>\d{1,2})_(?P<year>\d{4})_'
        r'(?P<hour>\d{1,2})_(?P<minute>\d{1,2})_'
        r'Gluc_(?P<glucose>\d+)_'
        r'Sys_(?P<systolic>\d+)_'
        r'Dia_(?P<diastolic>\d+)_'
        r'HR_(?P<hr>\d+)',
        base_normalized
    )

    if not m:
        # Fallback: use filename stem as folder name
        folder = re.sub(r'[^\w\-]', '_', base)
        return {
            'name': 'Unknown',
            'date': 'unknown',
            'time': 'unknown',
            'glucose': None,
            'systolic': None,
            'diastolic': None,
            'hr': None,
            'folder_name': folder,
            'raw_filename': filename,
        }

    d = m.groupdict()
    name = d['name'].capitalize()
    date_str = f"{d['day'].zfill(2)}_{d['month'].zfill(2)}_{d['year']}"
    time_str = f"{d['hour'].zfill(2)}_{d['minute'].zfill(2)}"
    folder_name = f"{name}_{date_str}_{time_str}_Gluc_{d['glucose']}"

    return {
        'name': name,
        'date': date_str,
        'time': time_str,
        'glucose': int(d['glucose']),
        'systolic': int(d['systolic']),
        'diastolic': int(d['diastolic']),
        'hr': int(d['hr']),
        'folder_name': folder_name,
        'raw_filename': filename,
    }


# ============================================================
# Pipeline functions (same as run_lab_pipeline_app.py)
# ============================================================
def detect_peaks(ppg_signal, height_threshold=20, distance_threshold=None, fs=100):
    if distance_threshold is None:
        distance_threshold = 0.8 * fs
    peaks = []
    for i in range(1, len(ppg_signal) - 1):
        if ppg_signal[i - 1] < ppg_signal[i] > ppg_signal[i + 1]:
            if ppg_signal[i] > height_threshold:
                if len(peaks) == 0 or (i - peaks[-1]) > distance_threshold:
                    peaks.append(i)
    return peaks


def count_peaks(window, height_threshold=None):
    if len(window) < 3:
        return 0
    if height_threshold is None:
        height_threshold = np.median(window)
    count = 0
    for i in range(1, len(window) - 1):
        if window[i - 1] < window[i] > window[i + 1] and window[i] > height_threshold:
            count += 1
    return count


def extract_windows(ppg_signal, peaks, window_size, skip_single_peak_check=False):
    windows = []
    for peak in peaks:
        window_start = max(0, peak - window_size // 2)
        window_end = min(len(ppg_signal), peak + window_size // 2)
        window = ppg_signal[window_start:window_end]
        if skip_single_peak_check or count_peaks(window) == 1:
            windows.append(window)
    return windows


def compute_template(windows):
    if not windows:
        return np.array([])
    lengths = [len(w) for w in windows]
    most_common_length = max(set(lengths), key=lengths.count)
    filtered_windows = [w for w in windows if len(w) == most_common_length]
    if not filtered_windows:
        return np.array([])
    return np.mean(np.stack(filtered_windows, axis=0), axis=0)


def cosine_similarity(window, template):
    if len(window) != len(template):
        min_len = min(len(window), len(template))
        window = window[:min_len]
        template = template[:min_len]
    dot_product = np.sum(window * template)
    mag_w = np.sqrt(np.sum(window ** 2))
    mag_t = np.sqrt(np.sum(template ** 2))
    if mag_w == 0 or mag_t == 0:
        return 0.0
    return dot_product / (mag_w * mag_t)


def filter_windows_by_similarity(windows, template, similarity_threshold=0.85):
    accepted, rejected, similarities = [], [], []
    for w in windows:
        sim = cosine_similarity(w, template)
        similarities.append(sim)
        if sim >= similarity_threshold:
            accepted.append(w)
        else:
            rejected.append(w)
    return accepted, rejected, similarities


# ============================================================
# Full pipeline for a single file
# ============================================================
def process_single_file(csv_path, sampling_rate, target_sr,
                        height_multiplier, distance_multiplier,
                        similarity_threshold, skip_single_peak_check):
    """
    Run full pipeline on a single CSV file.
    Returns (filtered_windows, stats_dict) or (None, error_string).
    """
    stats = {}

    # Step 1: Load
    df = pd.read_csv(csv_path)
    if 'time' not in df.columns or 'ppg' not in df.columns:
        return None, f"Missing 'time'/'ppg' columns. Found: {list(df.columns)}"

    raw_time = df['time'].values.astype(float)
    raw_signal = df['ppg'].values.astype(float)
    stats['raw_samples'] = len(raw_signal)

    # Step 2: Cleanse (remove NaN)
    valid = ~(np.isnan(raw_time) | np.isnan(raw_signal))
    clean_time = raw_time[valid]
    clean_signal = raw_signal[valid]
    stats['clean_samples'] = len(clean_signal)
    stats['nan_dropped'] = int((~valid).sum())

    if len(clean_signal) < 100:
        return None, f"Too few valid samples after cleansing: {len(clean_signal)}"

    # Verify sampling rate
    median_dt = float(np.median(np.diff(clean_time)))
    est_sr = 1.0 / median_dt if median_dt > 0 else sampling_rate
    sr = sampling_rate  # trust user-provided

    # Step 3: Downsample
    if sr > target_sr:
        n_out = int(len(clean_signal) * target_sr / sr)
        ds_signal = resample(clean_signal, n_out)
        ds_time = np.linspace(clean_time[0], clean_time[-1], n_out)
        effective_sr = target_sr
        stats['downsampled'] = True
        stats['downsample_ratio'] = f"{sr:.0f} -> {target_sr:.0f} Hz"
    else:
        ds_signal = clean_signal.copy()
        ds_time = clean_time.copy()
        effective_sr = sr
        stats['downsampled'] = False

    stats['ds_samples'] = len(ds_signal)

    # Step 4: Preprocess
    # 4a: DC removal
    dc_mean = float(np.mean(ds_signal))
    signal_dc = ds_signal - dc_mean

    # 4b: Bandpass filter (0.5 - 10 Hz, Butterworth order 4)
    nyquist = effective_sr / 2
    low_cut = 0.5 / nyquist
    high_cut = min(10.0 / nyquist, 0.99)
    b, a = butter(4, [low_cut, high_cut], btype='band')
    signal_bp = filtfilt(b, a, signal_dc)

    # 4c: Savitzky-Golay smoothing
    wl = int(effective_sr * 0.05)
    if wl % 2 == 0:
        wl += 1
    if wl < 5:
        wl = 5
    try:
        signal_sg = savgol_filter(signal_bp, wl, 3)
    except Exception:
        signal_sg = signal_bp

    preprocessed = signal_sg

    # Step 5: Peak detection
    sig_mean = float(np.mean(preprocessed))
    sig_std = float(np.std(preprocessed))
    h_thresh = sig_mean + height_multiplier * sig_std
    d_thresh = distance_multiplier * effective_sr

    peaks = detect_peaks(preprocessed, height_threshold=h_thresh,
                         distance_threshold=d_thresh, fs=effective_sr)
    stats['num_peaks'] = len(peaks)

    if len(peaks) < 2:
        return None, f"Only {len(peaks)} peaks detected - insufficient"

    # Estimate HR
    intervals = np.diff(np.array(peaks)) / effective_sr
    stats['est_hr'] = float(60.0 / np.mean(intervals)) if np.mean(intervals) > 0 else 0

    # Step 6: Window extraction (1-second windows)
    window_size = int(effective_sr * 1.0)
    all_windows = extract_windows(preprocessed, peaks, window_size,
                                  skip_single_peak_check=skip_single_peak_check)
    stats['num_windows_extracted'] = len(all_windows)
    stats['windows_rejected_single_peak'] = len(peaks) - len(all_windows)

    if len(all_windows) < 1:
        return None, "No windows extracted after peak-based filtering"

    # Step 7: Template computation
    template = compute_template(all_windows)
    if len(template) == 0:
        return None, "Template computation failed"

    # Step 8: Similarity-based filtering
    accepted, rejected, similarities = filter_windows_by_similarity(
        all_windows, template, similarity_threshold
    )
    stats['num_accepted'] = len(accepted)
    stats['num_rejected_similarity'] = len(rejected)
    stats['mean_similarity'] = float(np.mean(similarities)) if similarities else 0

    if len(accepted) < 1:
        return None, "No windows passed similarity filter"

    return accepted, stats


def save_ppg_windows(windows, output_path):
    """Save filtered windows to ppg_windows.csv in long format."""
    rows = []
    for wi, window in enumerate(windows):
        for si, amp in enumerate(window):
            rows.append({
                'window_index': wi,
                'sample_index': si,
                'amplitude': float(amp),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return len(df)


# ============================================================
# Model builder (same as inference_lab_ppg.py)
# ============================================================
def build_model_from_checkpoint(state_dict):
    conv1_kernel = state_dict['conv1.weight'].shape[2]
    has_multi_fc = 'fc1.weight' in state_dict

    class Flex(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=conv1_kernel,
                                   stride=2, padding=conv1_kernel // 2, bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 64, 3, stride=1)
            self.layer2 = self._make_layer(64, 128, 4, stride=2)
            self.layer3 = self._make_layer(128, 256, 6, stride=2)
            self.layer4 = self._make_layer(256, 512, 3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            if has_multi_fc:
                self.fc1 = nn.Linear(512, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc_out = nn.Linear(128, 1)
                self._multi = True
            else:
                self.fc = nn.Linear(512, 1)
                self._multi = False

        def _make_layer(self, inc, outc, blocks, stride=1):
            ds = None
            if stride != 1 or inc != outc:
                ds = nn.Sequential(nn.Conv1d(inc, outc, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(outc))
            layers = [ResidualBlock1D(inc, outc, stride=stride, downsample=ds)]
            for _ in range(1, blocks):
                layers.append(ResidualBlock1D(outc, outc))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self._multi:
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc_out(x)
            return self.fc(x)

    return Flex()


def run_inference(model, device, norm_params, ppg_windows_path, output_path):
    """
    Load ppg_windows.csv, normalize, run inference, save glucose_predictions.csv.
    Returns (predictions_array, stats_dict) or (None, error_string).
    """
    # Load windows
    ppg_df = pd.read_csv(ppg_windows_path)
    windows = []
    for wi in sorted(ppg_df['window_index'].unique()):
        w = ppg_df[ppg_df['window_index'] == wi].sort_values('sample_index')['amplitude'].values
        windows.append(w)

    if not windows:
        return None, "No windows in ppg_windows.csv"

    # Align lengths
    lens = [len(w) for w in windows]
    target_len = Counter(lens).most_common(1)[0][0]
    aligned = []
    for w in windows:
        if len(w) == target_len:
            aligned.append(w)
        elif len(w) > target_len:
            aligned.append(w[:target_len])
        else:
            padded = np.zeros(target_len)
            padded[:len(w)] = w
            aligned.append(padded)

    ppg_data = np.array(aligned, dtype=np.float32)

    # Normalize PPG
    ppg_norm_type = norm_params.get('ppg_normalization', 'per_window')
    if ppg_norm_type == 'global' and norm_params.get('ppg_mean') is not None:
        ppg_mean = norm_params['ppg_mean']
        ppg_std = norm_params['ppg_std']
        ppg_data = (ppg_data - ppg_mean) / ppg_std
    else:
        m = np.mean(ppg_data, axis=1, keepdims=True)
        s = np.std(ppg_data, axis=1, keepdims=True)
        s[s == 0] = 1.0
        ppg_data = (ppg_data - m) / s

    glucose_mean = norm_params.get('glucose_mean', 0.0)
    glucose_std = norm_params.get('glucose_std', 1.0)

    # Inference
    model.eval()
    all_preds = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(ppg_data), batch_size):
            batch = torch.tensor(ppg_data[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(device)
            preds = model(batch).cpu().numpy().flatten()
            all_preds.extend(preds)

    # Denormalize
    predictions = np.array(all_preds) * glucose_std + glucose_mean

    # Save
    pred_df = pd.DataFrame({
        'window_index': range(len(predictions)),
        'predicted_glucose_mg_dl': predictions,
    })
    pred_df.to_csv(output_path, index=False)

    stats = {
        'num_windows': len(predictions),
        'pred_mean': float(np.mean(predictions)),
        'pred_std': float(np.std(predictions)),
        'pred_median': float(np.median(predictions)),
        'pred_min': float(np.min(predictions)),
        'pred_max': float(np.max(predictions)),
    }
    return predictions, stats


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Batch Lab PPG Pipeline: process all CSV files and run inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input_dir', required=True,
                        help='Folder containing raw PPG CSV files')
    parser.add_argument('--model_path', required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--output_dir', required=True,
                        help='Root output folder for results')
    parser.add_argument('--sampling_rate', type=float, default=500,
                        help='Original sampling rate in Hz (default: 500)')
    parser.add_argument('--target_sr', type=float, default=100,
                        help='Target sampling rate after downsampling (default: 100)')
    parser.add_argument('--height_multiplier', type=float, default=0.3,
                        help='Peak detection height multiplier (default: 0.3)')
    parser.add_argument('--distance_multiplier', type=float, default=0.8,
                        help='Peak detection distance multiplier (default: 0.8)')
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                        help='Template similarity threshold (default: 0.85)')
    parser.add_argument('--skip_single_peak_check', action='store_true',
                        help='Skip the single-peak-per-window filter')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    # Discover CSV files
    csv_files = sorted([f for f in input_dir.iterdir()
                        if f.suffix.lower() == '.csv'])
    if not csv_files:
        print(f"ERROR: No CSV files found in {input_dir}")
        sys.exit(1)

    print("=" * 80)
    print("BATCH LAB PPG PIPELINE")
    print("=" * 80)
    print(f"Input folder:  {input_dir}")
    print(f"Model:         {model_path}")
    print(f"Output folder: {output_dir}")
    print(f"Files found:   {len(csv_files)}")
    print(f"Sampling rate: {args.sampling_rate} Hz -> {args.target_sr} Hz")
    print(f"Peak params:   height_mult={args.height_multiplier}, "
          f"distance_mult={args.distance_multiplier}")
    print(f"Similarity:    {args.similarity_threshold}")
    print(f"Skip single-peak check: {args.skip_single_peak_check}")
    print("=" * 80)

    # Load model once
    print("\nLoading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    norm_params = checkpoint.get('normalization', {})
    model = build_model_from_checkpoint(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')}), device={device}")
    print(f"Normalization: {norm_params}")

    # Process each file
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    success = 0
    failed = 0

    for idx, csv_path in enumerate(csv_files, 1):
        print(f"\n{'-' * 70}")
        print(f"[{idx}/{len(csv_files)}] {csv_path.name}")
        print(f"{'-' * 70}")

        meta = parse_filename(csv_path.name)
        folder_name = meta['folder_name']
        sample_dir = output_dir / meta['name'] / folder_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Person: {meta['name']}, Date: {meta['date']}, Time: {meta['time']}")
        if meta['glucose'] is not None:
            print(f"  Reference glucose: {meta['glucose']} mg/dL")
        print(f"  Output: {sample_dir}")

        # Run pipeline
        try:
            windows, pipe_result = process_single_file(
                csv_path=str(csv_path),
                sampling_rate=args.sampling_rate,
                target_sr=args.target_sr,
                height_multiplier=args.height_multiplier,
                distance_multiplier=args.distance_multiplier,
                similarity_threshold=args.similarity_threshold,
                skip_single_peak_check=args.skip_single_peak_check,
            )

            if windows is None:
                print(f"  FAILED: {pipe_result}")
                results.append({**meta, 'status': 'FAILED', 'error': pipe_result})
                failed += 1
                continue

            stats = pipe_result
            print(f"  Peaks: {stats['num_peaks']}, "
                  f"Windows: {stats['num_windows_extracted']}, "
                  f"Accepted: {stats['num_accepted']}, "
                  f"HR: {stats['est_hr']:.0f} bpm")

            # Save ppg_windows.csv
            ppg_path = sample_dir / 'ppg_windows.csv'
            save_ppg_windows(windows, str(ppg_path))
            print(f"  Saved ppg_windows.csv ({stats['num_accepted']} windows)")

            # Run inference
            pred_path = sample_dir / 'glucose_predictions.csv'
            predictions, inf_result = run_inference(
                model, device, norm_params, str(ppg_path), str(pred_path)
            )

            if predictions is None:
                print(f"  Inference FAILED: {inf_result}")
                results.append({**meta, 'status': 'PARTIAL', 'error': inf_result, **stats})
                failed += 1
                continue

            inf_stats = inf_result
            print(f"  Predictions: mean={inf_stats['pred_mean']:.1f}, "
                  f"std={inf_stats['pred_std']:.1f}, "
                  f"median={inf_stats['pred_median']:.1f} mg/dL")

            if meta['glucose'] is not None:
                error = inf_stats['pred_mean'] - meta['glucose']
                abs_error = abs(error)
                rel_error = abs_error / meta['glucose'] * 100
                print(f"  vs Reference {meta['glucose']}: "
                      f"error={error:+.1f}, abs_error={abs_error:.1f}, "
                      f"MARD={rel_error:.1f}%")
            else:
                error = None
                abs_error = None
                rel_error = None

            results.append({
                **meta,
                'status': 'OK',
                'error': None,
                **stats,
                **inf_stats,
                'ref_glucose': meta['glucose'],
                'pred_error': error,
                'pred_abs_error': abs_error,
                'pred_mard': rel_error,
            })
            success += 1

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
            results.append({**meta, 'status': 'ERROR', 'error': str(e)})
            failed += 1

    # ---- Summary ----
    print(f"\n{'=' * 80}")
    print(f"BATCH COMPLETE: {success} succeeded, {failed} failed out of {len(csv_files)}")
    print(f"{'=' * 80}")

    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_path = output_dir / 'batch_summary.csv'
    summary_df.to_csv(str(summary_path), index=False)
    print(f"Summary saved: {summary_path}")

    # Print summary table
    ok_results = [r for r in results if r['status'] == 'OK']
    if ok_results:
        print(f"\n{'Person':<12} {'Date':<12} {'Time':<6} "
              f"{'Ref':>4} {'Pred':>6} {'Err':>6} {'MARD':>6} {'Win':>4}")
        print("-" * 70)
        for r in ok_results:
            ref = r.get('ref_glucose', '')
            pred = r.get('pred_mean', 0)
            err = r.get('pred_error')
            mard = r.get('pred_mard')
            nw = r.get('num_accepted', 0)
            err_str = f"{err:+.1f}" if err is not None else "N/A"
            mard_str = f"{mard:.1f}%" if mard is not None else "N/A"
            ref_str = str(ref) if ref is not None else "?"
            print(f"{r['name']:<12} {r['date']:<12} {r['time']:<6} "
                  f"{ref_str:>4} {pred:>6.1f} {err_str:>6} {mard_str:>6} {nw:>4}")

        # Overall stats
        if any(r.get('pred_abs_error') is not None for r in ok_results):
            valid = [r for r in ok_results if r.get('pred_abs_error') is not None]
            avg_mae = np.mean([r['pred_abs_error'] for r in valid])
            avg_mard = np.mean([r['pred_mard'] for r in valid])
            print(f"\nOverall: MAE={avg_mae:.1f} mg/dL, MARD={avg_mard:.1f}% "
                  f"({len(valid)} samples with reference glucose)")


if __name__ == '__main__':
    main()
