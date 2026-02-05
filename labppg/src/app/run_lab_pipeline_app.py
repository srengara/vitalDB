#!/usr/bin/env python
"""
Lab PPG Pipeline Visualization Web App
========================================
Interactive step-by-step visualization of the PPG processing pipeline.

Upload a raw PPG CSV (time, ppg columns) and watch each processing step:
1. Load & Validate
2. Data Cleansing
3. Downsampling
4. Signal Preprocessing (DC removal, Bandpass, Smoothing)
5. Peak Detection
6. Window Extraction
7. Template Computation
8. Template-based Filtering
9. Save Output & Run Inference

Usage:
    python run_lab_pipeline_app.py [--port 5002]
"""

import os
import sys
import time as time_module
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, render_template_string, request, jsonify, send_file
from scipy.signal import butter, filtfilt, savgol_filter, resample

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
session_data = {}

# ---------------------------------------------------------------------------
# Pipeline functions (inlined from generate_lab_training_data.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plot helpers (Base64-encoded PNG)
# ---------------------------------------------------------------------------
PLOT_STYLE = dict(figsize=(12, 4), dpi=100)


def _fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _style_ax(ax, title, xlabel='Time (s)', ylabel='Amplitude'):
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, color='#bdc3c7', fontsize=10)
    ax.set_ylabel(ylabel, color='#bdc3c7', fontsize=10)
    ax.tick_params(colors='#bdc3c7')
    ax.grid(True, alpha=0.2, color='#555')
    for spine in ax.spines.values():
        spine.set_color('#333')


def plot_signal(time_arr, signal, title, max_seconds=30, color='#667eea'):
    fig, ax = plt.subplots(**PLOT_STYLE)
    fig.set_facecolor('#16213e')
    mask = time_arr <= time_arr[0] + max_seconds
    ax.plot(time_arr[mask], signal[mask], color=color, linewidth=0.6)
    _style_ax(ax, title)
    return _fig_to_base64(fig)


def plot_overlay(time_arr, before, after, title, label_before='Before', label_after='After',
                 color_before='#555', color_after='#2ecc71', max_seconds=10):
    fig, ax = plt.subplots(**PLOT_STYLE)
    fig.set_facecolor('#16213e')
    mask = time_arr <= time_arr[0] + max_seconds
    ax.plot(time_arr[mask], before[mask], color=color_before, linewidth=0.6,
            alpha=0.5, label=label_before, linestyle='--')
    ax.plot(time_arr[mask], after[mask], color=color_after, linewidth=0.6,
            label=label_after)
    _style_ax(ax, title)
    ax.legend(facecolor='#1a1a2e', edgecolor='#555', labelcolor='white', fontsize=9)
    return _fig_to_base64(fig)


def plot_peaks(time_arr, signal, peak_indices, title, max_seconds=30):
    fig, ax = plt.subplots(**PLOT_STYLE)
    fig.set_facecolor('#16213e')
    mask = time_arr <= time_arr[0] + max_seconds
    ax.plot(time_arr[mask], signal[mask], color='#667eea', linewidth=0.6)
    # draw peaks within range
    for pi in peak_indices:
        if pi < len(time_arr) and time_arr[pi] <= time_arr[0] + max_seconds:
            ax.plot(time_arr[pi], signal[pi], 'o', color='#e74c3c', markersize=4)
    peak_count_in_view = sum(1 for pi in peak_indices if pi < len(time_arr) and time_arr[pi] <= time_arr[0] + max_seconds)
    _style_ax(ax, f'{title}  ({peak_count_in_view} peaks shown in first {max_seconds}s)')
    return _fig_to_base64(fig)


def plot_windows_overlay(windows, sr, title, color='#667eea', max_windows=20):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.set_facecolor('#16213e')
    n = min(len(windows), max_windows)
    for i in range(n):
        t = np.arange(len(windows[i])) / sr
        ax.plot(t, windows[i], color=color, alpha=0.3, linewidth=0.7)
    _style_ax(ax, f'{title}  (showing {n} of {len(windows)})')
    return _fig_to_base64(fig)


def plot_template(template, sr):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.set_facecolor('#16213e')
    t = np.arange(len(template)) / sr
    ax.plot(t, template, color='#2ecc71', linewidth=2)
    ax.fill_between(t, template, alpha=0.25, color='#2ecc71')
    _style_ax(ax, 'Mean Template Waveform')
    return _fig_to_base64(fig)


def plot_accepted_rejected(accepted, rejected, template, sr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), dpi=100)
    fig.set_facecolor('#16213e')
    t_template = np.arange(len(template)) / sr

    n_a = min(len(accepted), 20)
    for i in range(n_a):
        t_w = np.arange(len(accepted[i])) / sr
        ax1.plot(t_w, accepted[i], color='#2ecc71', alpha=0.25, linewidth=0.7)
    ax1.plot(t_template, template, color='#e74c3c', linewidth=1.5, label='Template')
    _style_ax(ax1, f'Accepted ({len(accepted)})')
    ax1.legend(facecolor='#1a1a2e', edgecolor='#555', labelcolor='white', fontsize=9)

    n_r = min(len(rejected), 20)
    for i in range(n_r):
        t_w = np.arange(len(rejected[i])) / sr
        ax2.plot(t_w, rejected[i], color='#95a5a6', alpha=0.4, linewidth=0.7)
    ax2.plot(t_template, template, color='#e74c3c', linewidth=1.5, linestyle='--', label='Template')
    _style_ax(ax2, f'Rejected ({len(rejected)})')
    ax2.legend(facecolor='#1a1a2e', edgecolor='#555', labelcolor='white', fontsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_similarity_histogram(similarities, threshold):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    fig.set_facecolor('#16213e')
    sims = np.array(similarities)
    accepted = sims[sims >= threshold]
    rejected_s = sims[sims < threshold]
    if len(accepted):
        ax.hist(accepted, bins=30, color='#2ecc71', alpha=0.7, label=f'Accepted ({len(accepted)})')
    if len(rejected_s):
        ax.hist(rejected_s, bins=30, color='#95a5a6', alpha=0.7, label=f'Rejected ({len(rejected_s)})')
    ax.axvline(threshold, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold})')
    _style_ax(ax, 'Cosine Similarity Distribution', xlabel='Similarity', ylabel='Count')
    ax.legend(facecolor='#1a1a2e', edgecolor='#555', labelcolor='white', fontsize=9)
    return _fig_to_base64(fig)


def plot_predictions(predictions):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    fig.set_facecolor('#16213e')
    x = np.arange(len(predictions))
    ax.scatter(x, predictions, c='#667eea', s=8, alpha=0.6)
    # clinical zones
    ax.axhspan(0, 70, alpha=0.08, color='#e74c3c', label='Hypoglycemia')
    ax.axhspan(70, 100, alpha=0.08, color='#2ecc71', label='Normal')
    ax.axhspan(100, 125, alpha=0.08, color='#f39c12', label='Prediabetes')
    ax.axhspan(125, 400, alpha=0.08, color='#e74c3c', label='Diabetes')
    ax.axhline(70, color='#e74c3c', alpha=0.3, linewidth=0.5)
    ax.axhline(100, color='#2ecc71', alpha=0.3, linewidth=0.5)
    ax.axhline(125, color='#f39c12', alpha=0.3, linewidth=0.5)
    _style_ax(ax, 'Predicted Glucose per Window', xlabel='Window Index', ylabel='Glucose (mg/dL)')
    ax.legend(facecolor='#1a1a2e', edgecolor='#555', labelcolor='white', fontsize=8, loc='upper right')
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB


# ---- Routes ---------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/upload', methods=['POST'])
def upload_csv():
    try:
        f = request.files.get('csv_file')
        if f is None:
            return jsonify({'error': 'No file uploaded'}), 400

        sampling_rate = float(request.form.get('sampling_rate', 100))
        target_sr = float(request.form.get('target_sampling_rate', 100))
        height_mult = float(request.form.get('height_multiplier', 0.3))
        distance_mult = float(request.form.get('distance_multiplier', 0.8))
        sim_thresh = float(request.form.get('similarity_threshold', 0.85))

        df = pd.read_csv(f)
        if 'time' not in df.columns or 'ppg' not in df.columns:
            return jsonify({'error': "CSV must contain 'time' and 'ppg' columns. Found: " + str(list(df.columns))}), 400

        sid = f'lab_{int(time_module.time()*1000)}'
        raw_time = df['time'].values.astype(float)
        raw_signal = df['ppg'].values.astype(float)

        nan_time = int(np.isnan(raw_time).sum())
        nan_ppg = int(np.isnan(raw_signal).sum())

        session_data[sid] = {
            'config': {
                'sampling_rate': sampling_rate,
                'target_sampling_rate': target_sr,
                'height_multiplier': height_mult,
                'distance_multiplier': distance_mult,
                'similarity_threshold': sim_thresh,
            },
            'raw_time': raw_time,
            'raw_signal': raw_signal,
        }

        # Plot raw signal (use indices as time if time has NaNs)
        valid = ~(np.isnan(raw_time) | np.isnan(raw_signal))
        plot_t = raw_time[valid]
        plot_s = raw_signal[valid]
        p = plot_signal(plot_t, plot_s, 'Step 1: Raw PPG Signal')

        return jsonify({
            'session_id': sid,
            'total_samples': len(df),
            'duration_seconds': float(plot_t[-1] - plot_t[0]) if len(plot_t) > 1 else 0,
            'nan_count_time': nan_time,
            'nan_count_ppg': nan_ppg,
            'sampling_rate': sampling_rate,
            'plot_raw': p,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cleanse', methods=['POST'])
def cleanse_data():
    try:
        sid = request.json['session_id']
        s = session_data[sid]
        raw_time = s['raw_time']
        raw_signal = s['raw_signal']

        valid = ~(np.isnan(raw_time) | np.isnan(raw_signal))
        clean_time = raw_time[valid]
        clean_signal = raw_signal[valid]

        sr = s['config']['sampling_rate']
        est_sr = None
        sr_warn = None
        if len(clean_time) > 1:
            median_interval = float(np.median(np.diff(clean_time)))
            if median_interval > 0:
                est_sr = 1.0 / median_interval
                if abs(est_sr - sr) > sr * 0.1:
                    sr_warn = f'Detected SR ~{est_sr:.1f} Hz differs from declared {sr} Hz'

        s['clean_time'] = clean_time
        s['clean_signal'] = clean_signal

        p = plot_signal(clean_time, clean_signal, 'Step 2: Cleansed Signal')

        return jsonify({
            'original_samples': len(raw_time),
            'cleansed_samples': len(clean_time),
            'dropped_rows': int(len(raw_time) - len(clean_time)),
            'estimated_sr': round(est_sr, 1) if est_sr else None,
            'sr_mismatch_warning': sr_warn,
            'duration_seconds': float(clean_time[-1] - clean_time[0]) if len(clean_time) > 1 else 0,
            'plot_cleansed': p,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/downsample', methods=['POST'])
def downsample_signal():
    try:
        sid = request.json['session_id']
        s = session_data[sid]
        cfg = s['config']
        sr = cfg['sampling_rate']
        target_sr = cfg['target_sampling_rate']

        clean_time = s['clean_time']
        clean_signal = s['clean_signal']

        if sr > target_sr:
            n_out = int(len(clean_signal) * target_sr / sr)
            ds_signal = resample(clean_signal, n_out)
            ds_time = np.linspace(clean_time[0], clean_time[-1], n_out)
            s['ds_time'] = ds_time
            s['ds_signal'] = ds_signal
            s['effective_sr'] = target_sr

            p = plot_overlay(clean_time, clean_signal, np.interp(clean_time, ds_time, ds_signal),
                             'Step 3: Downsampling', f'Original ({sr:.0f} Hz)',
                             f'Downsampled ({target_sr:.0f} Hz)',
                             max_seconds=5)

            return jsonify({
                'downsampled': True,
                'original_sr': sr,
                'target_sr': target_sr,
                'original_samples': len(clean_signal),
                'downsampled_samples': len(ds_signal),
                'downsample_factor': round(sr / target_sr, 1),
                'plot_downsampled': p,
            })
        else:
            s['ds_time'] = clean_time.copy()
            s['ds_signal'] = clean_signal.copy()
            s['effective_sr'] = sr

            p = plot_signal(clean_time, clean_signal,
                            f'Step 3: No Downsampling Needed ({sr:.0f} Hz <= {target_sr:.0f} Hz)')

            return jsonify({
                'downsampled': False,
                'original_sr': sr,
                'target_sr': target_sr,
                'original_samples': len(clean_signal),
                'downsampled_samples': len(clean_signal),
                'downsample_factor': 1,
                'plot_downsampled': p,
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess_signal_route():
    try:
        sid = request.json['session_id']
        s = session_data[sid]
        sr = s['effective_sr']
        time_arr = s['ds_time']
        signal = s['ds_signal']

        # Sub-step 4a: DC removal
        dc_mean = float(np.mean(signal))
        signal_dc = signal - dc_mean

        # Sub-step 4b: Bandpass filter (0.5-10 Hz)
        nyquist = sr / 2
        low_cut = 0.5 / nyquist
        high_cut = min(10.0 / nyquist, 0.99)
        try:
            b, a = butter(4, [low_cut, high_cut], btype='band')
            signal_bp = filtfilt(b, a, signal_dc)
        except Exception:
            signal_bp = signal_dc

        # Sub-step 4c: Savitzky-Golay smoothing
        wl = int(sr * 0.05)
        if wl % 2 == 0:
            wl += 1
        if wl < 5:
            wl = 5
        try:
            signal_sg = savgol_filter(signal_bp, wl, 3)
        except Exception:
            signal_sg = signal_bp

        s['signal_dc'] = signal_dc
        s['signal_bp'] = signal_bp
        s['signal_sg'] = signal_sg

        p1 = plot_overlay(time_arr, signal, signal_dc,
                          'Step 4a: DC Component Removal', 'Raw', 'DC Removed',
                          color_after='#3498db')
        p2 = plot_overlay(time_arr, signal_dc, signal_bp,
                          'Step 4b: Butterworth Bandpass (0.5-10 Hz)', 'DC Removed', 'Bandpass Filtered',
                          color_after='#2ecc71')
        p3 = plot_overlay(time_arr, signal_bp, signal_sg,
                          'Step 4c: Savitzky-Golay Smoothing', 'Filtered', 'Smoothed',
                          color_after='#f39c12')

        return jsonify({
            'dc_mean_removed': round(dc_mean, 4),
            'filter_low': 0.5,
            'filter_high': 10.0,
            'filter_order': 4,
            'savgol_window_ms': 50,
            'savgol_window_samples': wl,
            'savgol_order': 3,
            'signal_stats': {
                'mean': round(float(np.mean(signal_sg)), 4),
                'std': round(float(np.std(signal_sg)), 4),
                'min': round(float(np.min(signal_sg)), 4),
                'max': round(float(np.max(signal_sg)), 4),
            },
            'plot_dc_removal': p1,
            'plot_bandpass': p2,
            'plot_smoothing': p3,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect_peaks', methods=['POST'])
def detect_peaks_route():
    try:
        sid = request.json['session_id']
        s = session_data[sid]
        cfg = s['config']
        sr = s['effective_sr']
        signal = s['signal_sg']
        time_arr = s['ds_time']

        sig_mean = float(np.mean(signal))
        sig_std = float(np.std(signal))
        h_thresh = sig_mean + cfg['height_multiplier'] * sig_std
        d_thresh = cfg['distance_multiplier'] * sr

        peaks = detect_peaks(signal, height_threshold=h_thresh,
                             distance_threshold=d_thresh, fs=sr)
        s['peaks'] = peaks

        mean_interval = None
        hr = None
        if len(peaks) > 1:
            intervals = np.diff(np.array(peaks)) / sr
            mean_interval = float(np.mean(intervals))
            hr = 60.0 / mean_interval if mean_interval > 0 else None

        p = plot_peaks(time_arr, signal, peaks, 'Step 5: Peak Detection')

        return jsonify({
            'num_peaks': len(peaks),
            'height_threshold': round(h_thresh, 4),
            'distance_threshold': round(d_thresh, 1),
            'mean_peak_interval_sec': round(mean_interval, 3) if mean_interval else None,
            'estimated_hr_bpm': round(hr, 1) if hr else None,
            'plot_peaks': p,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract_windows', methods=['POST'])
def extract_windows_route():
    try:
        sid = request.json['session_id']
        s = session_data[sid]
        sr = s['effective_sr']
        signal = s['signal_sg']
        peaks = s['peaks']

        skip_single_peak_check = request.json.get('skip_single_peak_check', False)
        window_size = int(sr * 1.0)  # 1-second windows
        windows = extract_windows(signal, peaks, window_size, skip_single_peak_check=skip_single_peak_check)
        s['all_windows'] = windows
        s['window_size'] = window_size

        p = None
        if windows:
            p = plot_windows_overlay(windows, sr, 'Step 6: Extracted Windows')

        return jsonify({
            'num_windows': len(windows),
            'window_size_samples': window_size,
            'window_duration_sec': 1.0,
            'rejected_count': len(peaks) - len(windows),
            'plot_windows': p,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compute_template', methods=['POST'])
def compute_template_route():
    try:
        sid = request.json['session_id']
        s = session_data[sid]
        sr = s['effective_sr']
        windows = s['all_windows']

        template = compute_template(windows)
        s['template'] = template

        p = plot_template(template, sr)

        return jsonify({
            'template_length': len(template),
            'num_windows_used': len([w for w in windows if len(w) == len(template)]),
            'template_stats': {
                'mean': round(float(np.mean(template)), 4),
                'std': round(float(np.std(template)), 4),
                'min': round(float(np.min(template)), 4),
                'max': round(float(np.max(template)), 4),
            },
            'plot_template': p,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/filter_windows', methods=['POST'])
def filter_windows_route():
    try:
        sid = request.json['session_id']
        s = session_data[sid]
        cfg = s['config']
        sr = s['effective_sr']
        windows = s['all_windows']
        template = s['template']
        thresh = cfg['similarity_threshold']

        accepted, rejected, similarities = filter_windows_by_similarity(windows, template, thresh)
        s['filtered_windows'] = accepted
        s['rejected_windows'] = rejected
        s['similarities'] = similarities

        p1 = plot_accepted_rejected(accepted, rejected, template, sr)
        p2 = plot_similarity_histogram(similarities, thresh)

        sims_arr = np.array(similarities)
        acc_sims = sims_arr[sims_arr >= thresh]
        rej_sims = sims_arr[sims_arr < thresh]

        return jsonify({
            'total_windows': len(windows),
            'accepted_windows': len(accepted),
            'rejected_windows': len(rejected),
            'filtering_rate_pct': round(len(accepted) / len(windows) * 100, 1) if windows else 0,
            'similarity_threshold': thresh,
            'mean_similarity_accepted': round(float(np.mean(acc_sims)), 3) if len(acc_sims) else None,
            'mean_similarity_rejected': round(float(np.mean(rej_sims)), 3) if len(rej_sims) else None,
            'plot_accepted_rejected': p1,
            'plot_similarity_histogram': p2,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_output', methods=['POST'])
def save_output():
    try:
        sid = request.json['session_id']
        s = session_data[sid]
        filtered_windows = s['filtered_windows']

        output_dir = os.path.join('.', 'inference_data')
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for wi, window in enumerate(filtered_windows):
            for si, amp in enumerate(window):
                rows.append({'window_index': wi, 'sample_index': si, 'amplitude': float(amp)})

        ppg_df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, 'ppg_windows.csv')
        ppg_df.to_csv(out_path, index=False)

        s['output_dir'] = output_dir
        s['ppg_windows_file'] = out_path

        return jsonify({
            'file_path': out_path,
            'num_windows': len(filtered_windows),
            'window_length': len(filtered_windows[0]) if filtered_windows else 0,
            'total_rows': len(rows),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<session_id>/ppg_windows')
def download_ppg(session_id):
    try:
        s = session_data[session_id]
        return send_file(s['ppg_windows_file'], as_attachment=True,
                         download_name='ppg_windows.csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<session_id>/predictions')
def download_predictions(session_id):
    try:
        s = session_data[session_id]
        return send_file(s['predictions_file'], as_attachment=True,
                         download_name='predictions.csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run_inference', methods=['POST'])
def run_inference_route():
    try:
        sid = request.json['session_id']
        model_path = request.json.get('model_path', '')
        s = session_data[sid]

        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}'}), 400

        # Import inference function
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from inference_lab_ppg import run_inference

        data_dir = s['output_dir']
        pred_output_dir = os.path.join(data_dir, 'predictions')
        predictions = run_inference(model_path, data_dir,
                                    output_dir=pred_output_dir)

        # Store predictions file path for download
        s['predictions_file'] = os.path.join(pred_output_dir, 'glucose_predictions.csv')

        p = plot_predictions(predictions)

        mean_g = float(np.mean(predictions))
        if mean_g < 70:
            interp = 'Hypoglycemia'
        elif mean_g <= 100:
            interp = 'Normal fasting glucose'
        elif mean_g <= 125:
            interp = 'Prediabetes range'
        else:
            interp = 'Diabetes range'

        return jsonify({
            'num_predictions': len(predictions),
            'mean_glucose_mgdl': round(mean_g, 1),
            'std_glucose_mgdl': round(float(np.std(predictions)), 1),
            'min_glucose_mgdl': round(float(np.min(predictions)), 1),
            'max_glucose_mgdl': round(float(np.max(predictions)), 1),
            'clinical_interpretation': interp,
            'plot_predictions': p,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Lab PPG Processing Pipeline</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#e8e8e8;min-height:100vh}
.container{max-width:1100px;margin:0 auto;padding:20px}
.header{text-align:center;padding:30px 0 20px}
.header h1{font-size:2em;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:5px}
.header p{color:#bdc3c7;font-size:0.95em}
.step{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-left:4px solid #667eea;border-radius:10px;margin:15px 0;padding:20px;transition:opacity 0.3s}
.step.disabled{opacity:0.35;pointer-events:none}
.step-header{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.step-num{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:0.9em;flex-shrink:0}
.step-title{font-size:1.1em;font-weight:600;color:#fff}
.btn{padding:8px 22px;border:none;border-radius:6px;cursor:pointer;font-size:0.9em;font-weight:600;transition:all 0.2s}
.btn-primary{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 15px rgba(102,126,234,0.4)}
.btn-secondary{background:rgba(255,255,255,0.1);color:#fff;border:1px solid rgba(255,255,255,0.2)}
.btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.config-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin:12px 0}
.config-item label{display:block;font-size:0.8em;color:#bdc3c7;margin-bottom:3px}
.config-item input{width:100%;padding:6px 10px;border-radius:5px;border:1px solid rgba(255,255,255,0.15);background:rgba(0,0,0,0.3);color:#fff;font-size:0.9em}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:12px 0}
.stat-card{background:rgba(102,126,234,0.12);border:1px solid rgba(102,126,234,0.25);border-radius:8px;padding:12px;text-align:center}
.stat-label{font-size:0.75em;color:#bdc3c7;margin-bottom:2px}
.stat-value{font-size:1.3em;font-weight:bold;color:#667eea}
.stat-unit{font-size:0.7em;color:#95a5a6}
.status{padding:8px 14px;border-radius:6px;margin:10px 0;font-size:0.85em;display:none}
.status.show{display:block}
.status-success{background:rgba(46,204,113,0.15);border:1px solid rgba(46,204,113,0.3);color:#2ecc71}
.status-error{background:rgba(231,76,60,0.15);border:1px solid rgba(231,76,60,0.3);color:#e74c3c}
.status-info{background:rgba(52,152,219,0.15);border:1px solid rgba(52,152,219,0.3);color:#3498db}
.status-warning{background:rgba(243,156,18,0.15);border:1px solid rgba(243,156,18,0.3);color:#f39c12}
.plot-container{margin:12px 0;text-align:center}
.plot-container img{max-width:100%;border-radius:8px;border:1px solid rgba(255,255,255,0.1)}
.sub-step{margin:10px 0 10px 20px;padding:10px;border-left:2px solid rgba(102,126,234,0.3);border-radius:0 8px 8px 0}
.file-upload{border:2px dashed rgba(102,126,234,0.4);border-radius:10px;padding:20px;text-align:center;cursor:pointer;transition:border-color 0.2s}
.file-upload:hover{border-color:#667eea}
.file-upload input[type=file]{display:none}
.file-upload .file-name{color:#667eea;font-weight:600;margin-top:8px}
.inference-section{margin-top:15px;padding:15px;background:rgba(102,126,234,0.08);border-radius:8px;border:1px solid rgba(102,126,234,0.2)}
.inference-section h4{color:#667eea;margin-bottom:10px}
.spinner{display:none;width:20px;height:20px;border:3px solid rgba(255,255,255,0.2);border-top:3px solid #667eea;border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto}
@keyframes spin{to{transform:rotate(360deg)}}
.hidden{display:none}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Lab PPG Processing Pipeline</h1>
    <p>Step-by-step visualization of PPG signal processing for glucose inference</p>
  </div>

  <!-- Step 1: Upload -->
  <div class="step" id="step1">
    <div class="step-header"><div class="step-num">1</div><div class="step-title">Upload & Configure</div></div>
    <div class="file-upload" onclick="document.getElementById('csvFile').click()">
      <div>Click to select CSV file (columns: time, ppg)</div>
      <input type="file" id="csvFile" accept=".csv" onchange="onFileSelected()">
      <div class="file-name" id="fileName"></div>
    </div>
    <div class="config-grid">
      <div class="config-item"><label>Sampling Rate (Hz)</label><input id="cfgSR" type="number" value="100" step="1"></div>
      <div class="config-item"><label>Target SR (Hz)</label><input id="cfgTargetSR" type="number" value="100" step="1"></div>
      <div class="config-item"><label>Height Multiplier</label><input id="cfgHeight" type="number" value="0.3" step="0.1"></div>
      <div class="config-item"><label>Distance Multiplier</label><input id="cfgDistance" type="number" value="0.8" step="0.1"></div>
      <div class="config-item"><label>Similarity Threshold</label><input id="cfgSimilarity" type="number" value="0.85" step="0.05" min="0" max="1"></div>
    </div>
    <button class="btn btn-primary" id="btnUpload" onclick="uploadCSV()">Upload & Validate</button>
    <div class="spinner" id="spin1"></div>
    <div class="status" id="status1"></div>
    <div class="stats-grid" id="stats1"></div>
    <div class="plot-container" id="plot1"></div>
  </div>

  <!-- Step 2: Cleanse -->
  <div class="step disabled" id="step2">
    <div class="step-header"><div class="step-num">2</div><div class="step-title">Data Cleansing</div></div>
    <button class="btn btn-primary" onclick="runStep('cleanse',2)">Remove NaN & Validate SR</button>
    <div class="spinner" id="spin2"></div>
    <div class="status" id="status2"></div>
    <div class="stats-grid" id="stats2"></div>
    <div class="plot-container" id="plot2"></div>
  </div>

  <!-- Step 3: Downsample -->
  <div class="step disabled" id="step3">
    <div class="step-header"><div class="step-num">3</div><div class="step-title">Downsampling</div></div>
    <button class="btn btn-primary" onclick="runStep('downsample',3)">Downsample Signal</button>
    <div class="spinner" id="spin3"></div>
    <div class="status" id="status3"></div>
    <div class="stats-grid" id="stats3"></div>
    <div class="plot-container" id="plot3"></div>
  </div>

  <!-- Step 4: Preprocess -->
  <div class="step disabled" id="step4">
    <div class="step-header"><div class="step-num">4</div><div class="step-title">Signal Preprocessing</div></div>
    <button class="btn btn-primary" onclick="runStep('preprocess',4)">Run Preprocessing</button>
    <div class="spinner" id="spin4"></div>
    <div class="status" id="status4"></div>
    <div class="stats-grid" id="stats4"></div>
    <div class="sub-step"><b>4a.</b> DC Component Removal<div class="plot-container" id="plot4a"></div></div>
    <div class="sub-step"><b>4b.</b> Butterworth Bandpass Filter<div class="plot-container" id="plot4b"></div></div>
    <div class="sub-step"><b>4c.</b> Savitzky-Golay Smoothing<div class="plot-container" id="plot4c"></div></div>
  </div>

  <!-- Step 5: Peaks -->
  <div class="step disabled" id="step5">
    <div class="step-header"><div class="step-num">5</div><div class="step-title">Peak Detection</div></div>
    <button class="btn btn-primary" onclick="runStep('detect_peaks',5)">Detect Peaks</button>
    <div class="spinner" id="spin5"></div>
    <div class="status" id="status5"></div>
    <div class="stats-grid" id="stats5"></div>
    <div class="plot-container" id="plot5"></div>
  </div>

  <!-- Step 6: Windows -->
  <div class="step disabled" id="step6">
    <div class="step-header"><div class="step-num">6</div><div class="step-title">Window Extraction</div></div>
    <div style="margin:6px 0">
      <label style="color:#ccc;cursor:pointer"><input type="checkbox" id="skipSinglePeakCheck"> Skip single-peak check (keep all windows)</label>
    </div>
    <button class="btn btn-primary" onclick="runStep('extract_windows',6)">Extract Windows</button>
    <div class="spinner" id="spin6"></div>
    <div class="status" id="status6"></div>
    <div class="stats-grid" id="stats6"></div>
    <div class="plot-container" id="plot6"></div>
  </div>

  <!-- Step 7: Template -->
  <div class="step disabled" id="step7">
    <div class="step-header"><div class="step-num">7</div><div class="step-title">Template Computation</div></div>
    <button class="btn btn-primary" onclick="runStep('compute_template',7)">Compute Template</button>
    <div class="spinner" id="spin7"></div>
    <div class="status" id="status7"></div>
    <div class="stats-grid" id="stats7"></div>
    <div class="plot-container" id="plot7"></div>
  </div>

  <!-- Step 8: Filter -->
  <div class="step disabled" id="step8">
    <div class="step-header"><div class="step-num">8</div><div class="step-title">Template-based Filtering</div></div>
    <button class="btn btn-primary" onclick="runStep('filter_windows',8)">Filter by Similarity</button>
    <div class="spinner" id="spin8"></div>
    <div class="status" id="status8"></div>
    <div class="stats-grid" id="stats8"></div>
    <div class="plot-container" id="plot8a"></div>
    <div class="plot-container" id="plot8b"></div>
  </div>

  <!-- Step 9: Save & Inference -->
  <div class="step disabled" id="step9">
    <div class="step-header"><div class="step-num">9</div><div class="step-title">Save Output & Inference</div></div>
    <button class="btn btn-primary" onclick="runSave()">Save ppg_windows.csv</button>
    <div class="spinner" id="spin9"></div>
    <div class="status" id="status9"></div>
    <div class="stats-grid" id="stats9"></div>
    <div id="downloadSection" class="hidden" style="margin:10px 0">
      <button class="btn btn-secondary" onclick="downloadCSV()">Download ppg_windows.csv</button>
    </div>

    <div class="inference-section hidden" id="inferenceSection">
      <h4>Run Glucose Inference</h4>
      <div class="config-grid">
        <div class="config-item" style="grid-column:span 2"><label>Model Path</label>
          <input id="modelPath" type="text" value="model/latest_model_174012/best_model.pth" style="width:100%">
        </div>
      </div>
      <button class="btn btn-primary" onclick="runInference()" style="margin-top:8px">Run Inference</button>
      <div class="spinner" id="spinInf"></div>
      <div class="status" id="statusInf"></div>
      <div class="stats-grid" id="statsInf"></div>
      <div class="plot-container" id="plotInf"></div>
      <div id="downloadPredSection" class="hidden" style="margin:10px 0">
        <button class="btn btn-secondary" onclick="downloadPredictions()">Download predictions.csv</button>
      </div>
    </div>
  </div>
</div>

<script>
let sessionId = null;

function showStatus(n, msg, type) {
  const el = document.getElementById('status' + n);
  el.className = 'status show status-' + type;
  el.textContent = msg;
}

function showSpinner(n, show) {
  document.getElementById('spin' + n).style.display = show ? 'block' : 'none';
}

function enableStep(n) {
  document.getElementById('step' + n).classList.remove('disabled');
}

function renderStats(containerId, stats) {
  const el = document.getElementById(containerId);
  el.innerHTML = stats.map(s =>
    `<div class="stat-card"><div class="stat-label">${s.label}</div><div class="stat-value">${s.value}</div>${s.unit ? '<div class="stat-unit">'+s.unit+'</div>' : ''}</div>`
  ).join('');
}

function showPlot(containerId, imgData) {
  if (!imgData) return;
  document.getElementById(containerId).innerHTML = `<img src="${imgData}">`;
}

function onFileSelected() {
  const f = document.getElementById('csvFile');
  document.getElementById('fileName').textContent = f.files.length ? f.files[0].name : '';
}

async function uploadCSV() {
  const fileInput = document.getElementById('csvFile');
  if (!fileInput.files.length) { showStatus(1, 'Please select a CSV file', 'error'); return; }

  showSpinner(1, true);
  const fd = new FormData();
  fd.append('csv_file', fileInput.files[0]);
  fd.append('sampling_rate', document.getElementById('cfgSR').value);
  fd.append('target_sampling_rate', document.getElementById('cfgTargetSR').value);
  fd.append('height_multiplier', document.getElementById('cfgHeight').value);
  fd.append('distance_multiplier', document.getElementById('cfgDistance').value);
  fd.append('similarity_threshold', document.getElementById('cfgSimilarity').value);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const d = await res.json();
    if (d.error) { showStatus(1, d.error, 'error'); showSpinner(1, false); return; }

    sessionId = d.session_id;
    showStatus(1, 'File loaded successfully', 'success');
    renderStats('stats1', [
      { label: 'Total Samples', value: d.total_samples.toLocaleString() },
      { label: 'Duration', value: d.duration_seconds.toFixed(1), unit: 'seconds' },
      { label: 'NaN (time)', value: d.nan_count_time },
      { label: 'NaN (ppg)', value: d.nan_count_ppg },
      { label: 'Sampling Rate', value: d.sampling_rate, unit: 'Hz' },
    ]);
    showPlot('plot1', d.plot_raw);
    enableStep(2);
  } catch (e) { showStatus(1, 'Error: ' + e.message, 'error'); }
  showSpinner(1, false);
}

async function runStep(endpoint, stepNum) {
  showSpinner(stepNum, true);
  try {
    let payload = { session_id: sessionId };
    if (stepNum === 6) {
      payload.skip_single_peak_check = document.getElementById('skipSinglePeakCheck').checked;
    }
    const res = await fetch('/api/' + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const d = await res.json();
    if (d.error) { showStatus(stepNum, d.error, 'error'); showSpinner(stepNum, false); return; }

    if (stepNum === 2) {
      showStatus(2, d.sr_mismatch_warning ? d.sr_mismatch_warning : 'Cleansed successfully', d.sr_mismatch_warning ? 'warning' : 'success');
      renderStats('stats2', [
        { label: 'Original', value: d.original_samples.toLocaleString() },
        { label: 'Cleansed', value: d.cleansed_samples.toLocaleString() },
        { label: 'Dropped', value: d.dropped_rows },
        { label: 'Est. SR', value: d.estimated_sr || '-', unit: 'Hz' },
        { label: 'Duration', value: d.duration_seconds.toFixed(1), unit: 'sec' },
      ]);
      showPlot('plot2', d.plot_cleansed);
      enableStep(3);
    }
    else if (stepNum === 3) {
      showStatus(3, d.downsampled ? `Downsampled ${d.downsample_factor}x` : 'No downsampling needed', 'success');
      renderStats('stats3', [
        { label: 'Original SR', value: d.original_sr, unit: 'Hz' },
        { label: 'Target SR', value: d.target_sr, unit: 'Hz' },
        { label: 'Original', value: d.original_samples.toLocaleString() },
        { label: 'Downsampled', value: d.downsampled_samples.toLocaleString() },
        { label: 'Factor', value: d.downsample_factor + 'x' },
      ]);
      showPlot('plot3', d.plot_downsampled);
      enableStep(4);
    }
    else if (stepNum === 4) {
      showStatus(4, 'Preprocessing complete', 'success');
      renderStats('stats4', [
        { label: 'DC Mean Removed', value: d.dc_mean_removed },
        { label: 'Bandpass', value: d.filter_low + '-' + d.filter_high, unit: 'Hz' },
        { label: 'Filter Order', value: d.filter_order },
        { label: 'SavGol Window', value: d.savgol_window_samples, unit: 'samples' },
        { label: 'Signal Std', value: d.signal_stats.std },
      ]);
      showPlot('plot4a', d.plot_dc_removal);
      showPlot('plot4b', d.plot_bandpass);
      showPlot('plot4c', d.plot_smoothing);
      enableStep(5);
    }
    else if (stepNum === 5) {
      showStatus(5, `Detected ${d.num_peaks} peaks`, 'success');
      renderStats('stats5', [
        { label: 'Peaks Found', value: d.num_peaks },
        { label: 'Height Thresh', value: d.height_threshold },
        { label: 'Distance Thresh', value: d.distance_threshold },
        { label: 'Mean Interval', value: d.mean_peak_interval_sec ? d.mean_peak_interval_sec + 's' : '-' },
        { label: 'Est. HR', value: d.estimated_hr_bpm ? d.estimated_hr_bpm : '-', unit: 'bpm' },
      ]);
      showPlot('plot5', d.plot_peaks);
      enableStep(6);
    }
    else if (stepNum === 6) {
      showStatus(6, `Extracted ${d.num_windows} windows`, 'success');
      renderStats('stats6', [
        { label: 'Windows', value: d.num_windows },
        { label: 'Window Size', value: d.window_size_samples, unit: 'samples' },
        { label: 'Duration', value: d.window_duration_sec, unit: 'sec' },
        { label: 'Rejected', value: d.rejected_count },
      ]);
      showPlot('plot6', d.plot_windows);
      enableStep(7);
    }
    else if (stepNum === 7) {
      showStatus(7, 'Template computed', 'success');
      renderStats('stats7', [
        { label: 'Template Length', value: d.template_length, unit: 'samples' },
        { label: 'Windows Used', value: d.num_windows_used },
        { label: 'Mean', value: d.template_stats.mean },
        { label: 'Std', value: d.template_stats.std },
      ]);
      showPlot('plot7', d.plot_template);
      enableStep(8);
    }
    else if (stepNum === 8) {
      showStatus(8, `Accepted ${d.accepted_windows} / ${d.total_windows} windows (${d.filtering_rate_pct}%)`, 'success');
      renderStats('stats8', [
        { label: 'Accepted', value: d.accepted_windows },
        { label: 'Rejected', value: d.rejected_windows },
        { label: 'Filter Rate', value: d.filtering_rate_pct + '%' },
        { label: 'Threshold', value: d.similarity_threshold },
        { label: 'Mean Sim (Acc)', value: d.mean_similarity_accepted || '-' },
        { label: 'Mean Sim (Rej)', value: d.mean_similarity_rejected || '-' },
      ]);
      showPlot('plot8a', d.plot_accepted_rejected);
      showPlot('plot8b', d.plot_similarity_histogram);
      enableStep(9);
    }
  } catch (e) { showStatus(stepNum, 'Error: ' + e.message, 'error'); }
  showSpinner(stepNum, false);
}

async function runSave() {
  showSpinner(9, true);
  try {
    const res = await fetch('/api/save_output', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    const d = await res.json();
    if (d.error) { showStatus(9, d.error, 'error'); showSpinner(9, false); return; }

    showStatus(9, `Saved ${d.num_windows} windows (${d.total_rows.toLocaleString()} rows)`, 'success');
    renderStats('stats9', [
      { label: 'Windows', value: d.num_windows },
      { label: 'Window Length', value: d.window_length, unit: 'samples' },
      { label: 'Total Rows', value: d.total_rows.toLocaleString() },
      { label: 'File', value: d.file_path },
    ]);
    document.getElementById('downloadSection').classList.remove('hidden');
    document.getElementById('inferenceSection').classList.remove('hidden');
  } catch (e) { showStatus(9, 'Error: ' + e.message, 'error'); }
  showSpinner(9, false);
}

function downloadCSV() {
  window.open('/api/download/' + sessionId + '/ppg_windows', '_blank');
}

function downloadPredictions() {
  window.open('/api/download/' + sessionId + '/predictions', '_blank');
}

async function runInference() {
  const mp = document.getElementById('modelPath').value;
  if (!mp) { showStatus('Inf', 'Please enter a model path', 'error'); return; }
  document.getElementById('spinInf').style.display = 'block';

  try {
    const res = await fetch('/api/run_inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, model_path: mp })
    });
    const d = await res.json();
    if (d.error) { showStatus('Inf', d.error, 'error'); document.getElementById('spinInf').style.display = 'none'; return; }

    showStatus('Inf', 'Inference complete: ' + d.clinical_interpretation, 'success');
    renderStats('statsInf', [
      { label: 'Predictions', value: d.num_predictions },
      { label: 'Mean Glucose', value: d.mean_glucose_mgdl, unit: 'mg/dL' },
      { label: 'Std', value: d.std_glucose_mgdl, unit: 'mg/dL' },
      { label: 'Min', value: d.min_glucose_mgdl, unit: 'mg/dL' },
      { label: 'Max', value: d.max_glucose_mgdl, unit: 'mg/dL' },
    ]);
    showPlot('plotInf', d.plot_predictions);
    document.getElementById('downloadPredSection').classList.remove('hidden');
  } catch (e) { showStatus('Inf', 'Error: ' + e.message, 'error'); }
  document.getElementById('spinInf').style.display = 'none';
}
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab PPG Pipeline Visualization')
    parser.add_argument('--port', type=int, default=5002, help='Port (default: 5002)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Lab PPG Processing Pipeline")
    print(f"  Open http://localhost:{args.port} in your browser")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=args.port, debug=True)
