"""
Lab PPG Inference Report with Clarke Error Grid Analysis
Generates comprehensive HTML report from per-sample glucose predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os


BASE_DIR = Path(r"C:\IITM\vitalDB\data\LABPPG\ppg_windows_dir")

# Reference glucose values from inference_results.csv
SAMPLES = [
    ("Sub1Sample1", 98),
    ("Sub1Sample2", 98),
    ("Sub1Sample3", 158),
    ("Sub1Sample4", 100),
    ("Sub1Sample5", 112),
    ("Sub2Sample1", 148),
    ("Sub3Sample1", 163),
    ("Sub4Sample1", 157),
]


def load_all_predictions():
    """Load predictions from each sample's predictions/ folder and pair with reference glucose."""
    rows = []
    for sample_name, ref_glucose in SAMPLES:
        pred_file = BASE_DIR / sample_name / "predictions" / "glucose_predictions.csv"
        if not pred_file.exists():
            print(f"[WARN] Missing: {pred_file}")
            continue
        df = pd.read_csv(pred_file)
        for _, row in df.iterrows():
            rows.append({
                'sample': sample_name,
                'window_index': int(row['window_index']),
                'predicted_glucose_mg_dl': row['predicted_glucose_mg_dl'],
                'reference_glucose_mg_dl': ref_glucose,
            })
    all_df = pd.DataFrame(rows)
    all_df['error'] = all_df['predicted_glucose_mg_dl'] - all_df['reference_glucose_mg_dl']
    all_df['abs_error'] = np.abs(all_df['error'])
    all_df['rel_error_pct'] = all_df['abs_error'] / all_df['reference_glucose_mg_dl'] * 100
    return all_df


def clarke_zone(ref, pred):
    """Determine Clarke Error Grid zone for a single (ref, pred) pair."""
    if (ref >= 100 and abs(pred - ref) / ref <= 0.20) or \
       (ref < 100 and abs(pred - ref) <= 20):
        return 'A'
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return 'E'
    if (ref >= 240 and pred <= 70) or (ref <= 70 and pred >= 180):
        return 'D'
    # For ref in 70-180 range, pred outside 70-180
    if ref >= 70 and ref <= 180 and (pred > 180 or pred < 70):
        if (ref > 180 and pred < 70) or (ref < 70 and pred > 180):
            return 'C'
        return 'B'
    return 'B'


def zone_analysis(ref_arr, pred_arr):
    """Return zone counts and percentages."""
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for r, p in zip(ref_arr, pred_arr):
        zones[clarke_zone(r, p)] += 1
    total = sum(zones.values())
    pcts = {k: (v / total * 100) if total > 0 else 0 for k, v in zones.items()}
    return zones, pcts


def create_clarke_svg(ref_arr, pred_arr, width=520, height=520):
    """Create SVG Clarke Error Grid."""
    margin = 55
    ps = width - 2 * margin
    mx = 300  # max axis value for lab PPG range

    def sx(v): return margin + (v / mx) * ps
    def sy(v): return height - margin - (v / mx) * ps

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>']

    # Grid
    for v in range(50, mx + 1, 50):
        a = 0.2 if v % 100 != 0 else 0.35
        svg.append(f'<line x1="{sx(v)}" y1="{margin}" x2="{sx(v)}" y2="{height-margin}" stroke="#555" stroke-width="0.5" opacity="{a}"/>')
        svg.append(f'<line x1="{margin}" y1="{sy(v)}" x2="{width-margin}" y2="{sy(v)}" stroke="#555" stroke-width="0.5" opacity="{a}"/>')

    # Axes
    svg.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#888" stroke-width="1.5"/>')

    # Perfect line
    svg.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(mx)}" y2="{sy(mx)}" stroke="#667eea" stroke-width="1.5" opacity="0.5" stroke-dasharray="8,4"/>')

    # +/- 20% boundaries
    svg.append(f'<line x1="{sx(0)}" y1="{sy(20)}" x2="{sx(100)}" y2="{sy(120)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    svg.append(f'<line x1="{sx(100)}" y1="{sy(120)}" x2="{sx(mx)}" y2="{sy(mx*1.2)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    svg.append(f'<line x1="{sx(20)}" y1="{sy(0)}" x2="{sx(100)}" y2="{sy(80)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    svg.append(f'<line x1="{sx(100)}" y1="{sy(80)}" x2="{sx(mx)}" y2="{sy(mx*0.8)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')

    # Hypo/hyper lines
    svg.append(f'<line x1="{sx(70)}" y1="{margin}" x2="{sx(70)}" y2="{height-margin}" stroke="#e74c3c" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')
    svg.append(f'<line x1="{margin}" y1="{sy(70)}" x2="{width-margin}" y2="{sy(70)}" stroke="#e74c3c" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')
    svg.append(f'<line x1="{sx(180)}" y1="{margin}" x2="{sx(180)}" y2="{height-margin}" stroke="#f39c12" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')
    svg.append(f'<line x1="{margin}" y1="{sy(180)}" x2="{width-margin}" y2="{sy(180)}" stroke="#f39c12" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')

    # Zone labels
    svg.append(f'<text x="{sx(150)}" y="{sy(150)}" font-size="28" fill="rgba(255,255,255,0.12)" font-weight="bold" font-family="Segoe UI,Arial">A</text>')
    svg.append(f'<text x="{sx(70)}" y="{sy(220)}" font-size="22" fill="rgba(255,255,255,0.1)" font-weight="bold" font-family="Segoe UI,Arial">B</text>')
    svg.append(f'<text x="{sx(220)}" y="{sy(70)}" font-size="22" fill="rgba(255,255,255,0.1)" font-weight="bold" font-family="Segoe UI,Arial">B</text>')

    # Color map per sample for distinct visualization
    sample_colors = {
        'Sub1Sample1': '#e74c3c', 'Sub1Sample2': '#e67e22', 'Sub1Sample3': '#f1c40f',
        'Sub1Sample4': '#2ecc71', 'Sub1Sample5': '#1abc9c',
        'Sub2Sample1': '#3498db', 'Sub3Sample1': '#9b59b6', 'Sub4Sample1': '#e84393',
    }

    # Plot points - since we have few unique ref values, add jitter to ref for visibility
    np.random.seed(42)
    svg.append('<g>')
    for r, p, s in zip(ref_arr, pred_arr, [None]*len(ref_arr)):
        color = '#3498db'
        x = sx(min(r, mx))
        y = sy(min(p, mx))
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" opacity="0.6"/>')
    svg.append('</g>')

    # Tick labels
    for v in range(0, mx + 1, 50):
        svg.append(f'<text x="{sx(v)}" y="{height-margin+16}" text-anchor="middle" font-size="10" fill="#aaa" font-family="Segoe UI,Arial">{v}</text>')
        svg.append(f'<text x="{margin-10}" y="{sy(v)+4}" text-anchor="end" font-size="10" fill="#aaa" font-family="Segoe UI,Arial">{v}</text>')

    svg.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12" fill="#ccc" font-family="Segoe UI,Arial">Reference Glucose (mg/dL)</text>')
    svg.append(f'<text x="13" y="{height/2}" text-anchor="middle" font-size="12" fill="#ccc" font-family="Segoe UI,Arial" transform="rotate(-90,13,{height/2})">Predicted Glucose (mg/dL)</text>')
    svg.append('</svg>')
    return '\n'.join(svg)


def create_clarke_svg_colored(all_df, width=520, height=520):
    """Clarke grid with points colored by sample."""
    margin = 55
    ps = width - 2 * margin
    mx = 300

    def sx(v): return margin + (v / mx) * ps
    def sy(v): return height - margin - (v / mx) * ps

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>']

    for v in range(50, mx + 1, 50):
        a = 0.2 if v % 100 != 0 else 0.35
        svg.append(f'<line x1="{sx(v)}" y1="{margin}" x2="{sx(v)}" y2="{height-margin}" stroke="#555" stroke-width="0.5" opacity="{a}"/>')
        svg.append(f'<line x1="{margin}" y1="{sy(v)}" x2="{width-margin}" y2="{sy(v)}" stroke="#555" stroke-width="0.5" opacity="{a}"/>')

    svg.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#888" stroke-width="1.5"/>')
    svg.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(mx)}" y2="{sy(mx)}" stroke="#667eea" stroke-width="1.5" opacity="0.5" stroke-dasharray="8,4"/>')

    # Zone boundaries
    svg.append(f'<line x1="{sx(0)}" y1="{sy(20)}" x2="{sx(100)}" y2="{sy(120)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    svg.append(f'<line x1="{sx(100)}" y1="{sy(120)}" x2="{sx(mx)}" y2="{sy(mx*1.2)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    svg.append(f'<line x1="{sx(20)}" y1="{sy(0)}" x2="{sx(100)}" y2="{sy(80)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    svg.append(f'<line x1="{sx(100)}" y1="{sy(80)}" x2="{sx(mx)}" y2="{sy(mx*0.8)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')

    svg.append(f'<line x1="{sx(70)}" y1="{margin}" x2="{sx(70)}" y2="{height-margin}" stroke="#e74c3c" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')
    svg.append(f'<line x1="{margin}" y1="{sy(70)}" x2="{width-margin}" y2="{sy(70)}" stroke="#e74c3c" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')
    svg.append(f'<line x1="{sx(180)}" y1="{margin}" x2="{sx(180)}" y2="{height-margin}" stroke="#f39c12" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')
    svg.append(f'<line x1="{margin}" y1="{sy(180)}" x2="{width-margin}" y2="{sy(180)}" stroke="#f39c12" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')

    svg.append(f'<text x="{sx(150)}" y="{sy(150)}" font-size="28" fill="rgba(255,255,255,0.12)" font-weight="bold" font-family="Segoe UI,Arial">A</text>')

    sample_colors = {
        'Sub1Sample1': '#e74c3c', 'Sub1Sample2': '#e67e22', 'Sub1Sample3': '#f1c40f',
        'Sub1Sample4': '#2ecc71', 'Sub1Sample5': '#1abc9c',
        'Sub2Sample1': '#3498db', 'Sub3Sample1': '#9b59b6', 'Sub4Sample1': '#e84393',
    }

    # Add small horizontal jitter so overlapping ref values spread out
    np.random.seed(42)
    svg.append('<g>')
    for _, row in all_df.iterrows():
        r = row['reference_glucose_mg_dl'] + np.random.uniform(-3, 3)
        p = row['predicted_glucose_mg_dl']
        color = sample_colors.get(row['sample'], '#3498db')
        x = sx(min(r, mx))
        y = sy(min(p, mx))
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" opacity="0.7"/>')
    svg.append('</g>')

    for v in range(0, mx + 1, 50):
        svg.append(f'<text x="{sx(v)}" y="{height-margin+16}" text-anchor="middle" font-size="10" fill="#aaa" font-family="Segoe UI,Arial">{v}</text>')
        svg.append(f'<text x="{margin-10}" y="{sy(v)+4}" text-anchor="end" font-size="10" fill="#aaa" font-family="Segoe UI,Arial">{v}</text>')

    svg.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12" fill="#ccc" font-family="Segoe UI,Arial">Reference Glucose (mg/dL)</text>')
    svg.append(f'<text x="13" y="{height/2}" text-anchor="middle" font-size="12" fill="#ccc" font-family="Segoe UI,Arial" transform="rotate(-90,13,{height/2})">Predicted Glucose (mg/dL)</text>')
    svg.append('</svg>')
    return '\n'.join(svg)


def create_sample_bar_svg(sample_stats, width=700, height=350):
    """SVG bar chart of MAE per sample."""
    margin_l, margin_r, margin_t, margin_b = 110, 20, 30, 40
    pw = width - margin_l - margin_r
    ph = height - margin_t - margin_b
    n = len(sample_stats)
    bar_h = ph / n * 0.7
    gap = ph / n * 0.3

    max_mae = max(s['mae'] for s in sample_stats) * 1.15

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>']

    for i, s in enumerate(sample_stats):
        y = margin_t + i * (bar_h + gap)
        bw = (s['mae'] / max_mae) * pw
        # Color by MAE quality
        color = '#2ecc71' if s['mae'] < 20 else '#f39c12' if s['mae'] < 35 else '#e74c3c'
        svg.append(f'<rect x="{margin_l}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}" opacity="0.85"/>')
        svg.append(f'<text x="{margin_l - 6}" y="{y + bar_h/2 + 4:.1f}" text-anchor="end" font-size="11" fill="#ccc" font-family="Segoe UI,Arial">{s["sample"]}</text>')
        svg.append(f'<text x="{margin_l + bw + 6:.1f}" y="{y + bar_h/2 + 4:.1f}" font-size="11" fill="#eee" font-family="Segoe UI,Arial">{s["mae"]:.1f} mg/dL (ref: {s["ref"]:.0f})</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


def create_prediction_spread_svg(all_df, width=750, height=300):
    """SVG showing predicted distributions per sample as box-like plots."""
    margin_l, margin_r, margin_t, margin_b = 110, 30, 25, 35
    pw = width - margin_l - margin_r
    ph = height - margin_t - margin_b

    samples = all_df['sample'].unique()
    n = len(samples)
    row_h = ph / n

    # Determine x range
    all_preds = all_df['predicted_glucose_mg_dl'].values
    all_refs = all_df['reference_glucose_mg_dl'].values
    x_min = min(all_preds.min(), all_refs.min()) - 10
    x_max = max(all_preds.max(), all_refs.max()) + 10

    def sx(v): return margin_l + ((v - x_min) / (x_max - x_min)) * pw

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>']

    for i, sample in enumerate(samples):
        sub = all_df[all_df['sample'] == sample]
        preds = sub['predicted_glucose_mg_dl'].values
        ref = sub['reference_glucose_mg_dl'].iloc[0]
        y_center = margin_t + i * row_h + row_h / 2

        # Label
        svg.append(f'<text x="{margin_l - 6}" y="{y_center + 4}" text-anchor="end" font-size="10" fill="#ccc" font-family="Segoe UI,Arial">{sample}</text>')

        # Horizontal line for range
        svg.append(f'<line x1="{sx(preds.min())}" y1="{y_center}" x2="{sx(preds.max())}" y2="{y_center}" stroke="#3498db" stroke-width="2" opacity="0.6"/>')

        # Individual predictions
        for p in preds:
            svg.append(f'<circle cx="{sx(p):.1f}" cy="{y_center}" r="2.5" fill="#3498db" opacity="0.5"/>')

        # Mean prediction
        svg.append(f'<circle cx="{sx(preds.mean()):.1f}" cy="{y_center}" r="5" fill="#3498db" stroke="#fff" stroke-width="1"/>')

        # Reference glucose (diamond)
        rx = sx(ref)
        svg.append(f'<polygon points="{rx},{y_center-7} {rx+5},{y_center} {rx},{y_center+7} {rx-5},{y_center}" fill="#e74c3c" opacity="0.9"/>')

    # X-axis ticks
    for v in range(int(x_min // 20) * 20, int(x_max) + 20, 20):
        if v >= x_min and v <= x_max:
            svg.append(f'<text x="{sx(v)}" y="{height - 8}" text-anchor="middle" font-size="9" fill="#aaa" font-family="Segoe UI,Arial">{v}</text>')

    # Legend
    svg.append(f'<circle cx="{margin_l + 10}" cy="{margin_t - 10}" r="4" fill="#3498db" stroke="#fff" stroke-width="0.5"/>')
    svg.append(f'<text x="{margin_l + 18}" y="{margin_t - 6}" font-size="10" fill="#ccc" font-family="Segoe UI,Arial">Predicted (mean=circle)</text>')
    svg.append(f'<polygon points="{margin_l + 180},{margin_t-17} {margin_l + 185},{margin_t-10} {margin_l + 180},{margin_t-3} {margin_l + 175},{margin_t-10}" fill="#e74c3c"/>')
    svg.append(f'<text x="{margin_l + 192}" y="{margin_t - 6}" font-size="10" fill="#ccc" font-family="Segoe UI,Arial">Reference glucose</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


def main():
    print("=" * 80)
    print("LAB PPG INFERENCE REPORT")
    print("=" * 80)

    all_df = load_all_predictions()
    print(f"Loaded {len(all_df)} total predictions across {all_df['sample'].nunique()} samples")

    ref_arr = all_df['reference_glucose_mg_dl'].values
    pred_arr = all_df['predicted_glucose_mg_dl'].values
    errors = pred_arr - ref_arr

    # Overall metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mard = np.mean(np.abs(errors) / ref_arr) * 100
    median_ae = np.median(np.abs(errors))
    mean_bias = np.mean(errors)

    # Clarke zones
    zones, pcts = zone_analysis(ref_arr, pred_arr)
    ab_pct = pcts['A'] + pcts['B']

    print(f"MAE: {mae:.2f} mg/dL")
    print(f"RMSE: {rmse:.2f} mg/dL")
    print(f"MARD: {mard:.2f}%")
    print(f"Clarke A: {pcts['A']:.1f}%, B: {pcts['B']:.1f}%, A+B: {ab_pct:.1f}%")

    # Per-sample stats
    sample_stats = []
    for sample_name, ref_glucose in SAMPLES:
        sub = all_df[all_df['sample'] == sample_name]
        if len(sub) == 0:
            continue
        preds = sub['predicted_glucose_mg_dl'].values
        s_mae = np.mean(np.abs(preds - ref_glucose))
        s_mard = np.mean(np.abs(preds - ref_glucose) / ref_glucose) * 100
        s_zones, s_pcts = zone_analysis(
            np.full(len(preds), ref_glucose), preds
        )
        sample_stats.append({
            'sample': sample_name,
            'ref': ref_glucose,
            'n_windows': len(sub),
            'pred_mean': preds.mean(),
            'pred_std': preds.std(),
            'pred_median': np.median(preds),
            'mae': s_mae,
            'mard': s_mard,
            'bias': np.mean(preds - ref_glucose),
            'zone_a': s_pcts['A'],
            'zone_b': s_pcts['B'],
            'zone_ab': s_pcts['A'] + s_pcts['B'],
        })

    # SVGs
    clarke_svg = create_clarke_svg_colored(all_df)
    bar_svg = create_sample_bar_svg(sample_stats)
    spread_svg = create_prediction_spread_svg(all_df)

    # Accuracy thresholds
    within_15 = np.mean(np.abs(errors) <= 15) * 100
    within_20 = np.mean(np.abs(errors) <= 20) * 100
    within_30 = np.mean(np.abs(errors) <= 30) * 100
    within_20pct = np.mean(np.where(ref_arr >= 100,
                                     np.abs(errors) / ref_arr <= 0.20,
                                     np.abs(errors) <= 20)) * 100

    # Subject-level aggregation
    subject_stats = {}
    for s in sample_stats:
        subj = s['sample'][:4]  # Sub1, Sub2, etc.
        if subj not in subject_stats:
            subject_stats[subj] = {'samples': [], 'total_windows': 0, 'weighted_mae': 0}
        subject_stats[subj]['samples'].append(s)
        subject_stats[subj]['total_windows'] += s['n_windows']

    # Best/worst sample
    best = min(sample_stats, key=lambda x: x['mae'])
    worst = max(sample_stats, key=lambda x: x['mae'])

    ab_class = 'good' if ab_pct >= 95 else 'warn' if ab_pct >= 85 else 'bad'

    sample_colors = {
        'Sub1Sample1': '#e74c3c', 'Sub1Sample2': '#e67e22', 'Sub1Sample3': '#f1c40f',
        'Sub1Sample4': '#2ecc71', 'Sub1Sample5': '#1abc9c',
        'Sub2Sample1': '#3498db', 'Sub3Sample1': '#9b59b6', 'Sub4Sample1': '#e84393',
    }

    # ---- BUILD HTML ----
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Lab PPG Inference Report</title>
<style>
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #e0e0e0;
    padding: 30px;
    max-width: 1300px;
    margin: 0 auto;
    line-height: 1.6;
}}
h1 {{ color: #667eea; text-align: center; font-size: 2.2em; margin-bottom: 5px; }}
h2 {{ color: #764ba2; border-bottom: 2px solid #764ba2; padding-bottom: 8px; margin-top: 45px; }}
h3 {{ color: #667eea; margin-top: 20px; }}
.subtitle {{ text-align: center; color: #95a5a6; font-size: 1.05em; margin-bottom: 30px; }}
.metric-container {{
    display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin: 22px 0;
}}
.metric-card {{
    background: rgba(102,126,234,0.12); border: 1px solid rgba(102,126,234,0.25);
    border-radius: 14px; padding: 16px 26px; text-align: center; min-width: 130px;
}}
.metric-label {{ color: #95a5a6; font-size: 0.82em; margin-bottom: 3px; }}
.metric-value {{ font-size: 1.7em; font-weight: bold; color: #667eea; }}
.metric-unit {{ font-size: 0.72em; color: #7f8c8d; }}
table {{
    border-collapse: collapse; width: 100%; margin: 16px 0;
    background: rgba(255,255,255,0.02); border-radius: 10px; overflow: hidden; font-size: 0.9em;
}}
th, td {{ padding: 9px 13px; text-align: center; border: 1px solid rgba(255,255,255,0.08); }}
th {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff; font-weight: 600;
}}
tr:nth-child(even) {{ background: rgba(255,255,255,0.02); }}
.zone-A {{ color: #2ecc71; font-weight: bold; }}
.zone-B {{ color: #3498db; font-weight: bold; }}
.zone-C {{ color: #f39c12; font-weight: bold; }}
.zone-D {{ color: #e74c3c; font-weight: bold; }}
.zone-E {{ color: #9b59b6; font-weight: bold; }}
.good {{ color: #2ecc71; }} .warn {{ color: #f39c12; }} .bad {{ color: #e74c3c; }}
.chart-container {{
    display: flex; justify-content: center; margin: 22px 0;
    background: rgba(0,0,0,0.2); border-radius: 14px; padding: 18px;
}}
.two-col {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 18px 0;
}}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.zone-bar {{
    display: flex; height: 38px; border-radius: 10px; overflow: hidden; margin: 8px 0;
}}
.zone-seg {{
    display: flex; align-items: center; justify-content: center;
    font-weight: bold; font-size: 0.85em; color: #fff;
}}
.insight-box {{
    background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
    border-left: 4px solid #667eea; padding: 16px 20px; margin: 16px 0; border-radius: 0 10px 10px 0;
}}
.insight-box h3 {{ margin-top: 0; }}
.legend {{
    display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; margin: 12px 0; font-size: 0.88em;
}}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-color {{ width: 14px; height: 14px; border-radius: 3px; }}
.highlight-row {{ background: rgba(102,126,234,0.15) !important; font-weight: bold; }}
.sample-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
</style>
</head>
<body>

<h1>Lab PPG Inference Report</h1>
<p class="subtitle">
    Latest Model (predictions/) | {len(all_df)} Total Predictions |
    {len(SAMPLES)} Samples | 4 Subjects |
    Reference Range: {ref_arr.min():.0f}&ndash;{ref_arr.max():.0f} mg/dL
</p>

<!-- ===== SUMMARY ===== -->
<h2>1. Summary Metrics</h2>
<div class="metric-container">
    <div class="metric-card"><div class="metric-label">Total Windows</div><div class="metric-value">{len(all_df)}</div></div>
    <div class="metric-card"><div class="metric-label">Samples</div><div class="metric-value">{len(SAMPLES)}</div></div>
    <div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value">{mae:.1f}</div><div class="metric-unit">mg/dL</div></div>
    <div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">{rmse:.1f}</div><div class="metric-unit">mg/dL</div></div>
    <div class="metric-card"><div class="metric-label">MARD</div><div class="metric-value">{mard:.1f}%</div></div>
    <div class="metric-card"><div class="metric-label">Median AE</div><div class="metric-value">{median_ae:.1f}</div><div class="metric-unit">mg/dL</div></div>
    <div class="metric-card"><div class="metric-label">Mean Bias</div><div class="metric-value">{mean_bias:+.1f}</div><div class="metric-unit">mg/dL</div></div>
</div>

<h3>Accuracy Thresholds</h3>
<table style="max-width:550px; margin: 12px auto;">
    <tr><th>Threshold</th><th>% Within</th></tr>
    <tr><td>+/- 15 mg/dL</td><td>{within_15:.1f}%</td></tr>
    <tr><td>+/- 20 mg/dL</td><td>{within_20:.1f}%</td></tr>
    <tr><td>+/- 30 mg/dL</td><td>{within_30:.1f}%</td></tr>
    <tr><td>+/- 20% (ISO 15197)</td><td>{within_20pct:.1f}%</td></tr>
</table>

<!-- ===== CLARKE ERROR GRID ===== -->
<h2>2. Clarke Error Grid Analysis</h2>

<div class="two-col">
    <div class="chart-container">{clarke_svg}</div>
    <div>
        <h3>Zone Distribution</h3>
        <div class="zone-bar">
            {"".join(f'<div class="zone-seg" style="width:{pcts[z]}%; background:{c};">{pcts[z]:.1f}%</div>' if pcts[z] > 2 else f'<div class="zone-seg" style="width:max({pcts[z]}%,2px); background:{c};"></div>' for z, c in [("A","#2ecc71"),("B","#3498db"),("C","#f39c12"),("D","#e74c3c"),("E","#9b59b6")])}
        </div>

        <table>
            <tr><th>Zone</th><th>Description</th><th>Count</th><th>%</th></tr>
            <tr><td class="zone-A">A</td><td>Clinically Accurate</td><td>{zones['A']}</td><td class="zone-A">{pcts['A']:.1f}%</td></tr>
            <tr><td class="zone-B">B</td><td>Benign Errors</td><td>{zones['B']}</td><td class="zone-B">{pcts['B']:.1f}%</td></tr>
            <tr><td class="zone-C">C</td><td>Overcorrection</td><td>{zones['C']}</td><td class="zone-C">{pcts['C']:.1f}%</td></tr>
            <tr><td class="zone-D">D</td><td>Failure to Detect</td><td>{zones['D']}</td><td class="zone-D">{pcts['D']:.1f}%</td></tr>
            <tr><td class="zone-E">E</td><td>Erroneous Treatment</td><td>{zones['E']}</td><td class="zone-E">{pcts['E']:.1f}%</td></tr>
            <tr class="highlight-row">
                <td colspan="2">Clinically Acceptable (A+B)</td>
                <td>{zones['A']+zones['B']}</td>
                <td class="{ab_class}">{ab_pct:.1f}%</td>
            </tr>
        </table>

        <div class="legend">
            {"".join(f'<div class="legend-item"><div class="sample-dot" style="background:{sample_colors[s]}"></div>{s} ({r} mg/dL)</div>' for s, r in SAMPLES)}
        </div>
    </div>
</div>

<!-- ===== PREDICTION SPREAD ===== -->
<h2>3. Prediction Spread per Sample</h2>
<div class="chart-container">{spread_svg}</div>
<p style="text-align:center; color:#95a5a6; font-size:0.88em;">
    Blue circles = individual predictions (large circle = mean) | Red diamond = reference glucose
</p>

<!-- ===== PER-SAMPLE MAE ===== -->
<h2>4. MAE per Sample</h2>
<div class="chart-container">{bar_svg}</div>

<!-- ===== PER-SAMPLE TABLE ===== -->
<h2>5. Detailed Per-Sample Results</h2>
<table>
    <tr>
        <th>Sample</th><th>Ref (mg/dL)</th><th>Windows</th>
        <th>Pred Mean</th><th>Pred Std</th><th>Pred Median</th>
        <th>MAE</th><th>MARD</th><th>Bias</th>
        <th>Zone A</th><th>Zone B</th><th>A+B</th>
    </tr>"""

    for s in sample_stats:
        ab = s['zone_ab']
        html += f"""
    <tr>
        <td style="text-align:left;"><span class="sample-dot" style="background:{sample_colors[s['sample']]}"></span>{s['sample']}</td>
        <td>{s['ref']:.0f}</td>
        <td>{s['n_windows']}</td>
        <td>{s['pred_mean']:.1f}</td>
        <td>{s['pred_std']:.1f}</td>
        <td>{s['pred_median']:.1f}</td>
        <td class="{'good' if s['mae'] < 20 else 'warn' if s['mae'] < 35 else 'bad'}">{s['mae']:.1f}</td>
        <td>{s['mard']:.1f}%</td>
        <td>{s['bias']:+.1f}</td>
        <td class="zone-A">{s['zone_a']:.0f}%</td>
        <td class="zone-B">{s['zone_b']:.0f}%</td>
        <td class="{'good' if ab >= 95 else 'warn' if ab >= 80 else 'bad'}">{ab:.0f}%</td>
    </tr>"""

    # Overall row
    html += f"""
    <tr class="highlight-row">
        <td colspan="2">Overall</td>
        <td>{len(all_df)}</td>
        <td>{pred_arr.mean():.1f}</td>
        <td>{pred_arr.std():.1f}</td>
        <td>{np.median(pred_arr):.1f}</td>
        <td>{mae:.1f}</td>
        <td>{mard:.1f}%</td>
        <td>{mean_bias:+.1f}</td>
        <td class="zone-A">{pcts['A']:.0f}%</td>
        <td class="zone-B">{pcts['B']:.0f}%</td>
        <td class="{ab_class}">{ab_pct:.0f}%</td>
    </tr>
</table>

<!-- ===== SUBJECT-LEVEL ===== -->
<h2>6. Subject-Level Summary</h2>
<table>
    <tr><th>Subject</th><th>Samples</th><th>Total Windows</th><th>Ref Range</th><th>Avg MAE</th><th>Avg MARD</th></tr>"""

    for subj, info in sorted(subject_stats.items()):
        refs = [s['ref'] for s in info['samples']]
        avg_mae = np.mean([s['mae'] for s in info['samples']])
        avg_mard = np.mean([s['mard'] for s in info['samples']])
        tw = sum(s['n_windows'] for s in info['samples'])
        html += f"""
    <tr>
        <td>{subj}</td>
        <td>{len(info['samples'])}</td>
        <td>{tw}</td>
        <td>{min(refs):.0f}&ndash;{max(refs):.0f}</td>
        <td class="{'good' if avg_mae < 20 else 'warn' if avg_mae < 35 else 'bad'}">{avg_mae:.1f}</td>
        <td>{avg_mard:.1f}%</td>
    </tr>"""

    html += f"""
</table>

<!-- ===== CLINICAL ASSESSMENT ===== -->
<h2>7. Clinical Assessment</h2>

<div class="insight-box">
    <h3>Performance Summary</h3>
    <ul>
        <li><b>Zone A (Clinically Accurate):</b> <span class="{'good' if pcts['A'] >= 50 else 'warn'}">{pcts['A']:.1f}%</span></li>
        <li><b>Clinically Acceptable (A+B):</b> <span class="{ab_class}">{ab_pct:.1f}%</span></li>
        <li><b>ISO 15197:2013:</b> {'Meets' if ab_pct >= 95 else 'Does NOT meet'} the 95% A+B requirement ({within_20pct:.1f}% within +/-20%/20 mg/dL)</li>
        <li><b>Dangerous Zones (D+E):</b> <span class="{'good' if pcts['D']+pcts['E'] < 1 else 'bad'}">{pcts['D']+pcts['E']:.1f}%</span></li>
    </ul>
</div>

<div class="insight-box">
    <h3>Key Observations</h3>
    <ul>
        <li><b>Best sample:</b> {best['sample']} (MAE={best['mae']:.1f} mg/dL, ref={best['ref']:.0f} mg/dL)</li>
        <li><b>Worst sample:</b> {worst['sample']} (MAE={worst['mae']:.1f} mg/dL, ref={worst['ref']:.0f} mg/dL)</li>
        <li><b>Mean bias:</b> {mean_bias:+.1f} mg/dL &mdash; model tends to {"overpredict" if mean_bias > 0 else "underpredict"} on average</li>"""

    # Check if model predicts well for high glucose
    high_ref_samples = [s for s in sample_stats if s['ref'] >= 140]
    low_ref_samples = [s for s in sample_stats if s['ref'] < 110]
    if high_ref_samples and low_ref_samples:
        high_mae = np.mean([s['mae'] for s in high_ref_samples])
        low_mae = np.mean([s['mae'] for s in low_ref_samples])
        html += f"""
        <li><b>Low glucose (&lt;110) avg MAE:</b> {low_mae:.1f} mg/dL vs <b>High glucose (&ge;140) avg MAE:</b> {high_mae:.1f} mg/dL</li>"""

    html += f"""
        <li>Prediction range: {pred_arr.min():.1f}&ndash;{pred_arr.max():.1f} mg/dL vs reference range: {ref_arr.min():.0f}&ndash;{ref_arr.max():.0f} mg/dL</li>
    </ul>
</div>

<div class="insight-box">
    <h3>Recommendations</h3>
    <ul>"""

    if ab_pct < 95:
        html += f"""
        <li>Model does not meet clinical acceptability threshold (A+B = {ab_pct:.1f}%, target 95%)</li>"""
    else:
        html += f"""
        <li class="good">Model meets clinical acceptability threshold (A+B = {ab_pct:.1f}% &ge; 95%)</li>"""

    if mard > 20:
        html += f"""
        <li>MARD of {mard:.1f}% exceeds typical CGM targets (10-15%); significant improvement needed</li>"""

    if abs(mean_bias) > 15:
        html += f"""
        <li>Systematic bias of {mean_bias:+.1f} mg/dL suggests calibration offset may help</li>"""

    html += """
        <li>Consider collecting more lab PPG samples across a wider glucose range for better evaluation</li>
        <li>Intra-subject consistency (multiple samples from Subject 1) can help assess reproducibility</li>
    </ul>
</div>

<p style="text-align:center; color:#555; margin-top:50px; font-size:0.82em;">
    Generated by analyze_labppg_inference.py | Lab PPG Dataset
</p>

</body>
</html>"""

    output_path = BASE_DIR / "LAB_PPG_INFERENCE_REPORT_LATEST.html"
    output_path.write_text(html, encoding='utf-8')
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
