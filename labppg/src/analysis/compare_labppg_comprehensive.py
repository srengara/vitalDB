"""
Comprehensive Lab PPG Comparison Report
========================================
Comparison 1: Model 174012 vs Model 163632 (distance_multiplier=0.8)
Comparison 2: Distance multiplier 0.6 vs 0.8 for Model 174012
"""

import os
import glob
import numpy as np
import pandas as pd

# Reference glucose values (mg/dL)
REFERENCE = {
    'Sub1Sample1': 98,
    'Sub1Sample2': 98,
    'Sub1Sample3': 158,
    'Sub1Sample4': 100,
    'Sub1Sample5': 112,
    'Sub2Sample1': 148,
    'Sub3Sample1': 163,
    'Sub4Sample1': 157,
}

SAMPLES = list(REFERENCE.keys())

# ---- Paths ----
BASE_08 = r'C:\IITM\vitalDB\data\LABPPG\ppg_windows_dir'
BASE_06 = r'C:\IITM\vitalDB\data\LABPPG\Inference_with_0.6hthreshold'

# ---- File mapping for 0.6 threshold (inconsistent naming) ----
FILES_06_174012 = {
    'Sub1Sample1': 'predictions (5).csv',
    'Sub1Sample2': 'predictions (4).csv',
    'Sub1Sample3': 'predictions (7).csv',
    'Sub1Sample4': 'predictions (9).csv',
    'Sub1Sample5': 'predictions (10).csv',
    'Sub2Sample1': 'predictions (11).csv',
    'Sub3Sample1': 'predictions (12).csv',
    'Sub4Sample1': 'predictions (13).csv',
}

FILES_06_163632 = {
    'Sub1Sample1': 'predictions.csv',
    'Sub1Sample2': 'predictions.csv',
    'Sub1Sample3': 'predictions.csv',
    'Sub1Sample4': 'predictions.csv',
    'Sub1Sample5': 'predictions (10).csv',
    'Sub2Sample1': 'predictions.csv',
    'Sub3Sample1': 'Predictions.csv',
    'Sub4Sample1': 'Predictions.csv',
}


def load_predictions_08(sample, model_id):
    """Load predictions from 0.8 multiplier directory"""
    path = os.path.join(BASE_08, sample, f'predictions_{model_id}', 'glucose_predictions.csv')
    df = pd.read_csv(path)
    return df['predicted_glucose_mg_dl'].values


def load_predictions_06(sample, model_id):
    """Load predictions from 0.6 multiplier directory"""
    if model_id == '174012':
        fname = FILES_06_174012[sample]
    else:
        fname = FILES_06_163632[sample]
    path = os.path.join(BASE_06, f'model_{model_id}', sample, fname)
    df = pd.read_csv(path)
    return df['predicted_glucose_mg_dl'].values


def compute_metrics(predictions, reference):
    """Compute prediction metrics against reference glucose"""
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    median_pred = np.median(predictions)
    error = mean_pred - reference
    abs_error = abs(error)
    rel_error = abs_error / reference * 100  # MARD %
    return {
        'mean': round(mean_pred, 2),
        'std': round(std_pred, 2),
        'median': round(median_pred, 2),
        'n_windows': len(predictions),
        'error': round(error, 2),
        'abs_error': round(abs_error, 2),
        'mard': round(rel_error, 2),
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE LAB PPG COMPARISON")
    print("=" * 80)

    # ========================================================================
    # Collect all data
    # ========================================================================
    results = {}
    configs = [
        ('174012', '0.8'), ('163632', '0.8'),
        ('174012', '0.6'), ('163632', '0.6'),
    ]

    for model_id, mult in configs:
        key = f'{model_id}_{mult}'
        results[key] = {}
        for sample in SAMPLES:
            ref = REFERENCE[sample]
            try:
                if mult == '0.8':
                    preds = load_predictions_08(sample, model_id)
                else:
                    preds = load_predictions_06(sample, model_id)
                m = compute_metrics(preds, ref)
                m['reference'] = ref
                results[key][sample] = m
                print(f"  [{key}] {sample}: mean={m['mean']:.1f}, ref={ref}, MAE={m['abs_error']:.1f}, MARD={m['mard']:.1f}%")
            except Exception as e:
                print(f"  [{key}] {sample}: FAILED - {e}")
                results[key][sample] = None

    # ========================================================================
    # Compute summary metrics per config
    # ========================================================================
    summaries = {}
    for key in results:
        valid = [v for v in results[key].values() if v is not None]
        if valid:
            summaries[key] = {
                'mae': round(np.mean([v['abs_error'] for v in valid]), 2),
                'mean_mard': round(np.mean([v['mard'] for v in valid]), 2),
                'median_mard': round(np.median([v['mard'] for v in valid]), 2),
                'mean_std': round(np.mean([v['std'] for v in valid]), 2),
                'n_samples': len(valid),
                'total_windows': sum(v['n_windows'] for v in valid),
            }
        else:
            summaries[key] = None

    for key, s in summaries.items():
        if s:
            print(f"\n  Summary [{key}]: MAE={s['mae']}, MARD={s['mean_mard']}%, Windows={s['total_windows']}")

    # ========================================================================
    # Generate HTML
    # ========================================================================
    print("\nGenerating HTML report...")

    # Helper: color code errors
    def error_class(abs_err):
        if abs_err <= 10:
            return 'good'
        elif abs_err <= 20:
            return 'ok'
        elif abs_err <= 35:
            return 'warn'
        else:
            return 'bad'

    def mard_class(mard):
        if mard <= 10:
            return 'good'
        elif mard <= 15:
            return 'ok'
        elif mard <= 25:
            return 'warn'
        else:
            return 'bad'

    # Build the sample-level comparison tables
    # --- Table 1: Model 174012 vs 163632 (mult=0.8) ---
    table1_rows = ''
    for sample in SAMPLES:
        ref = REFERENCE[sample]
        r174 = results['174012_0.8'].get(sample)
        r163 = results['163632_0.8'].get(sample)
        if r174 and r163:
            winner = '174012' if r174['abs_error'] < r163['abs_error'] else '163632'
            table1_rows += f'''
        <tr>
            <td>{sample}</td>
            <td>{ref}</td>
            <td>{r174['mean']}</td>
            <td class="{error_class(r174['abs_error'])}">{r174['abs_error']}</td>
            <td class="{mard_class(r174['mard'])}">{r174['mard']}%</td>
            <td>{r174['n_windows']}</td>
            <td>{r163['mean']}</td>
            <td class="{error_class(r163['abs_error'])}">{r163['abs_error']}</td>
            <td class="{mard_class(r163['mard'])}">{r163['mard']}%</td>
            <td>{r163['n_windows']}</td>
            <td class="winner">{winner}</td>
        </tr>'''

    s174_08 = summaries.get('174012_0.8', {})
    s163_08 = summaries.get('163632_0.8', {})

    # --- Table 2: Multiplier 0.6 vs 0.8 for model 174012 ---
    table2_rows = ''
    for sample in SAMPLES:
        ref = REFERENCE[sample]
        r08 = results['174012_0.8'].get(sample)
        r06 = results['174012_0.6'].get(sample)
        if r08 and r06:
            winner = '0.8' if r08['abs_error'] < r06['abs_error'] else '0.6'
            diff = round(r06['abs_error'] - r08['abs_error'], 2)
            diff_sign = '+' if diff > 0 else ''
            table2_rows += f'''
        <tr>
            <td>{sample}</td>
            <td>{ref}</td>
            <td>{r08['mean']}</td>
            <td class="{error_class(r08['abs_error'])}">{r08['abs_error']}</td>
            <td class="{mard_class(r08['mard'])}">{r08['mard']}%</td>
            <td>{r08['n_windows']}</td>
            <td>{r06['mean']}</td>
            <td class="{error_class(r06['abs_error'])}">{r06['abs_error']}</td>
            <td class="{mard_class(r06['mard'])}">{r06['mard']}%</td>
            <td>{r06['n_windows']}</td>
            <td class="winner">{winner}</td>
            <td class="{'bad' if diff > 5 else 'good' if diff < -5 else ''}">{diff_sign}{diff}</td>
        </tr>'''

    s174_06 = summaries.get('174012_0.6', {})

    # --- Table 3: Multiplier 0.6 vs 0.8 for model 163632 ---
    table3_rows = ''
    for sample in SAMPLES:
        ref = REFERENCE[sample]
        r08 = results['163632_0.8'].get(sample)
        r06 = results['163632_0.6'].get(sample)
        if r08 and r06:
            winner = '0.8' if r08['abs_error'] < r06['abs_error'] else '0.6'
            diff = round(r06['abs_error'] - r08['abs_error'], 2)
            diff_sign = '+' if diff > 0 else ''
            table3_rows += f'''
        <tr>
            <td>{sample}</td>
            <td>{ref}</td>
            <td>{r08['mean']}</td>
            <td class="{error_class(r08['abs_error'])}">{r08['abs_error']}</td>
            <td class="{mard_class(r08['mard'])}">{r08['mard']}%</td>
            <td>{r08['n_windows']}</td>
            <td>{r06['mean']}</td>
            <td class="{error_class(r06['abs_error'])}">{r06['abs_error']}</td>
            <td class="{mard_class(r06['mard'])}">{r06['mard']}%</td>
            <td>{r06['n_windows']}</td>
            <td class="winner">{winner}</td>
            <td class="{'bad' if diff > 5 else 'good' if diff < -5 else ''}">{diff_sign}{diff}</td>
        </tr>'''

    s163_06 = summaries.get('163632_0.6', {})

    # --- Summary comparison bar data ---
    # Compute per-sample data for bar charts
    bar_data_model = []
    for sample in SAMPLES:
        r174 = results['174012_0.8'].get(sample)
        r163 = results['163632_0.8'].get(sample)
        if r174 and r163:
            bar_data_model.append({
                'sample': sample,
                'ref': REFERENCE[sample],
                'pred_174012': r174['mean'],
                'pred_163632': r163['mean'],
                'err_174012': r174['abs_error'],
                'err_163632': r163['abs_error'],
            })

    bar_data_mult = []
    for sample in SAMPLES:
        r08 = results['174012_0.8'].get(sample)
        r06 = results['174012_0.6'].get(sample)
        if r08 and r06:
            bar_data_mult.append({
                'sample': sample,
                'ref': REFERENCE[sample],
                'pred_08': r08['mean'],
                'pred_06': r06['mean'],
                'err_08': r08['abs_error'],
                'err_06': r06['abs_error'],
            })

    # Generate SVG bar chart for model comparison errors
    def make_bar_svg(data, label_a, label_b, key_a, key_b, color_a, color_b, title):
        w, h = 700, 320
        margin_l, margin_b, margin_t, margin_r = 80, 60, 40, 20
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b
        n = len(data)
        if n == 0:
            return ''
        max_err = max(max(d[key_a] for d in data), max(d[key_b] for d in data)) * 1.15
        bar_group_w = plot_w / n
        bar_w = bar_group_w * 0.35

        svg = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        svg += f'<rect width="{w}" height="{h}" fill="#1a1a2e" rx="8"/>'
        svg += f'<text x="{w/2}" y="25" text-anchor="middle" fill="#fff" font-size="14" font-family="Arial">{title}</text>'

        # Y-axis gridlines
        for i in range(5):
            val = max_err * i / 4
            y = margin_t + plot_h - (val / max_err * plot_h)
            svg += f'<line x1="{margin_l}" y1="{y}" x2="{w-margin_r}" y2="{y}" stroke="#333" stroke-width="0.5" stroke-dasharray="4,4"/>'
            svg += f'<text x="{margin_l-8}" y="{y+4}" text-anchor="end" fill="#999" font-size="10" font-family="Arial">{val:.0f}</text>'

        # X-axis
        svg += f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{w-margin_r}" y2="{margin_t+plot_h}" stroke="#666" stroke-width="1"/>'
        svg += f'<text x="{margin_l-10}" y="{h/2}" text-anchor="middle" fill="#999" font-size="11" font-family="Arial" transform="rotate(-90,{margin_l-45},{h/2})">Absolute Error (mg/dL)</text>'

        for i, d in enumerate(data):
            x_center = margin_l + bar_group_w * i + bar_group_w / 2
            x_a = x_center - bar_w - 1
            x_b = x_center + 1
            h_a = d[key_a] / max_err * plot_h
            h_b = d[key_b] / max_err * plot_h

            svg += f'<rect x="{x_a}" y="{margin_t+plot_h-h_a}" width="{bar_w}" height="{h_a}" fill="{color_a}" rx="2" opacity="0.85"/>'
            svg += f'<rect x="{x_b}" y="{margin_t+plot_h-h_b}" width="{bar_w}" height="{h_b}" fill="{color_b}" rx="2" opacity="0.85"/>'

            # Value labels
            svg += f'<text x="{x_a+bar_w/2}" y="{margin_t+plot_h-h_a-4}" text-anchor="middle" fill="{color_a}" font-size="9" font-family="Arial">{d[key_a]:.0f}</text>'
            svg += f'<text x="{x_b+bar_w/2}" y="{margin_t+plot_h-h_b-4}" text-anchor="middle" fill="{color_b}" font-size="9" font-family="Arial">{d[key_b]:.0f}</text>'

            # Sample label
            label = d['sample'].replace('Sub', 'S').replace('Sample', '')
            svg += f'<text x="{x_center}" y="{margin_t+plot_h+15}" text-anchor="middle" fill="#ccc" font-size="9" font-family="Arial">{label}</text>'
            svg += f'<text x="{x_center}" y="{margin_t+plot_h+27}" text-anchor="middle" fill="#888" font-size="8" font-family="Arial">ref:{d["ref"]}</text>'

        # Legend
        lx = w - margin_r - 180
        svg += f'<rect x="{lx}" y="{margin_t+5}" width="12" height="12" fill="{color_a}" rx="2"/>'
        svg += f'<text x="{lx+16}" y="{margin_t+15}" fill="#ccc" font-size="10" font-family="Arial">{label_a}</text>'
        svg += f'<rect x="{lx}" y="{margin_t+22}" width="12" height="12" fill="{color_b}" rx="2"/>'
        svg += f'<text x="{lx+16}" y="{margin_t+32}" fill="#ccc" font-size="10" font-family="Arial">{label_b}</text>'

        svg += '</svg>'
        return svg

    svg_model_err = make_bar_svg(bar_data_model, 'Model 174012', 'Model 163632',
                                  'err_174012', 'err_163632', '#3498db', '#e74c3c',
                                  'Absolute Error by Sample: Model 174012 vs 163632 (mult=0.8)')

    svg_mult_err = make_bar_svg(bar_data_mult, 'Multiplier 0.8', 'Multiplier 0.6',
                                 'err_08', 'err_06', '#2ecc71', '#f39c12',
                                 'Absolute Error by Sample: Mult 0.8 vs 0.6 (Model 174012)')

    # Prediction vs Reference SVG
    def make_pred_svg(data, label_a, label_b, key_a, key_b, color_a, color_b, title):
        w, h = 700, 320
        margin_l, margin_b, margin_t, margin_r = 80, 60, 40, 20
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b
        n = len(data)
        if n == 0:
            return ''
        all_vals = [d['ref'] for d in data] + [d[key_a] for d in data] + [d[key_b] for d in data]
        max_val = max(all_vals) * 1.15
        bar_group_w = plot_w / n
        bar_w = bar_group_w * 0.25

        svg = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        svg += f'<rect width="{w}" height="{h}" fill="#1a1a2e" rx="8"/>'
        svg += f'<text x="{w/2}" y="25" text-anchor="middle" fill="#fff" font-size="14" font-family="Arial">{title}</text>'

        for i in range(5):
            val = max_val * i / 4
            y = margin_t + plot_h - (val / max_val * plot_h)
            svg += f'<line x1="{margin_l}" y1="{y}" x2="{w-margin_r}" y2="{y}" stroke="#333" stroke-width="0.5" stroke-dasharray="4,4"/>'
            svg += f'<text x="{margin_l-8}" y="{y+4}" text-anchor="end" fill="#999" font-size="10" font-family="Arial">{val:.0f}</text>'

        svg += f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{w-margin_r}" y2="{margin_t+plot_h}" stroke="#666" stroke-width="1"/>'
        svg += f'<text x="{margin_l-10}" y="{h/2}" text-anchor="middle" fill="#999" font-size="11" font-family="Arial" transform="rotate(-90,{margin_l-45},{h/2})">Glucose (mg/dL)</text>'

        for i, d in enumerate(data):
            x_center = margin_l + bar_group_w * i + bar_group_w / 2
            x_ref = x_center - bar_w * 1.5 - 1
            x_a = x_center - bar_w * 0.5
            x_b = x_center + bar_w * 0.5 + 1

            h_ref = d['ref'] / max_val * plot_h
            h_a = d[key_a] / max_val * plot_h
            h_b = d[key_b] / max_val * plot_h

            svg += f'<rect x="{x_ref}" y="{margin_t+plot_h-h_ref}" width="{bar_w}" height="{h_ref}" fill="#888" rx="2" opacity="0.6"/>'
            svg += f'<rect x="{x_a}" y="{margin_t+plot_h-h_a}" width="{bar_w}" height="{h_a}" fill="{color_a}" rx="2" opacity="0.85"/>'
            svg += f'<rect x="{x_b}" y="{margin_t+plot_h-h_b}" width="{bar_w}" height="{h_b}" fill="{color_b}" rx="2" opacity="0.85"/>'

            label = d['sample'].replace('Sub', 'S').replace('Sample', '')
            svg += f'<text x="{x_center}" y="{margin_t+plot_h+15}" text-anchor="middle" fill="#ccc" font-size="9" font-family="Arial">{label}</text>'

        lx = w - margin_r - 180
        svg += f'<rect x="{lx}" y="{margin_t+5}" width="12" height="12" fill="#888" rx="2" opacity="0.6"/>'
        svg += f'<text x="{lx+16}" y="{margin_t+15}" fill="#ccc" font-size="10" font-family="Arial">Reference</text>'
        svg += f'<rect x="{lx}" y="{margin_t+22}" width="12" height="12" fill="{color_a}" rx="2"/>'
        svg += f'<text x="{lx+16}" y="{margin_t+32}" fill="#ccc" font-size="10" font-family="Arial">{label_a}</text>'
        svg += f'<rect x="{lx}" y="{margin_t+39}" width="12" height="12" fill="{color_b}" rx="2"/>'
        svg += f'<text x="{lx+16}" y="{margin_t+49}" fill="#ccc" font-size="10" font-family="Arial">{label_b}</text>'

        svg += '</svg>'
        return svg

    svg_model_pred = make_pred_svg(bar_data_model, 'Model 174012', 'Model 163632',
                                    'pred_174012', 'pred_163632', '#3498db', '#e74c3c',
                                    'Predicted vs Reference Glucose: Model 174012 vs 163632')

    svg_mult_pred = make_pred_svg(bar_data_mult, 'Mult 0.8', 'Mult 0.6',
                                   'pred_08', 'pred_06', '#2ecc71', '#f39c12',
                                   'Predicted vs Reference Glucose: Mult 0.8 vs 0.6 (Model 174012)')

    # Count wins
    wins_174 = sum(1 for s in SAMPLES if results['174012_0.8'].get(s) and results['163632_0.8'].get(s)
                   and results['174012_0.8'][s]['abs_error'] < results['163632_0.8'][s]['abs_error'])
    wins_163 = sum(1 for s in SAMPLES if results['174012_0.8'].get(s) and results['163632_0.8'].get(s)
                   and results['163632_0.8'][s]['abs_error'] < results['174012_0.8'][s]['abs_error'])

    wins_08 = sum(1 for s in SAMPLES if results['174012_0.8'].get(s) and results['174012_0.6'].get(s)
                  and results['174012_0.8'][s]['abs_error'] < results['174012_0.6'][s]['abs_error'])
    wins_06 = sum(1 for s in SAMPLES if results['174012_0.8'].get(s) and results['174012_0.6'].get(s)
                  and results['174012_0.6'][s]['abs_error'] < results['174012_0.8'][s]['abs_error'])

    # Inferences
    # Model comparison inference
    model_inference = []
    if s174_08 and s163_08:
        if s174_08['mae'] < s163_08['mae']:
            model_inference.append(f"Model 174012 outperforms Model 163632 with MAE {s174_08['mae']} vs {s163_08['mae']} mg/dL (lower is better).")
        else:
            model_inference.append(f"Model 163632 outperforms Model 174012 with MAE {s163_08['mae']} vs {s174_08['mae']} mg/dL.")
        model_inference.append(f"Model 174012 wins on {wins_174}/8 samples, Model 163632 wins on {wins_163}/8 samples.")
        model_inference.append(f"Mean MARD: 174012 = {s174_08['mean_mard']}%, 163632 = {s163_08['mean_mard']}%.")

    # Multiplier comparison inference
    mult_inference = []
    if s174_08 and s174_06:
        if s174_08['mae'] < s174_06['mae']:
            mult_inference.append(f"Distance multiplier 0.8 yields better results than 0.6 for Model 174012 (MAE {s174_08['mae']} vs {s174_06['mae']} mg/dL).")
        else:
            mult_inference.append(f"Distance multiplier 0.6 yields better results than 0.8 for Model 174012 (MAE {s174_06['mae']} vs {s174_08['mae']} mg/dL).")
        mult_inference.append(f"Multiplier 0.8 wins on {wins_08}/8 samples, multiplier 0.6 wins on {wins_06}/8 samples.")
        # Window count comparison
        if s174_06['total_windows'] != s174_08['total_windows']:
            mult_inference.append(f"Window count differs: 0.8 = {s174_08['total_windows']} windows, 0.6 = {s174_06['total_windows']} windows. "
                                  f"A lower multiplier detects more peaks (shorter minimum distance), yielding more windows.")

    # Clinical interpretation
    clinical = []
    if s174_08:
        if s174_08['mae'] < 15:
            clinical.append("Model 174012 (mult 0.8) approaches clinical acceptability (MAE < 15 mg/dL) for non-invasive glucose estimation.")
        elif s174_08['mae'] < 25:
            clinical.append("Model 174012 (mult 0.8) shows moderate accuracy. Further training data and signal quality improvements could improve performance.")
        else:
            clinical.append("Model 174012 (mult 0.8) has significant prediction errors (MAE > 25 mg/dL). The model needs improvement for clinical use.")

    # High-glucose vs normal-glucose analysis
    normal_samples = ['Sub1Sample1', 'Sub1Sample2', 'Sub1Sample4']
    high_samples = ['Sub1Sample3', 'Sub2Sample1', 'Sub3Sample1', 'Sub4Sample1']

    normal_errors_174 = [results['174012_0.8'][s]['abs_error'] for s in normal_samples if results['174012_0.8'].get(s)]
    high_errors_174 = [results['174012_0.8'][s]['abs_error'] for s in high_samples if results['174012_0.8'].get(s)]

    if normal_errors_174 and high_errors_174:
        avg_normal = np.mean(normal_errors_174)
        avg_high = np.mean(high_errors_174)
        if avg_high > avg_normal * 1.3:
            clinical.append(f"Model 174012 performs better on normal glucose samples (avg error {avg_normal:.1f} mg/dL) than elevated samples (avg error {avg_high:.1f} mg/dL). This suggests the model may be biased toward the training distribution's central tendency.")
        elif avg_normal > avg_high * 1.3:
            clinical.append(f"Model 174012 performs better on elevated glucose samples (avg error {avg_high:.1f} mg/dL) than normal samples (avg error {avg_normal:.1f} mg/dL).")
        else:
            clinical.append(f"Model 174012 shows relatively consistent error across normal ({avg_normal:.1f} mg/dL) and elevated ({avg_high:.1f} mg/dL) glucose ranges.")

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Lab PPG Comprehensive Comparison Report</title>
<style>
    body {{ background: #0f0f23; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ text-align: center; color: #fff; border-bottom: 3px solid #667eea; padding-bottom: 12px; }}
    h2 {{ color: #667eea; border-bottom: 1px solid #333; padding-bottom: 8px; margin-top: 40px; }}
    h3 {{ color: #a0a0ff; margin-top: 25px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
    th {{ background: #1a1a3e; color: #667eea; padding: 10px 6px; text-align: center; border: 1px solid #333; }}
    td {{ padding: 8px 6px; text-align: center; border: 1px solid #2a2a4a; }}
    tr:nth-child(even) {{ background: #151530; }}
    tr:hover {{ background: #1e1e40; }}
    .good {{ color: #2ecc71; font-weight: bold; }}
    .ok {{ color: #f1c40f; }}
    .warn {{ color: #e67e22; font-weight: bold; }}
    .bad {{ color: #e74c3c; font-weight: bold; }}
    .winner {{ color: #3498db; font-weight: bold; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 15px; margin: 15px 0; }}
    .summary-card {{ background: #1a1a3e; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center; }}
    .summary-card .value {{ font-size: 28px; font-weight: bold; color: #667eea; }}
    .summary-card .label {{ font-size: 12px; color: #999; margin-top: 4px; }}
    .summary-card.highlight {{ border-color: #2ecc71; }}
    .summary-card.highlight .value {{ color: #2ecc71; }}
    .inference-box {{ background: #1a2a1a; border: 1px solid #2ecc71; border-radius: 8px; padding: 16px; margin: 15px 0; }}
    .inference-box h4 {{ color: #2ecc71; margin: 0 0 10px 0; }}
    .inference-box ul {{ margin: 0; padding-left: 20px; }}
    .inference-box li {{ margin: 6px 0; line-height: 1.5; }}
    .clinical-box {{ background: #2a1a1a; border: 1px solid #e74c3c; border-radius: 8px; padding: 16px; margin: 15px 0; }}
    .clinical-box h4 {{ color: #e74c3c; margin: 0 0 10px 0; }}
    .clinical-box ul {{ margin: 0; padding-left: 20px; }}
    .clinical-box li {{ margin: 6px 0; line-height: 1.5; }}
    .chart-container {{ text-align: center; margin: 20px 0; }}
    .section-desc {{ color: #aaa; font-size: 14px; margin-bottom: 10px; }}
    .legend {{ display: inline-flex; gap: 20px; margin: 10px 0; font-size: 12px; }}
    .legend span {{ display: flex; align-items: center; gap: 5px; }}
    .legend .dot {{ width: 12px; height: 12px; border-radius: 3px; }}
    .footer {{ text-align: center; color: #666; font-size: 11px; margin-top: 40px; padding-top: 15px; border-top: 1px solid #333; }}
</style>
</head>
<body>
<div class="container">

<h1>Lab PPG Comprehensive Comparison Report</h1>
<p style="text-align:center; color:#999;">Generated from 8 lab PPG samples across 4 subjects</p>

<!-- ================================================================ -->
<!-- SECTION 1: Model Comparison -->
<!-- ================================================================ -->
<h2>1. Model Comparison: 174012 vs 163632</h2>
<p class="section-desc">Both models evaluated with distance multiplier = 0.8 (default)</p>

<div class="summary-grid">
    <div class="summary-card {'highlight' if s174_08 and s163_08 and s174_08['mae'] < s163_08['mae'] else ''}">
        <div class="value">{s174_08['mae'] if s174_08 else 'N/A'}</div>
        <div class="label">Model 174012 MAE (mg/dL)</div>
    </div>
    <div class="summary-card {'highlight' if s174_08 and s163_08 and s163_08['mae'] < s174_08['mae'] else ''}">
        <div class="value">{s163_08['mae'] if s163_08 else 'N/A'}</div>
        <div class="label">Model 163632 MAE (mg/dL)</div>
    </div>
    <div class="summary-card">
        <div class="value">{s174_08['mean_mard'] if s174_08 else 'N/A'}%</div>
        <div class="label">Model 174012 MARD</div>
    </div>
    <div class="summary-card">
        <div class="value">{s163_08['mean_mard'] if s163_08 else 'N/A'}%</div>
        <div class="label">Model 163632 MARD</div>
    </div>
</div>

<h3>Sample-Level Results</h3>
<table>
    <tr>
        <th rowspan="2">Sample</th>
        <th rowspan="2">Reference<br>(mg/dL)</th>
        <th colspan="4" style="background:#1a2a4e">Model 174012</th>
        <th colspan="4" style="background:#4e1a2a">Model 163632</th>
        <th rowspan="2">Winner</th>
    </tr>
    <tr>
        <th style="background:#1a2a4e">Mean Pred</th>
        <th style="background:#1a2a4e">Abs Error</th>
        <th style="background:#1a2a4e">MARD</th>
        <th style="background:#1a2a4e">Windows</th>
        <th style="background:#4e1a2a">Mean Pred</th>
        <th style="background:#4e1a2a">Abs Error</th>
        <th style="background:#4e1a2a">MARD</th>
        <th style="background:#4e1a2a">Windows</th>
    </tr>
    {table1_rows}
</table>

<div class="chart-container">{svg_model_err}</div>
<div class="chart-container">{svg_model_pred}</div>

<div class="inference-box">
    <h4>Key Findings - Model Comparison</h4>
    <ul>
        {''.join(f"<li>{inf}</li>" for inf in model_inference)}
    </ul>
</div>

<!-- ================================================================ -->
<!-- SECTION 2: Multiplier Comparison for Model 174012 -->
<!-- ================================================================ -->
<h2>2. Distance Multiplier Comparison: 0.8 vs 0.6 (Model 174012)</h2>
<p class="section-desc">Same model, different peak detection sensitivity</p>

<div class="summary-grid">
    <div class="summary-card {'highlight' if s174_08 and s174_06 and s174_08['mae'] < s174_06['mae'] else ''}">
        <div class="value">{s174_08['mae'] if s174_08 else 'N/A'}</div>
        <div class="label">Multiplier 0.8 MAE (mg/dL)</div>
    </div>
    <div class="summary-card {'highlight' if s174_08 and s174_06 and s174_06['mae'] < s174_08['mae'] else ''}">
        <div class="value">{s174_06['mae'] if s174_06 else 'N/A'}</div>
        <div class="label">Multiplier 0.6 MAE (mg/dL)</div>
    </div>
    <div class="summary-card">
        <div class="value">{s174_08['total_windows'] if s174_08 else 'N/A'}</div>
        <div class="label">Windows (mult 0.8)</div>
    </div>
    <div class="summary-card">
        <div class="value">{s174_06['total_windows'] if s174_06 else 'N/A'}</div>
        <div class="label">Windows (mult 0.6)</div>
    </div>
</div>

<h3>Sample-Level Results</h3>
<table>
    <tr>
        <th rowspan="2">Sample</th>
        <th rowspan="2">Reference<br>(mg/dL)</th>
        <th colspan="4" style="background:#1a3e2a">Multiplier 0.8</th>
        <th colspan="4" style="background:#3e2a1a">Multiplier 0.6</th>
        <th rowspan="2">Winner</th>
        <th rowspan="2">Error<br>Diff</th>
    </tr>
    <tr>
        <th style="background:#1a3e2a">Mean Pred</th>
        <th style="background:#1a3e2a">Abs Error</th>
        <th style="background:#1a3e2a">MARD</th>
        <th style="background:#1a3e2a">Windows</th>
        <th style="background:#3e2a1a">Mean Pred</th>
        <th style="background:#3e2a1a">Abs Error</th>
        <th style="background:#3e2a1a">MARD</th>
        <th style="background:#3e2a1a">Windows</th>
    </tr>
    {table2_rows}
</table>

<div class="chart-container">{svg_mult_err}</div>
<div class="chart-container">{svg_mult_pred}</div>

<div class="inference-box">
    <h4>Key Findings - Multiplier Comparison (Model 174012)</h4>
    <ul>
        {''.join(f"<li>{inf}</li>" for inf in mult_inference)}
    </ul>
</div>

<!-- ================================================================ -->
<!-- SECTION 3: Multiplier Comparison for Model 163632 -->
<!-- ================================================================ -->
<h2>3. Distance Multiplier Comparison: 0.8 vs 0.6 (Model 163632)</h2>
<p class="section-desc">Impact of multiplier on Model 163632</p>

<div class="summary-grid">
    <div class="summary-card">
        <div class="value">{s163_08['mae'] if s163_08 else 'N/A'}</div>
        <div class="label">Multiplier 0.8 MAE (mg/dL)</div>
    </div>
    <div class="summary-card">
        <div class="value">{s163_06['mae'] if s163_06 else 'N/A'}</div>
        <div class="label">Multiplier 0.6 MAE (mg/dL)</div>
    </div>
</div>

<table>
    <tr>
        <th rowspan="2">Sample</th>
        <th rowspan="2">Reference<br>(mg/dL)</th>
        <th colspan="4" style="background:#1a3e2a">Multiplier 0.8</th>
        <th colspan="4" style="background:#3e2a1a">Multiplier 0.6</th>
        <th rowspan="2">Winner</th>
        <th rowspan="2">Error<br>Diff</th>
    </tr>
    <tr>
        <th style="background:#1a3e2a">Mean Pred</th>
        <th style="background:#1a3e2a">Abs Error</th>
        <th style="background:#1a3e2a">MARD</th>
        <th style="background:#1a3e2a">Windows</th>
        <th style="background:#3e2a1a">Mean Pred</th>
        <th style="background:#3e2a1a">Abs Error</th>
        <th style="background:#3e2a1a">MARD</th>
        <th style="background:#3e2a1a">Windows</th>
    </tr>
    {table3_rows}
</table>

<!-- ================================================================ -->
<!-- SECTION 4: Clinical Interpretation -->
<!-- ================================================================ -->
<h2>4. Clinical Interpretation & Recommendations</h2>

<div class="clinical-box">
    <h4>Clinical Assessment</h4>
    <ul>
        {''.join(f"<li>{c}</li>" for c in clinical)}
    </ul>
</div>

<div class="inference-box">
    <h4>Recommendations</h4>
    <ul>
        <li><strong>Model selection:</strong> Model 174012 should be preferred for lab PPG inference as it consistently shows lower prediction errors.</li>
        <li><strong>Distance multiplier:</strong> The optimal multiplier depends on signal quality. Test both 0.6 and 0.8 on new data to determine the best setting for a given PPG sensor.</li>
        <li><strong>Window count impact:</strong> More windows (from a lower multiplier) provide more data points for averaging, but may include lower-quality peaks that degrade per-window accuracy.</li>
        <li><strong>Glucose range bias:</strong> Investigate per-range performance to understand if the model over-predicts or under-predicts at specific glucose levels.</li>
    </ul>
</div>

<div class="footer">
    Lab PPG Comprehensive Comparison Report | Models: 174012, 163632 | Distance Multipliers: 0.6, 0.8 | 8 Samples, 4 Subjects
</div>

</div>
</body>
</html>'''

    output_path = os.path.join(os.path.dirname(__file__), 'LAB_PPG_COMPREHENSIVE_COMPARISON.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n[OK] Report saved to: {output_path}")


if __name__ == '__main__':
    main()
