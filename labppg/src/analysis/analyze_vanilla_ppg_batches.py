"""
Vanilla PPG Batch Inference Report
===================================
Combines Batch 1 and Batch 2 results into a comprehensive HTML report
with Clarke Error Grid analysis, per-person breakdown, and per-batch comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path


BASE = Path(r"C:\IITM\vitalDB\data\LABPPG")

SUMMARIES = [
    (BASE / "VanillaPPG_Batch_1_500Hz" / "results" / "batch_summary.csv",        "Batch 1", "Red"),
    (BASE / "VanillaPPG_Batch_1_500Hz" / "results" / "iR_PPG" / "batch_summary.csv", "Batch 1", "IR"),
    (BASE / "VanillaPPG_Batch_2_500Hz" / "results" / "batch_summary.csv",        "Batch 2", "Red"),
    (BASE / "VanillaPPG_Batch_2_500Hz" / "results" / "iR_PPG" / "batch_summary.csv", "Batch 2", "IR"),
]

OUTPUT_DIR = BASE


def clarke_zone(ref, pred):
    if (ref >= 100 and abs(pred - ref) / ref <= 0.20) or \
       (ref < 100 and abs(pred - ref) <= 20):
        return 'A'
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return 'E'
    if (ref >= 240 and pred <= 70) or (ref <= 70 and pred >= 180):
        return 'D'
    if ref >= 70 and ref <= 180 and (pred > 180 or pred < 70):
        if (ref > 180 and pred < 70) or (ref < 70 and pred > 180):
            return 'C'
        return 'B'
    return 'B'


def zone_analysis(refs, preds):
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for r, p in zip(refs, preds):
        zones[clarke_zone(r, p)] += 1
    total = sum(zones.values())
    pcts = {k: (v / total * 100) if total > 0 else 0 for k, v in zones.items()}
    return zones, pcts


def clarke_svg(refs, preds, persons, width=560, height=560):
    margin = 60
    ps = width - 2 * margin
    mx = 350

    def sx(v): return margin + (min(v, mx) / mx) * ps
    def sy(v): return height - margin - (min(v, mx) / mx) * ps

    # Assign colors per person
    unique_persons = sorted(set(persons))
    palette = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#1abc9c',
               '#3498db', '#9b59b6', '#e84393', '#fd79a8', '#00cec9',
               '#6c5ce7', '#d63031', '#74b9ff', '#a29bfe', '#55efc4']
    pcolor = {p: palette[i % len(palette)] for i, p in enumerate(unique_persons)}

    s = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
         f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>']

    # Grid
    for v in range(50, mx + 1, 50):
        a = 0.15 if v % 100 else 0.3
        s.append(f'<line x1="{sx(v)}" y1="{margin}" x2="{sx(v)}" y2="{height-margin}" stroke="#555" stroke-width="0.5" opacity="{a}"/>')
        s.append(f'<line x1="{margin}" y1="{sy(v)}" x2="{width-margin}" y2="{sy(v)}" stroke="#555" stroke-width="0.5" opacity="{a}"/>')

    # Axes
    s.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#888" stroke-width="1.5"/>')
    s.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#888" stroke-width="1.5"/>')

    # Perfect line
    s.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(mx)}" y2="{sy(mx)}" stroke="#667eea" stroke-width="1.5" opacity="0.5" stroke-dasharray="8,4"/>')

    # Zone boundaries (+/-20% / +/-20 mg/dL)
    s.append(f'<line x1="{sx(0)}" y1="{sy(20)}" x2="{sx(100)}" y2="{sy(120)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    s.append(f'<line x1="{sx(100)}" y1="{sy(120)}" x2="{sx(mx)}" y2="{sy(mx*1.2)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    s.append(f'<line x1="{sx(20)}" y1="{sy(0)}" x2="{sx(100)}" y2="{sy(80)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')
    s.append(f'<line x1="{sx(100)}" y1="{sy(80)}" x2="{sx(mx)}" y2="{sy(mx*0.8)}" stroke="#aaa" stroke-width="1" stroke-dasharray="5,3"/>')

    # Hypo/hyper lines
    for v, c in [(70, '#e74c3c'), (180, '#f39c12')]:
        s.append(f'<line x1="{sx(v)}" y1="{margin}" x2="{sx(v)}" y2="{height-margin}" stroke="{c}" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')
        s.append(f'<line x1="{margin}" y1="{sy(v)}" x2="{width-margin}" y2="{sy(v)}" stroke="{c}" stroke-width="0.8" opacity="0.3" stroke-dasharray="5,4"/>')

    # Zone labels
    s.append(f'<text x="{sx(140)}" y="{sy(140)}" font-size="30" fill="rgba(255,255,255,0.1)" font-weight="bold" font-family="Segoe UI">A</text>')
    s.append(f'<text x="{sx(70)}" y="{sy(250)}" font-size="22" fill="rgba(255,255,255,0.08)" font-weight="bold" font-family="Segoe UI">B</text>')
    s.append(f'<text x="{sx(250)}" y="{sy(70)}" font-size="22" fill="rgba(255,255,255,0.08)" font-weight="bold" font-family="Segoe UI">B</text>')

    # Data points
    np.random.seed(42)
    s.append('<g>')
    for r, p, person in zip(refs, preds, persons):
        color = pcolor.get(person, '#3498db')
        x = sx(r + np.random.uniform(-2, 2))
        y = sy(p)
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" opacity="0.75" stroke="#fff" stroke-width="0.5"/>')
    s.append('</g>')

    # Ticks
    for v in range(0, mx + 1, 50):
        s.append(f'<text x="{sx(v)}" y="{height-margin+16}" text-anchor="middle" font-size="10" fill="#aaa" font-family="Segoe UI">{v}</text>')
        s.append(f'<text x="{margin-10}" y="{sy(v)+4}" text-anchor="end" font-size="10" fill="#aaa" font-family="Segoe UI">{v}</text>')

    s.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12" fill="#ccc" font-family="Segoe UI">Reference Glucose (mg/dL)</text>')
    s.append(f'<text x="13" y="{height/2}" text-anchor="middle" font-size="12" fill="#ccc" font-family="Segoe UI" transform="rotate(-90,13,{height/2})">Predicted Mean Glucose (mg/dL)</text>')
    s.append('</svg>')
    return '\n'.join(s), pcolor


def scatter_ref_vs_pred_svg(refs, preds, width=500, height=350):
    """Simple ref vs pred scatter with regression line."""
    margin = 55
    ps = width - 2 * margin
    ph = height - 2 * margin
    mx = max(max(refs), max(preds)) * 1.1

    def sx(v): return margin + (v / mx) * ps
    def sy(v): return height - margin - (v / mx) * ph

    s = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
         f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>']

    # Perfect line
    s.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(mx)}" y2="{sy(mx)}" stroke="#667eea" stroke-width="1.5" opacity="0.4" stroke-dasharray="6,3"/>')

    # Regression line
    coeffs = np.polyfit(refs, preds, 1)
    y0 = coeffs[0] * 0 + coeffs[1]
    y1 = coeffs[0] * mx + coeffs[1]
    s.append(f'<line x1="{sx(0)}" y1="{sy(y0)}" x2="{sx(mx)}" y2="{sy(y1)}" stroke="#2ecc71" stroke-width="1.5" opacity="0.7"/>')

    # Points
    for r, p in zip(refs, preds):
        s.append(f'<circle cx="{sx(r):.1f}" cy="{sy(p):.1f}" r="4" fill="#3498db" opacity="0.7"/>')

    # Ticks
    step = 50
    for v in range(0, int(mx) + 1, step):
        s.append(f'<text x="{sx(v)}" y="{height-margin+14}" text-anchor="middle" font-size="9" fill="#aaa" font-family="Segoe UI">{v}</text>')
        s.append(f'<text x="{margin-8}" y="{sy(v)+3}" text-anchor="end" font-size="9" fill="#aaa" font-family="Segoe UI">{v}</text>')

    s.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#888" stroke-width="1"/>')
    s.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#888" stroke-width="1"/>')
    s.append(f'<text x="{width/2}" y="{height-5}" text-anchor="middle" font-size="11" fill="#ccc" font-family="Segoe UI">Reference (mg/dL)</text>')
    s.append(f'<text x="12" y="{height/2}" text-anchor="middle" font-size="11" fill="#ccc" font-family="Segoe UI" transform="rotate(-90,12,{height/2})">Predicted (mg/dL)</text>')
    s.append('</svg>')
    return '\n'.join(s)


def error_bar_svg(person_stats, width=700, height=None):
    """Horizontal MAE bars per person."""
    n = len(person_stats)
    if height is None:
        height = max(200, n * 32 + 60)
    margin_l, margin_r, margin_t, margin_b = 110, 30, 20, 30
    pw = width - margin_l - margin_r
    ph = height - margin_t - margin_b
    bar_h = min(22, ph / n - 4)
    max_mae = max(s['mae'] for s in person_stats) * 1.15

    s = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
         f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>']

    for i, st in enumerate(person_stats):
        y = margin_t + i * (bar_h + 4)
        bw = (st['mae'] / max_mae) * pw
        color = '#2ecc71' if st['mae'] < 20 else '#f39c12' if st['mae'] < 35 else '#e74c3c'
        s.append(f'<rect x="{margin_l}" y="{y:.0f}" width="{bw:.0f}" height="{bar_h:.0f}" rx="3" fill="{color}" opacity="0.85"/>')
        s.append(f'<text x="{margin_l-6}" y="{y+bar_h/2+4:.0f}" text-anchor="end" font-size="10" fill="#ccc" font-family="Segoe UI">{st["name"]} ({st["n"]})</text>')
        s.append(f'<text x="{margin_l+bw+5:.0f}" y="{y+bar_h/2+4:.0f}" font-size="10" fill="#eee" font-family="Segoe UI">{st["mae"]:.1f}</text>')

    s.append('</svg>')
    return '\n'.join(s)


def main():
    print("=" * 80)
    print("VANILLA PPG BATCH INFERENCE REPORT")
    print("=" * 80)

    # Load all summaries (Batch 1 Red, Batch 1 IR, Batch 2 Red, Batch 2 IR)
    frames = []
    for path, batch, ppg_type in SUMMARIES:
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        tmp = pd.read_csv(path)
        tmp['batch'] = batch
        tmp['ppg_type'] = ppg_type
        frames.append(tmp)
        print(f"  Loaded {batch} {ppg_type}: {len(tmp)} rows")
    df = pd.concat(frames, ignore_index=True)

    ok = df[df['status'] == 'OK'].copy()
    failed = df[df['status'] != 'OK'].copy()

    # Filter out 'Unknown' person
    ok = ok[ok['name'] != 'Unknown']

    refs = ok['ref_glucose'].values
    preds = ok['pred_mean'].values
    errors = preds - refs
    abs_errors = np.abs(errors)
    persons = ok['name'].values

    n_total = len(df)
    n_ok = len(ok)
    n_fail = len(failed)
    n_persons = ok['name'].nunique()

    # Metrics
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mard = np.mean(abs_errors / refs) * 100
    median_ae = np.median(abs_errors)
    mean_bias = np.mean(errors)

    # R2
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((refs - np.mean(refs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Clarke zones (using mean predictions per sample)
    zones, pcts = zone_analysis(refs, preds)
    ab_pct = pcts['A'] + pcts['B']

    # Accuracy thresholds
    within_15 = np.mean(abs_errors <= 15) * 100
    within_20 = np.mean(abs_errors <= 20) * 100
    within_30 = np.mean(abs_errors <= 30) * 100
    within_20pct = np.mean(np.where(refs >= 100,
                                     abs_errors / refs <= 0.20,
                                     abs_errors <= 20)) * 100

    # Per-person stats
    person_stats = []
    for name in sorted(ok['name'].unique()):
        sub = ok[ok['name'] == name]
        p_mae = sub['pred_abs_error'].mean()
        p_mard = sub['pred_mard'].mean()
        p_refs = sub['ref_glucose'].values
        p_preds = sub['pred_mean'].values
        p_zones, p_pcts = zone_analysis(p_refs, p_preds)
        person_stats.append({
            'name': name,
            'n': len(sub),
            'ref_range': f"{p_refs.min():.0f}-{p_refs.max():.0f}",
            'mae': p_mae,
            'mard': p_mard,
            'bias': np.mean(p_preds - p_refs),
            'pred_mean': p_preds.mean(),
            'zone_a': p_pcts['A'],
            'zone_ab': p_pcts['A'] + p_pcts['B'],
        })
    person_stats.sort(key=lambda x: x['mae'])

    # Per-batch stats
    batch_stats = []
    for batch_name in ['Batch 1', 'Batch 2']:
        sub = ok[ok['batch'] == batch_name]
        if len(sub) == 0:
            continue
        b_refs = sub['ref_glucose'].values
        b_preds = sub['pred_mean'].values
        b_errors = b_preds - b_refs
        b_zones, b_pcts = zone_analysis(b_refs, b_preds)
        batch_stats.append({
            'batch': batch_name,
            'n_total': len(df[df['batch'] == batch_name]),
            'n_ok': len(sub),
            'n_fail': len(df[(df['batch'] == batch_name) & (df['status'] != 'OK')]),
            'mae': np.mean(np.abs(b_errors)),
            'rmse': np.sqrt(np.mean(b_errors**2)),
            'mard': np.mean(np.abs(b_errors) / b_refs) * 100,
            'bias': np.mean(b_errors),
            'zone_a': b_pcts['A'],
            'zone_ab': b_pcts['A'] + b_pcts['B'],
        })

    # Per PPG-type stats (Red vs IR)
    ppg_type_stats = []
    for pt in ['Red', 'IR']:
        sub = ok[ok['ppg_type'] == pt]
        if len(sub) == 0:
            continue
        p_refs = sub['ref_glucose'].values
        p_preds = sub['pred_mean'].values
        p_errors = p_preds - p_refs
        p_zones, p_pcts = zone_analysis(p_refs, p_preds)
        ppg_type_stats.append({
            'ppg_type': pt,
            'n_total': len(df[df['ppg_type'] == pt]),
            'n_ok': len(sub),
            'n_fail': len(df[(df['ppg_type'] == pt) & (df['status'] != 'OK')]),
            'mae': np.mean(np.abs(p_errors)),
            'rmse': np.sqrt(np.mean(p_errors**2)),
            'mard': np.mean(np.abs(p_errors) / p_refs) * 100,
            'bias': np.mean(p_errors),
            'zone_a': p_pcts['A'],
            'zone_ab': p_pcts['A'] + p_pcts['B'],
        })

    # Cross-comparison: Batch x PPG type
    cross_stats = []
    for batch_name in ['Batch 1', 'Batch 2']:
        for pt in ['Red', 'IR']:
            sub = ok[(ok['batch'] == batch_name) & (ok['ppg_type'] == pt)]
            if len(sub) == 0:
                cross_stats.append({
                    'label': f"{batch_name} {pt}",
                    'n_ok': 0, 'mae': 0, 'mard': 0, 'bias': 0,
                    'zone_a': 0, 'zone_ab': 0,
                })
                continue
            c_refs = sub['ref_glucose'].values
            c_preds = sub['pred_mean'].values
            c_errors = c_preds - c_refs
            c_zones, c_pcts = zone_analysis(c_refs, c_preds)
            cross_stats.append({
                'label': f"{batch_name} {pt}",
                'n_ok': len(sub),
                'mae': np.mean(np.abs(c_errors)),
                'mard': np.mean(np.abs(c_errors) / c_refs) * 100,
                'bias': np.mean(c_errors),
                'zone_a': c_pcts['A'],
                'zone_ab': c_pcts['A'] + c_pcts['B'],
            })

    # SVGs
    clarke_svg_content, pcolor = clarke_svg(refs, preds, persons)
    scatter_svg = scatter_ref_vs_pred_svg(refs, preds)
    bar_svg = error_bar_svg(person_stats)

    print(f"Samples: {n_ok}/{n_total}")
    print(f"MAE: {mae:.1f}, MARD: {mard:.1f}%, R2: {r2:.3f}")
    print(f"Clarke A: {pcts['A']:.1f}%, A+B: {ab_pct:.1f}%")

    # ---- Failed samples summary ----
    fail_rows = ''
    for _, row in failed.iterrows():
        err_msg = str(row.get('error', 'Unknown'))[:60]
        fail_rows += f"""
        <tr>
            <td style="text-align:left;">{row.get('name','?')}</td>
            <td>{row.get('date','?')}</td><td>{row.get('time','?')}</td>
            <td>{row.get('batch','?')}</td>
            <td style="text-align:left;color:#e74c3c;">{err_msg}</td>
        </tr>"""

    # ---- Build HTML ----
    ab_class = 'good' if ab_pct >= 95 else 'warn' if ab_pct >= 85 else 'bad'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Vanilla PPG Batch Inference Report</title>
<style>
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #e0e0e0; padding: 30px; max-width: 1350px; margin: 0 auto; line-height: 1.6;
}}
h1 {{ color: #667eea; text-align: center; font-size: 2.2em; margin-bottom: 5px; }}
h2 {{ color: #764ba2; border-bottom: 2px solid #764ba2; padding-bottom: 8px; margin-top: 45px; }}
h3 {{ color: #667eea; margin-top: 20px; }}
.subtitle {{ text-align: center; color: #95a5a6; font-size: 1.05em; margin-bottom: 30px; }}
.metric-container {{ display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin: 22px 0; }}
.metric-card {{
    background: rgba(102,126,234,0.12); border: 1px solid rgba(102,126,234,0.25);
    border-radius: 14px; padding: 16px 26px; text-align: center; min-width: 130px;
}}
.metric-label {{ color: #95a5a6; font-size: 0.82em; margin-bottom: 3px; }}
.metric-value {{ font-size: 1.7em; font-weight: bold; color: #667eea; }}
.metric-unit {{ font-size: 0.72em; color: #7f8c8d; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; background: rgba(255,255,255,0.02); border-radius: 10px; overflow: hidden; font-size: 0.88em; }}
th, td {{ padding: 9px 13px; text-align: center; border: 1px solid rgba(255,255,255,0.08); }}
th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; font-weight: 600; }}
tr:nth-child(even) {{ background: rgba(255,255,255,0.02); }}
.zone-A {{ color: #2ecc71; font-weight: bold; }} .zone-B {{ color: #3498db; font-weight: bold; }}
.zone-C {{ color: #f39c12; font-weight: bold; }} .zone-D {{ color: #e74c3c; font-weight: bold; }}
.zone-E {{ color: #9b59b6; font-weight: bold; }}
.good {{ color: #2ecc71; }} .warn {{ color: #f39c12; }} .bad {{ color: #e74c3c; }}
.chart-container {{ display: flex; justify-content: center; margin: 22px 0; background: rgba(0,0,0,0.2); border-radius: 14px; padding: 18px; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 18px 0; }}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.zone-bar {{ display: flex; height: 38px; border-radius: 10px; overflow: hidden; margin: 8px 0; }}
.zone-seg {{ display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.85em; color: #fff; }}
.insight-box {{
    background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
    border-left: 4px solid #667eea; padding: 16px 20px; margin: 16px 0; border-radius: 0 10px 10px 0;
}}
.insight-box h3 {{ margin-top: 0; }}
.legend {{ display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin: 12px 0; font-size: 0.85em; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; }}
.legend-color {{ width: 12px; height: 12px; border-radius: 50%; }}
.highlight-row {{ background: rgba(102,126,234,0.15) !important; font-weight: bold; }}
td.left {{ text-align: left; }}
</style>
</head><body>

<h1>Vanilla PPG Inference Report</h1>
<p class="subtitle">
    Batch 1 + Batch 2 | Red PPG + IR PPG | {n_ok} Samples ({n_fail} failed) |
    {n_persons} Persons | Glucose Range: {refs.min():.0f}&ndash;{refs.max():.0f} mg/dL
</p>

<!-- ===== SUMMARY ===== -->
<h2>1. Summary Metrics</h2>
<div class="metric-container">
    <div class="metric-card"><div class="metric-label">Samples</div><div class="metric-value">{n_ok}</div><div class="metric-unit">of {n_total} total</div></div>
    <div class="metric-card"><div class="metric-label">Persons</div><div class="metric-value">{n_persons}</div></div>
    <div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value">{mae:.1f}</div><div class="metric-unit">mg/dL</div></div>
    <div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">{rmse:.1f}</div><div class="metric-unit">mg/dL</div></div>
    <div class="metric-card"><div class="metric-label">MARD</div><div class="metric-value">{mard:.1f}%</div></div>
    <div class="metric-card"><div class="metric-label">R&sup2;</div><div class="metric-value">{r2:.3f}</div></div>
    <div class="metric-card"><div class="metric-label">Median AE</div><div class="metric-value">{median_ae:.1f}</div><div class="metric-unit">mg/dL</div></div>
    <div class="metric-card"><div class="metric-label">Bias</div><div class="metric-value">{mean_bias:+.1f}</div><div class="metric-unit">mg/dL</div></div>
</div>

<h3>Accuracy Thresholds</h3>
<table style="max-width:550px; margin:12px auto;">
    <tr><th>Threshold</th><th>% Within</th></tr>
    <tr><td>+/- 15 mg/dL</td><td>{within_15:.1f}%</td></tr>
    <tr><td>+/- 20 mg/dL</td><td>{within_20:.1f}%</td></tr>
    <tr><td>+/- 30 mg/dL</td><td>{within_30:.1f}%</td></tr>
    <tr><td>+/- 20% (ISO 15197)</td><td>{within_20pct:.1f}%</td></tr>
</table>

<!-- ===== CLARKE ERROR GRID ===== -->
<h2>2. Clarke Error Grid Analysis</h2>
<div class="two-col">
    <div class="chart-container">{clarke_svg_content}</div>
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
            <tr class="highlight-row"><td colspan="2">Clinically Acceptable (A+B)</td><td>{zones['A']+zones['B']}</td><td class="{ab_class}">{ab_pct:.1f}%</td></tr>
        </table>
        <div class="legend">
            {"".join(f'<div class="legend-item"><div class="legend-color" style="background:{pcolor[p]}"></div>{p}</div>' for p in sorted(pcolor.keys()))}
        </div>
    </div>
</div>

<!-- ===== REF vs PRED SCATTER ===== -->
<h2>3. Reference vs Predicted Glucose</h2>
<div class="two-col">
    <div class="chart-container">{scatter_svg}</div>
    <div>
        <div class="insight-box">
            <h3>Correlation Analysis</h3>
            <ul>
                <li>Pearson R: {np.corrcoef(refs, preds)[0,1]:.3f}</li>
                <li>R&sup2;: {r2:.3f}</li>
                <li>Regression slope: {np.polyfit(refs, preds, 1)[0]:.2f}</li>
                <li>Mean bias: {mean_bias:+.1f} mg/dL ({"overprediction" if mean_bias > 0 else "underprediction"})</li>
                <li>Prediction range: {preds.min():.0f}&ndash;{preds.max():.0f} vs ref: {refs.min():.0f}&ndash;{refs.max():.0f} mg/dL</li>
            </ul>
        </div>
        <p style="color:#95a5a6; font-size:0.85em;">
            Blue dots = samples | Green line = regression | Dashed blue = perfect prediction
        </p>
    </div>
</div>

<!-- ===== BATCH COMPARISON ===== -->
<h2>4. Batch Comparison</h2>
<table>
    <tr><th>Batch</th><th>Total Files</th><th>Succeeded</th><th>Failed</th><th>MAE</th><th>RMSE</th><th>MARD</th><th>Bias</th><th>Zone A</th><th>A+B</th></tr>"""

    for bs in batch_stats:
        html += f"""
    <tr>
        <td>{bs['batch']}</td><td>{bs['n_total']}</td><td>{bs['n_ok']}</td><td>{bs['n_fail']}</td>
        <td>{bs['mae']:.1f}</td><td>{bs['rmse']:.1f}</td><td>{bs['mard']:.1f}%</td>
        <td>{bs['bias']:+.1f}</td>
        <td class="zone-A">{bs['zone_a']:.1f}%</td>
        <td class="{'good' if bs['zone_ab']>=95 else 'warn'}">{bs['zone_ab']:.1f}%</td>
    </tr>"""

    html += f"""
    <tr class="highlight-row">
        <td>Combined</td><td>{n_total}</td><td>{n_ok}</td><td>{n_fail}</td>
        <td>{mae:.1f}</td><td>{rmse:.1f}</td><td>{mard:.1f}%</td><td>{mean_bias:+.1f}</td>
        <td class="zone-A">{pcts['A']:.1f}%</td><td class="{ab_class}">{ab_pct:.1f}%</td>
    </tr>
</table>

<!-- ===== RED vs IR PPG ===== -->
<h2>5. Red PPG vs IR PPG Comparison</h2>
<table>
    <tr><th>PPG Type</th><th>Total</th><th>Succeeded</th><th>Failed</th><th>MAE</th><th>RMSE</th><th>MARD</th><th>Bias</th><th>Zone A</th><th>A+B</th></tr>"""

    for ps in ppg_type_stats:
        html += f"""
    <tr>
        <td><b>{ps['ppg_type']} PPG</b></td><td>{ps['n_total']}</td><td>{ps['n_ok']}</td><td>{ps['n_fail']}</td>
        <td class="{'good' if ps['mae']<20 else 'warn' if ps['mae']<35 else 'bad'}">{ps['mae']:.1f}</td>
        <td>{ps['rmse']:.1f}</td><td>{ps['mard']:.1f}%</td><td>{ps['bias']:+.1f}</td>
        <td class="zone-A">{ps['zone_a']:.1f}%</td>
        <td class="{'good' if ps['zone_ab']>=95 else 'warn'}">{ps['zone_ab']:.1f}%</td>
    </tr>"""

    html += """
</table>

<h3>Batch x PPG Type Cross-Comparison</h3>
<table>
    <tr><th>Configuration</th><th>Samples</th><th>MAE</th><th>MARD</th><th>Bias</th><th>Zone A</th><th>A+B</th></tr>"""

    for cs in cross_stats:
        if cs['n_ok'] == 0:
            html += f"""
    <tr><td>{cs['label']}</td><td colspan="6" style="color:#666;">No data</td></tr>"""
        else:
            html += f"""
    <tr>
        <td>{cs['label']}</td><td>{cs['n_ok']}</td>
        <td class="{'good' if cs['mae']<20 else 'warn' if cs['mae']<35 else 'bad'}">{cs['mae']:.1f}</td>
        <td>{cs['mard']:.1f}%</td><td>{cs['bias']:+.1f}</td>
        <td class="zone-A">{cs['zone_a']:.1f}%</td>
        <td class="{'good' if cs['zone_ab']>=95 else 'warn'}">{cs['zone_ab']:.1f}%</td>
    </tr>"""

    # Determine which PPG type is better
    red_st = next((p for p in ppg_type_stats if p['ppg_type'] == 'Red'), None)
    ir_st = next((p for p in ppg_type_stats if p['ppg_type'] == 'IR'), None)
    if red_st and ir_st and red_st['n_ok'] > 0 and ir_st['n_ok'] > 0:
        better = 'Red' if red_st['mae'] < ir_st['mae'] else 'IR'
        diff = abs(red_st['mae'] - ir_st['mae'])
        html += f"""
</table>
<div class="insight-box">
    <h3>Red vs IR Analysis</h3>
    <ul>
        <li><b>{better} PPG performs better</b> by {diff:.1f} mg/dL MAE</li>
        <li>Red PPG: MAE={red_st['mae']:.1f}, MARD={red_st['mard']:.1f}%, Zone A={red_st['zone_a']:.1f}%</li>
        <li>IR PPG: MAE={ir_st['mae']:.1f}, MARD={ir_st['mard']:.1f}%, Zone A={ir_st['zone_a']:.1f}%</li>
        <li>Bias difference: Red={red_st['bias']:+.1f} vs IR={ir_st['bias']:+.1f} mg/dL</li>
    </ul>
</div>"""
    else:
        html += "</table>"

    html += f"""

<!-- ===== PER-PERSON ===== -->
<h2>6. Per-Person Analysis</h2>
<div class="chart-container">{bar_svg}</div>
<p style="text-align:center; color:#95a5a6; font-size:0.85em;">MAE per person (number of samples in parentheses)</p>

<table>
    <tr><th>Person</th><th>Samples</th><th>Ref Range</th><th>MAE</th><th>MARD</th><th>Bias</th><th>Pred Mean</th><th>Zone A</th><th>A+B</th></tr>"""

    for ps in person_stats:
        html += f"""
    <tr>
        <td class="left">{ps['name']}</td><td>{ps['n']}</td>
        <td>{ps['ref_range']}</td>
        <td class="{'good' if ps['mae']<20 else 'warn' if ps['mae']<35 else 'bad'}">{ps['mae']:.1f}</td>
        <td>{ps['mard']:.1f}%</td><td>{ps['bias']:+.1f}</td><td>{ps['pred_mean']:.1f}</td>
        <td class="zone-A">{ps['zone_a']:.0f}%</td>
        <td class="{'good' if ps['zone_ab']>=95 else 'warn' if ps['zone_ab']>=80 else 'bad'}">{ps['zone_ab']:.0f}%</td>
    </tr>"""

    html += f"""
    <tr class="highlight-row">
        <td>Overall</td><td>{n_ok}</td><td>{refs.min():.0f}-{refs.max():.0f}</td>
        <td>{mae:.1f}</td><td>{mard:.1f}%</td><td>{mean_bias:+.1f}</td><td>{preds.mean():.1f}</td>
        <td class="zone-A">{pcts['A']:.0f}%</td><td class="{ab_class}">{ab_pct:.0f}%</td>
    </tr>
</table>

<!-- ===== DETAILED SAMPLE TABLE ===== -->
<h2>7. All Samples</h2>
<table>
    <tr><th>#</th><th>Person</th><th>Date</th><th>Time</th><th>Batch</th><th>PPG</th><th>Ref</th><th>Pred</th><th>Error</th><th>MARD</th><th>Win</th><th>Zone</th></tr>"""

    for i, (_, row) in enumerate(ok.sort_values(['name', 'ppg_type', 'date', 'time']).iterrows(), 1):
        err = row['pred_error']
        ref = row['ref_glucose']
        pred = row['pred_mean']
        z = clarke_zone(ref, pred)
        zcss = f'zone-{z}'
        pt = row.get('ppg_type', '?')
        html += f"""
    <tr>
        <td>{i}</td><td class="left">{row['name']}</td><td>{row['date']}</td><td>{row['time']}</td>
        <td>{row['batch']}</td><td>{pt}</td><td>{ref:.0f}</td><td>{pred:.1f}</td>
        <td class="{'good' if abs(err)<20 else 'warn' if abs(err)<40 else 'bad'}">{err:+.1f}</td>
        <td>{row['pred_mard']:.1f}%</td><td>{int(row.get('num_accepted',0))}</td>
        <td class="{zcss}">{z}</td>
    </tr>"""

    html += f"""
</table>

<!-- ===== FAILED SAMPLES ===== -->
<h2>8. Failed Samples ({n_fail})</h2>
<table>
    <tr><th>Person</th><th>Date</th><th>Time</th><th>Batch</th><th>Reason</th></tr>
    {fail_rows}
</table>

<!-- ===== CLINICAL ASSESSMENT ===== -->
<h2>9. Clinical Assessment</h2>

<div class="insight-box">
    <h3>Performance Summary</h3>
    <ul>
        <li><b>Zone A (Clinically Accurate):</b> <span class="{'good' if pcts['A']>=50 else 'warn'}">{pcts['A']:.1f}%</span></li>
        <li><b>Clinically Acceptable (A+B):</b> <span class="{ab_class}">{ab_pct:.1f}%</span></li>
        <li><b>ISO 15197:</b> {'Meets' if ab_pct >= 95 else 'Does NOT meet'} the 95% A+B requirement</li>
        <li><b>Within +/-20 mg/dL:</b> {within_20:.1f}% | <b>Within +/-30 mg/dL:</b> {within_30:.1f}%</li>
        <li><b>Dangerous Zones (D+E):</b> <span class="{'good' if pcts['D']+pcts['E']<1 else 'bad'}">{pcts['D']+pcts['E']:.1f}%</span></li>
    </ul>
</div>

<div class="insight-box">
    <h3>Key Observations</h3>
    <ul>"""

    if mae < 20:
        html += f'<li class="good">MAE of {mae:.1f} mg/dL is within acceptable clinical range</li>'
    elif mae < 35:
        html += f'<li class="warn">MAE of {mae:.1f} mg/dL is moderate; further improvement needed</li>'
    else:
        html += f'<li class="bad">MAE of {mae:.1f} mg/dL is high for clinical glucose monitoring</li>'

    if abs(mean_bias) > 10:
        html += f'<li class="warn">Systematic bias of {mean_bias:+.1f} mg/dL detected - model tends to {"overpredict" if mean_bias > 0 else "underpredict"}</li>'

    best_p = person_stats[0]
    worst_p = person_stats[-1]
    html += f"""
        <li><b>Best person:</b> {best_p['name']} (MAE={best_p['mae']:.1f}, n={best_p['n']})</li>
        <li><b>Worst person:</b> {worst_p['name']} (MAE={worst_p['mae']:.1f}, n={worst_p['n']})</li>
        <li><b>Batch consistency:</b> Batch 1 MAE={batch_stats[0]['mae']:.1f} vs Batch 2 MAE={batch_stats[1]['mae']:.1f} mg/dL</li>
        <li><b>Failure rate:</b> {n_fail}/{n_total} ({n_fail/n_total*100:.0f}%) - mostly due to single-peak check rejection</li>
        <li>Pearson correlation: {np.corrcoef(refs, preds)[0,1]:.3f}</li>
    </ul>
</div>

<div class="insight-box">
    <h3>Recommendations</h3>
    <ul>"""

    if ab_pct < 95:
        html += f'<li>Model needs improvement to reach 95% A+B threshold (currently {ab_pct:.1f}%)</li>'
    if n_fail > 5:
        html += f'<li>Consider relaxing single-peak check (--skip_single_peak_check) to reduce {n_fail} failures</li>'
    if abs(mean_bias) > 10:
        html += f'<li>Apply bias correction ({mean_bias:+.1f} mg/dL offset) as a post-processing step</li>'
    if r2 < 0.3:
        html += '<li>Low R2 indicates model needs architectural or training improvements for this data source</li>'

    html += """
        <li>This vanilla PPG data differs from VitalDB training data; domain adaptation or fine-tuning on lab PPG data may improve results</li>
        <li>Consider augmenting training set with vanilla PPG samples to close the domain gap</li>
    </ul>
</div>

<p style="text-align:center; color:#555; margin-top:50px; font-size:0.82em;">
    Generated by analyze_vanilla_ppg_batches.py | Batch 1 + Batch 2 Combined
</p>
</body></html>"""

    report_path = OUTPUT_DIR / "VANILLA_PPG_INFERENCE_REPORT.html"
    report_path.write_text(html, encoding='utf-8')
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
