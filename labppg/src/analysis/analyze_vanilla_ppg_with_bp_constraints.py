"""
Vanilla PPG Batch Inference Report with Blood Pressure Constraints
===================================================================
Generates report for Batch 1 results with systolic >= 130 and diastolic >= 90
"""

import numpy as np
import pandas as pd
from pathlib import Path


BASE = Path(r"C:\IITM\vitalDB\labppg\data")

# Blood pressure constraints
SYSTOLIC_MAX = 130
DIASTOLIC_MAX = 90

SUMMARIES = [
    (BASE / "VanillaPPG_Batch_1_500Hz" / "results" / "batch_summary.csv", "Batch 1", "Red"),
    (BASE / "VanillaPPG_Batch_1_500Hz" / "results" / "iR_PPG" / "batch_summary.csv", "Batch 1", "IR"),
]

OUTPUT_DIR = BASE.parent / "reports"


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
        x, y = sx(r), sy(p)
        jx = x + np.random.normal(0, 2)
        jy = y + np.random.normal(0, 2)
        s.append(f'<circle cx="{jx}" cy="{jy}" r="3" fill="{color}" opacity="0.7" stroke="rgba(255,255,255,0.3)" stroke-width="0.5"/>')
    s.append('</g>')

    # Legend
    s.append('<g>')
    y_pos = margin + 20
    for i, person in enumerate(unique_persons):
        color = pcolor[person]
        s.append(f'<circle cx="{width-margin-150}" cy="{y_pos + i*22}" r="3" fill="{color}"/>')
        s.append(f'<text x="{width-margin-135}" y="{y_pos+4 + i*22}" font-size="11" fill="#fff" font-family="Segoe UI">{person}</text>')
    s.append('</g>')

    # Axis labels
    s.append(f'<text x="{width//2}" y="{height-20}" text-anchor="middle" font-size="12" fill="#aaa" font-family="Segoe UI">Reference (mg/dL)</text>')
    s.append(f'<text x="20" y="{height//2}" text-anchor="middle" font-size="12" fill="#aaa" font-family="Segoe UI" transform="rotate(-90 20 {height//2})">Predicted (mg/dL)</text>')

    # Axis ticks and values
    for v in range(50, mx + 1, 50):
        s.append(f'<text x="{sx(v)}" y="{height-margin+18}" text-anchor="middle" font-size="10" fill="#888" font-family="Segoe UI">{v}</text>')
        s.append(f'<text x="{margin-20}" y="{sy(v)+4}" text-anchor="end" font-size="10" fill="#888" font-family="Segoe UI">{v}</text>')

    s.append('</svg>')
    return '\n'.join(s)


def main():
    dfs = []
    for path, batch, ppg_type in SUMMARIES:
        if path.exists():
            df = pd.read_csv(path)
            df['batch'] = batch
            df['ppg_type'] = ppg_type
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Apply blood pressure constraints
    df_bp_filtered = df[
        (df['systolic'] <= SYSTOLIC_MAX) & 
        (df['diastolic'] <= DIASTOLIC_MAX)
    ].copy()

    ok = df_bp_filtered[df_bp_filtered['status'] == 'OK'].copy()
    failed = df_bp_filtered[df_bp_filtered['status'] != 'OK'].copy()

    # Filter out 'Unknown' person
    ok = ok[ok['name'] != 'Unknown']

    if len(ok) == 0:
        print("No data available after applying blood pressure constraints!")
        return

    refs = ok['ref_glucose'].values
    preds = ok['pred_mean'].values
    errors = preds - refs
    abs_errors = np.abs(errors)
    persons = ok['name'].values

    n_total = len(df_bp_filtered)
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

    # Clarke SVG
    clarke_svg_str = clarke_svg(refs, preds, persons)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Vanilla PPG Inference Report (BP Constrained)</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            color: white;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.2em;
            font-weight: 300;
            letter-spacing: 1px;
        }}
        .header .subtitle {{
            font-size: 0.9em;
            opacity: 0.95;
            margin-top: 8px;
        }}
        h2 {{
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
            font-size: 1.5em;
            font-weight: 400;
        }}
        h3 {{
            color: #aaa;
            font-weight: 400;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #1e2139;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #999;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: 300;
            color: #fff;
        }}
        .good {{ color: #2ecc71; }}
        .warn {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #141829;
            margin: 20px 0;
            border-radius: 6px;
            overflow: hidden;
        }}
        th {{
            background: #1e2139;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #aaa;
            border-bottom: 2px solid #667eea;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #1e2139;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background: #1a1f35;
        }}
        .left {{ text-align: left; }}
        .good {{ background: rgba(46, 204, 113, 0.1); }}
        .warn {{ background: rgba(243, 156, 18, 0.1); }}
        .bad {{ background: rgba(231, 76, 60, 0.1); }}
        .insight-box {{
            background: #1e2139;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 6px;
        }}
        .insight-box h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .insight-box ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .insight-box li {{
            margin: 8px 0;
        }}
        .constraint-notice {{
            background: #2c3e50;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
            font-size: 0.95em;
        }}
        svg {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>Vanilla PPG Inference Report</h1>
    <div class="subtitle">Blood Pressure Constrained Analysis (Systolic ≤ {SYSTOLIC_MAX}, Diastolic ≤ {DIASTOLIC_MAX})</div>
</div>

<div class="constraint-notice">
    <strong>Applied Constraints:</strong> This report includes only samples with Systolic pressure ≤ {SYSTOLIC_MAX} mmHg AND Diastolic pressure ≤ {DIASTOLIC_MAX} mmHg.
    Original dataset: {len(df)} samples → Filtered dataset: {n_total} samples
</div>

<!-- ===== OVERVIEW ===== -->
<h2>1. Overview</h2>
<div class="stats-grid">
    <div class="stat-box">
        <div class="stat-label">Total Samples</div>
        <div class="stat-value">{n_total}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Successful</div>
        <div class="stat-value">{n_ok}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Failed</div>
        <div class="stat-value">{n_fail}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Subjects</div>
        <div class="stat-value">{n_persons}</div>
    </div>
</div>

<!-- ===== PRIMARY METRICS ===== -->
<h2>2. Primary Performance Metrics</h2>
<div class="stats-grid">
    <div class="stat-box">
        <div class="stat-label">MAE (mg/dL)</div>
        <div class="stat-value">{mae:.2f}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">RMSE (mg/dL)</div>
        <div class="stat-value">{rmse:.2f}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">MARD (%)</div>
        <div class="stat-value">{mard:.2f}%</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Bias (mg/dL)</div>
        <div class="stat-value {('good' if abs(mean_bias) <= 10 else 'warn' if abs(mean_bias) <= 25 else 'bad')}">{mean_bias:+.2f}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">R²</div>
        <div class="stat-value">{r2:.3f}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Median AE</div>
        <div class="stat-value">{median_ae:.2f}</div>
    </div>
</div>

<!-- ===== CLARKE ERROR GRID ===== -->
<h2>3. Clarke Error Grid Analysis</h2>
{clarke_svg_str}
<div class="stats-grid">
    <div class="stat-box">
        <div class="stat-label">Zone A</div>
        <div class="stat-value">{pcts['A']:.1f}%</div>
        <div style="font-size: 0.85em; color: #999; margin-top: 5px;">{zones['A']} / {sum(zones.values())}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Zone B</div>
        <div class="stat-value">{pcts['B']:.1f}%</div>
        <div style="font-size: 0.85em; color: #999; margin-top: 5px;">{zones['B']} / {sum(zones.values())}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Zone C</div>
        <div class="stat-value {('warn' if pcts['C'] > 0 else 'good')}">{pcts['C']:.1f}%</div>
        <div style="font-size: 0.85em; color: #999; margin-top: 5px;">{zones['C']} / {sum(zones.values())}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Zone D</div>
        <div class="stat-value {('bad' if pcts['D'] > 0 else 'good')}">{pcts['D']:.1f}%</div>
        <div style="font-size: 0.85em; color: #999; margin-top: 5px;">{zones['D']} / {sum(zones.values())}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Zone E</div>
        <div class="stat-value {('bad' if pcts['E'] > 0 else 'good')}">{pcts['E']:.1f}%</div>
        <div style="font-size: 0.85em; color: #999; margin-top: 5px;">{zones['E']} / {sum(zones.values())}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">A+B (Acceptable)</div>
        <div class="stat-value {('good' if ab_pct >= 95 else 'warn' if ab_pct >= 80 else 'bad')}">{ab_pct:.1f}%</div>
        <div style="font-size: 0.85em; color: #999; margin-top: 5px;">Clinically Acceptable</div>
    </div>
</div>

<!-- ===== ACCURACY THRESHOLDS ===== -->
<h2>4. Accuracy Thresholds</h2>
<div class="stats-grid">
    <div class="stat-box">
        <div class="stat-label">Within ±15 mg/dL</div>
        <div class="stat-value">{within_15:.1f}%</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Within ±20 mg/dL</div>
        <div class="stat-value">{within_20:.1f}%</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Within ±30 mg/dL</div>
        <div class="stat-value">{within_30:.1f}%</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Within ±20% or ±20 mg/dL</div>
        <div class="stat-value">{within_20pct:.1f}%</div>
    </div>
</div>

<!-- ===== PER-PERSON BREAKDOWN ===== -->
<h2>5. Per-Person Breakdown</h2>
<table>
    <tr>
        <th>Rank</th>
        <th class="left">Person</th>
        <th>Samples</th>
        <th>Ref Range (mg/dL)</th>
        <th>MAE (mg/dL)</th>
        <th>MARD (%)</th>
        <th>Bias (mg/dL)</th>
        <th>Pred Mean</th>
        <th>Zone A (%)</th>
        <th>Zone A+B (%)</th>
    </tr>
"""

    for i, row in enumerate(person_stats, 1):
        html += f"""    <tr>
        <td>{i}</td>
        <td class="left">{row['name']}</td>
        <td>{row['n']}</td>
        <td>{row['ref_range']}</td>
        <td class="{('good' if row['mae'] < 20 else 'warn' if row['mae'] < 35 else 'bad')}">{row['mae']:.2f}</td>
        <td>{row['mard']:.2f}%</td>
        <td>{row['bias']:+.2f}</td>
        <td>{row['pred_mean']:.1f}</td>
        <td>{row['zone_a']:.1f}%</td>
        <td>{row['zone_ab']:.1f}%</td>
    </tr>
"""

    html += """</table>

<!-- ===== DETAILED RESULTS ===== -->
<h2>6. Detailed Results</h2>
<table>
    <tr>
        <th>#</th>
        <th class="left">Person</th>
        <th>Date</th>
        <th>Time</th>
        <th>Sys/Dia</th>
        <th>Ref (mg/dL)</th>
        <th>Pred (mg/dL)</th>
        <th>Error (mg/dL)</th>
        <th>MARD (%)</th>
        <th>Windows</th>
        <th>Zone</th>
    </tr>
"""

    for i, (_, row) in enumerate(ok.iterrows(), 1):
        ref = row.get('ref_glucose', np.nan)
        pred = row.get('pred_mean', np.nan)
        err = pred - ref if not (np.isnan(ref) or np.isnan(pred)) else np.nan
        z = clarke_zone(ref, pred) if not (np.isnan(ref) or np.isnan(pred)) else '?'
        zcss = {'A': 'good', 'B': 'good', 'C': 'warn', 'D': 'bad', 'E': 'bad'}.get(z, '')
        html += f"""    <tr>
        <td>{i}</td>
        <td class="left">{row['name']}</td>
        <td>{row.get('date', '?')}</td>
        <td>{row.get('time', '?')}</td>
        <td>{row.get('systolic', '?'):.0f}/{row.get('diastolic', '?'):.0f}</td>
        <td>{ref:.0f}</td>
        <td>{pred:.1f}</td>
        <td class="{'good' if abs(err)<20 else 'warn' if abs(err)<40 else 'bad'}">{err:+.1f}</td>
        <td>{row['pred_mard']:.1f}%</td>
        <td>{int(row.get('num_accepted',0))}</td>
        <td class="{zcss}">{z}</td>
    </tr>
"""

    html += f"""</table>

<!-- ===== FAILED SAMPLES ===== -->
<h2>7. Failed Samples ({n_fail})</h2>
<table>
    <tr><th>Person</th><th>Date</th><th>Time</th><th>Sys/Dia</th><th>Reason</th></tr>
"""

    for _, row in failed.iterrows():
        html += f"""    <tr>
        <td>{row['name']}</td>
        <td>{row.get('date', '?')}</td>
        <td>{row.get('time', '?')}</td>
        <td>{row.get('systolic', '?'):.0f}/{row.get('diastolic', '?'):.0f}</td>
        <td>{row.get('error', 'Unknown')}</td>
    </tr>
"""

    html += """</table>

<!-- ===== CLINICAL ASSESSMENT ===== -->
<h2>8. Clinical Assessment</h2>

<div class="insight-box">
    <h3>Performance Summary</h3>
    <ul>
        <li><b>Zone A (Clinically Accurate):</b> <span class='""" + ('good' if pcts['A']>=50 else 'warn') + f"""'>{pcts['A']:.1f}%</span></li>
        <li><b>Clinically Acceptable (A+B):</b> <span class="{'good' if ab_pct >= 95 else 'warn' if ab_pct >= 80 else 'bad'}">{ab_pct:.1f}%</span></li>
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

    if len(person_stats) > 0:
        best_p = person_stats[0]
        worst_p = person_stats[-1]
        html += f"""
        <li><b>Best person:</b> {best_p['name']} (MAE={best_p['mae']:.1f}, n={best_p['n']})</li>
        <li><b>Worst person:</b> {worst_p['name']} (MAE={worst_p['mae']:.1f}, n={worst_p['n']})</li>"""

    html += f"""
        <li><b>Failure rate:</b> {n_fail}/{n_total} ({n_fail/n_total*100:.0f}%)</li>
        <li>Pearson correlation: {np.corrcoef(refs, preds)[0,1]:.3f}</li>
        <li><b>Note:</b> This analysis is constrained to normal blood pressure readings (Sys≤{SYSTOLIC_MAX}, Dia≤{DIASTOLIC_MAX})</li>
    </ul>
</div>

<div class="insight-box">
    <h3>Recommendations</h3>
    <ul>"""

    if ab_pct < 95:
        html += f'<li>Model needs improvement to reach 95% A+B threshold (currently {ab_pct:.1f}%)</li>'
    if abs(mean_bias) > 10:
        html += f'<li>Apply bias correction ({mean_bias:+.1f} mg/dL offset) as a post-processing step</li>'
    if r2 < 0.3:
        html += '<li>Low R2 indicates model needs architectural or training improvements for this data source</li>'

    html += """
        <li>This vanilla PPG data differs from VitalDB training data; domain adaptation or fine-tuning on lab PPG data may improve results</li>
        <li>Consider the impact of elevated blood pressure on PPG signal characteristics</li>
    </ul>
</div>

<p style="text-align:center; color:#555; margin-top:50px; font-size:0.82em;">
    Generated by analyze_vanilla_ppg_with_bp_constraints.py | Batch 1 (BP Constrained: Sys≤{SYSTOLIC_MAX}, Dia≤{DIASTOLIC_MAX})
</p>
</body></html>"""

    report_path = OUTPUT_DIR / "VANILLA_PPG_INFERENCE_REPORT_BP_CONSTRAINED.html"
    report_path.write_text(html, encoding='utf-8')
    print(f"\nReport saved: {report_path}")
    print(f"Constraints: Systolic <= {SYSTOLIC_MAX}, Diastolic <= {DIASTOLIC_MAX}")
    print(f"Original samples: {len(df)}, Filtered samples: {n_total}")
    print(f"Successful: {n_ok}, Failed: {n_fail}")


if __name__ == '__main__':
    main()
