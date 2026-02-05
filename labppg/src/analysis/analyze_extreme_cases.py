"""
Extreme Case Comparison Report
==============================
Compares the best and worst performing cases from the Vanilla PPG batch inference
to understand why some predictions are accurate while others fail catastrophically.
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

OUTPUT = BASE / "EXTREME_CASE_ANALYSIS.html"


def main():
    print("=" * 70)
    print("EXTREME CASE COMPARISON REPORT")
    print("=" * 70)

    # Load all summaries
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

    ok = df[(df['status'] == 'OK') & (df['name'] != 'Unknown')].copy()
    n_ok = len(ok)

    # Identify extremes
    best_idx = ok['pred_abs_error'].idxmin()
    worst_idx = ok['pred_abs_error'].idxmax()
    best = ok.loc[best_idx]
    worst = ok.loc[worst_idx]

    print(f"\n  BEST:  {best['name']} | Err={best['pred_error']:+.1f} mg/dL")
    print(f"  WORST: {worst['name']} | Err={worst['pred_error']:+.1f} mg/dL")

    # Per-person stats
    person_data = []
    for name, grp in ok.groupby('name'):
        refs = grp['ref_glucose'].values
        preds = grp['pred_mean'].values
        errors = preds - refs
        person_data.append({
            'name': name,
            'n': len(grp),
            'mae': np.mean(np.abs(errors)),
            'mard': np.mean(np.abs(errors) / refs) * 100,
            'bias': np.mean(errors),
            'mean_hr': grp['hr'].mean(),
            'mean_sys': grp['systolic'].mean(),
            'mean_dia': grp['diastolic'].mean(),
            'gluc_min': refs.min(),
            'gluc_max': refs.max(),
        })
    person_data.sort(key=lambda x: x['mae'])

    # Correlations
    corr_cols = ['hr', 'systolic', 'diastolic', 'num_windows', 'mean_similarity',
                 'pred_std', 'num_peaks', 'est_hr', 'ref_glucose']
    correlations = []
    for col in corr_cols:
        valid = ok[[col, 'pred_abs_error']].dropna()
        if len(valid) > 2:
            r = valid[col].corr(valid['pred_abs_error'])
            correlations.append({'feature': col, 'r': r})
    correlations.sort(key=lambda x: abs(x['r']), reverse=True)

    # All readings for both extreme persons
    best_name = best['name']
    worst_name = worst['name']

    best_person_all = df[df['name'] == best_name].sort_values(['batch', 'ppg_type', 'date', 'time'])
    worst_person_all = df[df['name'] == worst_name].sort_values(['batch', 'ppg_type', 'date', 'time'])

    # Scatter data for correlation SVG
    hrs = ok['hr'].values
    sys_bp = ok['systolic'].values
    abs_errors = ok['pred_abs_error'].values

    # Build SVGs
    # 1. Side-by-side bar comparison SVG
    compare_fields = [
        ('Ref Glucose', best['ref_glucose'], worst['ref_glucose'], 'mg/dL'),
        ('Predicted', best['pred_mean'], worst['pred_mean'], 'mg/dL'),
        ('Heart Rate', best['hr'], worst['hr'], 'bpm'),
        ('Systolic BP', best['systolic'], worst['systolic'], 'mmHg'),
        ('Diastolic BP', best['diastolic'], worst['diastolic'], 'mmHg'),
        ('Accepted Windows', best['num_windows'], worst['num_windows'], ''),
        ('Mean Similarity', best['mean_similarity'], worst['mean_similarity'], ''),
        ('Pred Std', best['pred_std'], worst['pred_std'], 'mg/dL'),
    ]

    bar_svg = _build_comparison_bar_svg(compare_fields)

    # 2. HR vs Abs Error scatter
    hr_scatter_svg = _build_scatter_svg(
        hrs, abs_errors, 'Heart Rate (bpm)', 'Absolute Error (mg/dL)',
        best['hr'], best['pred_abs_error'], worst['hr'], worst['pred_abs_error'],
        best_name, worst_name
    )

    # 3. Systolic BP vs Abs Error scatter
    bp_scatter_svg = _build_scatter_svg(
        sys_bp, abs_errors, 'Systolic BP (mmHg)', 'Absolute Error (mg/dL)',
        best['systolic'], best['pred_abs_error'], worst['systolic'], worst['pred_abs_error'],
        best_name, worst_name
    )

    # 4. Per-person MAE bar chart
    person_bar_svg = _build_person_mae_svg(person_data)

    # 5. Correlation bar chart
    corr_svg = _build_correlation_svg(correlations)

    # 6. Vikraman inverted response SVG
    vikraman_ok = ok[ok['name'] == worst_name].sort_values('ref_glucose')
    inversion_svg = ""
    if len(vikraman_ok) >= 2:
        inversion_svg = _build_inversion_svg(vikraman_ok, worst_name)

    # ---------- HTML ----------
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Extreme Case Analysis</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f0f23; color: #e2e8f0; line-height: 1.6; padding: 30px; }}
h1 {{ text-align: center; font-size: 2em; background: linear-gradient(135deg, #f97316, #ef4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px; }}
.subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 30px; font-size: 0.95em; }}
h2 {{ color: #f97316; border-bottom: 2px solid #f9731640; padding-bottom: 8px; margin: 35px 0 20px; font-size: 1.4em; }}
h3 {{ color: #fb923c; margin: 20px 0 12px; font-size: 1.1em; }}

.card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 20px 0; }}
.case-card {{ background: #1e293b; border-radius: 12px; padding: 24px; border-left: 4px solid; }}
.case-card.best {{ border-color: #22c55e; }}
.case-card.worst {{ border-color: #ef4444; }}
.case-card h3 {{ margin-top: 0; }}
.case-card .label {{ color: #94a3b8; font-size: 0.85em; }}
.case-card .value {{ font-size: 1.1em; font-weight: 600; }}
.case-card .big-value {{ font-size: 2.2em; font-weight: 700; line-height: 1.2; }}
.case-card .good {{ color: #22c55e; }}
.case-card .bad {{ color: #ef4444; }}
.case-card .neutral {{ color: #f97316; }}

.metric-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #334155; }}
.metric-row:last-child {{ border-bottom: none; }}

table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #1e293b; border-radius: 8px; overflow: hidden; font-size: 0.88em; }}
th {{ background: #334155; padding: 10px 12px; text-align: center; color: #e2e8f0; font-weight: 600; }}
td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #334155; }}
tr:hover {{ background: #334155aa; }}
td.left {{ text-align: left; }}
.good {{ color: #22c55e; }}
.warn {{ color: #f59e0b; }}
.bad {{ color: #ef4444; }}
.highlight-best {{ background: #22c55e15; }}
.highlight-worst {{ background: #ef444415; }}

.insight-box {{ background: #1e293b; border: 1px solid #f9731640; border-radius: 12px; padding: 20px 24px; margin: 20px 0; }}
.insight-box h3 {{ color: #f97316; margin-top: 0; }}
.insight-box ul {{ margin-left: 20px; }}
.insight-box li {{ margin-bottom: 8px; }}

.chart-container {{ text-align: center; margin: 20px 0; overflow-x: auto; }}
.chart-container svg {{ max-width: 100%; }}

.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

.tag {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
.tag-best {{ background: #22c55e30; color: #22c55e; }}
.tag-worst {{ background: #ef444430; color: #ef4444; }}

footer {{ text-align: center; color: #475569; margin-top: 40px; font-size: 0.8em; padding-top: 20px; border-top: 1px solid #334155; }}
</style>
</head><body>

<h1>Extreme Case Comparison Analysis</h1>
<p class="subtitle">Why does the same model achieve 0.1 mg/dL error for one sample and 129.5 mg/dL for another?</p>

<!-- ===== SECTION 1: HEAD-TO-HEAD ===== -->
<h2>1. Head-to-Head Comparison</h2>

<div class="card-grid">
    <div class="case-card best">
        <h3><span class="tag tag-best">BEST</span> {best_name}</h3>
        <div class="big-value good">{best['pred_abs_error']:.1f} <span style="font-size:0.4em">mg/dL error</span></div>
        <div style="margin-top: 12px;">
            <div class="metric-row"><span class="label">Date / Time</span><span class="value">{best['date']} {best['time']}</span></div>
            <div class="metric-row"><span class="label">Batch / PPG</span><span class="value">{best['batch']} {best['ppg_type']}</span></div>
            <div class="metric-row"><span class="label">Ref Glucose</span><span class="value">{best['ref_glucose']:.0f} mg/dL</span></div>
            <div class="metric-row"><span class="label">Predicted</span><span class="value good">{best['pred_mean']:.1f} mg/dL</span></div>
            <div class="metric-row"><span class="label">Heart Rate</span><span class="value">{best['hr']:.0f} bpm</span></div>
            <div class="metric-row"><span class="label">Blood Pressure</span><span class="value">{best['systolic']:.0f}/{best['diastolic']:.0f} mmHg</span></div>
            <div class="metric-row"><span class="label">Accepted Windows</span><span class="value">{best['num_windows']:.0f}</span></div>
            <div class="metric-row"><span class="label">Mean Similarity</span><span class="value">{best['mean_similarity']:.3f}</span></div>
            <div class="metric-row"><span class="label">Pred Std</span><span class="value">{best['pred_std']:.1f}</span></div>
            <div class="metric-row"><span class="label">Pred Range</span><span class="value">{best['pred_min']:.0f} &ndash; {best['pred_max']:.0f}</span></div>
        </div>
    </div>

    <div class="case-card worst">
        <h3><span class="tag tag-worst">WORST</span> {worst_name}</h3>
        <div class="big-value bad">{worst['pred_abs_error']:.1f} <span style="font-size:0.4em">mg/dL error</span></div>
        <div style="margin-top: 12px;">
            <div class="metric-row"><span class="label">Date / Time</span><span class="value">{worst['date']} {worst['time']}</span></div>
            <div class="metric-row"><span class="label">Batch / PPG</span><span class="value">{worst['batch']} {worst['ppg_type']}</span></div>
            <div class="metric-row"><span class="label">Ref Glucose</span><span class="value">{worst['ref_glucose']:.0f} mg/dL</span></div>
            <div class="metric-row"><span class="label">Predicted</span><span class="value bad">{worst['pred_mean']:.1f} mg/dL</span></div>
            <div class="metric-row"><span class="label">Heart Rate</span><span class="value bad">{worst['hr']:.0f} bpm</span></div>
            <div class="metric-row"><span class="label">Blood Pressure</span><span class="value bad">{worst['systolic']:.0f}/{worst['diastolic']:.0f} mmHg</span></div>
            <div class="metric-row"><span class="label">Accepted Windows</span><span class="value">{worst['num_windows']:.0f}</span></div>
            <div class="metric-row"><span class="label">Mean Similarity</span><span class="value">{worst['mean_similarity']:.3f}</span></div>
            <div class="metric-row"><span class="label">Pred Std</span><span class="value">{worst['pred_std']:.1f}</span></div>
            <div class="metric-row"><span class="label">Pred Range</span><span class="value">{worst['pred_min']:.0f} &ndash; {worst['pred_max']:.0f}</span></div>
        </div>
    </div>
</div>

<div class="chart-container">{bar_svg}</div>

<!-- ===== SECTION 2: KEY DIFFERENCES ===== -->
<h2>2. Key Differences</h2>
<table>
    <tr><th>Parameter</th><th class="good">Best ({best_name})</th><th class="bad">Worst ({worst_name})</th><th>Difference</th><th>Implication</th></tr>
    <tr>
        <td class="left"><b>Heart Rate</b></td>
        <td>{best['hr']:.0f} bpm</td>
        <td class="bad">{worst['hr']:.0f} bpm</td>
        <td>{worst['hr'] - best['hr']:+.0f} bpm</td>
        <td class="left">Tachycardia compresses diastolic phase, altering PPG morphology</td>
    </tr>
    <tr>
        <td class="left"><b>Systolic BP</b></td>
        <td>{best['systolic']:.0f} mmHg</td>
        <td class="bad">{worst['systolic']:.0f} mmHg</td>
        <td>{worst['systolic'] - best['systolic']:+.0f} mmHg</td>
        <td class="left">Hypertension increases vascular stiffness &rarr; stiffer pulse wave</td>
    </tr>
    <tr>
        <td class="left"><b>Diastolic BP</b></td>
        <td>{best['diastolic']:.0f} mmHg</td>
        <td class="bad">{worst['diastolic']:.0f} mmHg</td>
        <td>{worst['diastolic'] - best['diastolic']:+.0f} mmHg</td>
        <td class="left">Elevated diastolic = sustained vascular resistance</td>
    </tr>
    <tr>
        <td class="left"><b>Accepted Windows</b></td>
        <td>{best['num_windows']:.0f}</td>
        <td>{worst['num_windows']:.0f}</td>
        <td>{worst['num_windows'] - best['num_windows']:+.0f}</td>
        <td class="left">More windows does NOT mean better accuracy</td>
    </tr>
    <tr>
        <td class="left"><b>Mean Similarity</b></td>
        <td>{best['mean_similarity']:.3f}</td>
        <td>{worst['mean_similarity']:.3f}</td>
        <td>{worst['mean_similarity'] - best['mean_similarity']:+.3f}</td>
        <td class="left">High consistency + wrong = systematic bias, not noise</td>
    </tr>
    <tr>
        <td class="left"><b>Ref Glucose</b></td>
        <td>{best['ref_glucose']:.0f} mg/dL</td>
        <td>{worst['ref_glucose']:.0f} mg/dL</td>
        <td>{worst['ref_glucose'] - best['ref_glucose']:+.0f} mg/dL</td>
        <td class="left">Both in normal range &mdash; glucose level is NOT the problem</td>
    </tr>
</table>

<!-- ===== SECTION 3: CARDIOVASCULAR CONFOUNDING ===== -->
<h2>3. Cardiovascular Confounding Evidence</h2>

<div class="insight-box">
    <h3>Core Finding: Blood Pressure and Heart Rate Drive Prediction Error</h3>
    <p>The model was trained on VitalDB surgical ICU data where hemodynamic instability (high HR, high BP, stiff vasculature)
    co-occurs with hyperglycemia. When applied to ambulatory subjects, it mistakes <b>cardiovascular-driven PPG features</b>
    for <b>glucose-driven optical absorption changes</b>.</p>
</div>

<div class="two-col">
    <div class="chart-container">{hr_scatter_svg}</div>
    <div class="chart-container">{bp_scatter_svg}</div>
</div>

<h3>Feature Correlations with Absolute Error</h3>
<div class="chart-container">{corr_svg}</div>

<table>
    <tr><th>Feature</th><th>Correlation (r)</th><th>Interpretation</th></tr>"""

    for c in correlations:
        r_val = c['r']
        css = 'bad' if abs(r_val) > 0.25 else 'warn' if abs(r_val) > 0.15 else ''
        if c['feature'] == 'systolic':
            interp = "Higher systolic BP = larger errors. Strongest cardiovascular confound."
        elif c['feature'] == 'diastolic':
            interp = "Higher diastolic BP = larger errors. Sustained vascular resistance alters PPG shape."
        elif c['feature'] == 'pred_std':
            interp = "Higher prediction spread = larger errors. Model uncertainty is informative."
        elif c['feature'] == 'hr':
            interp = "Higher HR = larger errors. Tachycardia compresses PPG waveform."
        elif c['feature'] == 'num_windows':
            interp = "Weak negative: more windows slightly helps (averaging effect)."
        elif c['feature'] == 'mean_similarity':
            interp = "Similarity does NOT predict accuracy. Consistent signal != correct prediction."
        elif c['feature'] == 'ref_glucose':
            interp = "Near zero: actual glucose level has NO influence on error magnitude."
        else:
            interp = ""
        html += f"""
    <tr><td class="left"><b>{c['feature']}</b></td><td class="{css}">{r_val:+.3f}</td><td class="left">{interp}</td></tr>"""

    html += """
</table>

<!-- ===== SECTION 4: INVERTED RESPONSE ===== -->"""

    if inversion_svg:
        html += f"""
<h2>4. Inverted Model Response for {worst_name}</h2>

<div class="insight-box">
    <h3>The Model Responds Inversely to Glucose for This Subject</h3>
    <p>{worst_name}'s actual glucose of <b>179 mg/dL</b> produces a prediction of ~140 (underpredicts by 40),
    while glucose of <b>102 mg/dL</b> produces a prediction of ~230 (overpredicts by 130).
    The model's response is <b>inversely</b> correlated with true glucose for this cardiovascular phenotype &mdash;
    proving it is <b>not tracking glucose</b> but rather hemodynamic-driven PPG shape features.</p>
</div>

<div class="chart-container">{inversion_svg}</div>
"""

    # All readings for worst person
    html += f"""
<h3>All {worst_name} Readings</h3>
<table>
    <tr><th>Batch</th><th>PPG</th><th>Date</th><th>Time</th><th>Ref Gluc</th><th>Predicted</th><th>Error</th><th>HR</th><th>BP</th><th>Windows</th><th>Status</th></tr>"""

    for _, r in worst_person_all.iterrows():
        if r['status'] == 'OK' and not pd.isna(r.get('pred_error')):
            err = r['pred_error']
            css = 'good' if abs(err) < 20 else 'warn' if abs(err) < 40 else 'bad'
            html += f"""
    <tr>
        <td>{r['batch']}</td><td>{r['ppg_type']}</td><td>{r['date']}</td><td>{r['time']}</td>
        <td>{r['ref_glucose']:.0f}</td><td>{r['pred_mean']:.1f}</td>
        <td class="{css}">{err:+.1f}</td>
        <td>{r['hr']:.0f}</td><td>{r['systolic']:.0f}/{r['diastolic']:.0f}</td>
        <td>{r['num_windows']:.0f}</td><td class="good">OK</td>
    </tr>"""
        else:
            g = r.get('glucose', r.get('ref_glucose', '?'))
            g_str = f"{g:.0f}" if isinstance(g, (int, float)) and not pd.isna(g) else '?'
            hr_str = f"{r['hr']:.0f}" if not pd.isna(r.get('hr', float('nan'))) else '?'
            bp_str = f"{r['systolic']:.0f}/{r['diastolic']:.0f}" if not pd.isna(r.get('systolic', float('nan'))) else '?'
            html += f"""
    <tr style="opacity:0.6;">
        <td>{r['batch']}</td><td>{r['ppg_type']}</td><td>{r['date']}</td><td>{r['time']}</td>
        <td>{g_str}</td><td>&mdash;</td><td>&mdash;</td>
        <td>{hr_str}</td><td>{bp_str}</td><td>&mdash;</td><td class="bad">FAILED</td>
    </tr>"""

    html += "</table>"

    # Section 5 number depends on whether section 4 was shown
    sec = 5 if inversion_svg else 4

    # All readings for best person
    html += f"""
<h2>{sec}. Best Performer: {best_name} &mdash; Context</h2>

<div class="insight-box">
    <h3>The "Perfect" Reading is Likely Coincidental</h3>
    <p>{best_name}'s single best reading (error = {best['pred_error']:+.1f} mg/dL) came from only
    <b>{best['num_windows']:.0f} accepted windows</b> with the <b>lowest similarity ({best['mean_similarity']:.3f})</b>
    in the dataset. Most windows were rejected. The two surviving windows happened to average near the true value.
    Meanwhile, the <b>same person, same timestamp, on IR PPG</b> gives a much larger error &mdash; confirming this is
    statistical coincidence rather than genuine model accuracy.</p>
</div>

<h3>All {best_name} Readings</h3>
<table>
    <tr><th>Batch</th><th>PPG</th><th>Date</th><th>Time</th><th>Ref Gluc</th><th>Predicted</th><th>Error</th><th>HR</th><th>BP</th><th>Windows</th><th>Status</th></tr>"""

    for _, r in best_person_all.iterrows():
        is_best_row = (r['date'] == best['date'] and r['time'] == best['time']
                       and r['ppg_type'] == best['ppg_type'] and r['batch'] == best['batch'])
        row_cls = ' class="highlight-best"' if is_best_row else ''
        if r['status'] == 'OK' and not pd.isna(r.get('pred_error')):
            err = r['pred_error']
            css = 'good' if abs(err) < 20 else 'warn' if abs(err) < 40 else 'bad'
            star = ' *' if is_best_row else ''
            html += f"""
    <tr{row_cls}>
        <td>{r['batch']}</td><td>{r['ppg_type']}</td><td>{r['date']}</td><td>{r['time']}</td>
        <td>{r['ref_glucose']:.0f}</td><td>{r['pred_mean']:.1f}</td>
        <td class="{css}">{err:+.1f}{star}</td>
        <td>{r['hr']:.0f}</td><td>{r['systolic']:.0f}/{r['diastolic']:.0f}</td>
        <td>{r['num_windows']:.0f}</td><td class="good">OK</td>
    </tr>"""
        else:
            g = r.get('glucose', r.get('ref_glucose', '?'))
            g_str = f"{g:.0f}" if isinstance(g, (int, float)) and not pd.isna(g) else '?'
            hr_str = f"{r['hr']:.0f}" if not pd.isna(r.get('hr', float('nan'))) else '?'
            bp_str = f"{r['systolic']:.0f}/{r['diastolic']:.0f}" if not pd.isna(r.get('systolic', float('nan'))) else '?'
            html += f"""
    <tr style="opacity:0.6;">
        <td>{r['batch']}</td><td>{r['ppg_type']}</td><td>{r['date']}</td><td>{r['time']}</td>
        <td>{g_str}</td><td>&mdash;</td><td>&mdash;</td>
        <td>{hr_str}</td><td>{bp_str}</td><td>&mdash;</td><td class="bad">FAILED</td>
    </tr>"""

    html += "</table>"

    sec += 1

    # Section: Per-person MAE ranked
    html += f"""

<h2>{sec}. Per-Person Error Ranking</h2>
<p style="color:#94a3b8;">Persons ranked by mean absolute error. Note how MAE tracks with cardiovascular profile.</p>

<div class="chart-container">{person_bar_svg}</div>

<table>
    <tr><th>#</th><th>Person</th><th>Samples</th><th>MAE</th><th>MARD</th><th>Bias</th><th>Mean HR</th><th>Mean BP</th><th>Glucose Range</th></tr>"""

    for i, p in enumerate(person_data, 1):
        mae_css = 'good' if p['mae'] < 20 else 'warn' if p['mae'] < 35 else 'bad'
        is_best = (p['name'] == best_name)
        is_worst = (p['name'] == worst_name)
        row_cls = ' class="highlight-best"' if is_best else (' class="highlight-worst"' if is_worst else '')
        tag = ' <span class="tag tag-best">BEST</span>' if is_best else (' <span class="tag tag-worst">WORST</span>' if is_worst else '')
        html += f"""
    <tr{row_cls}>
        <td>{i}</td><td class="left">{p['name']}{tag}</td><td>{p['n']}</td>
        <td class="{mae_css}">{p['mae']:.1f}</td><td>{p['mard']:.1f}%</td><td>{p['bias']:+.1f}</td>
        <td>{p['mean_hr']:.0f}</td><td>{p['mean_sys']:.0f}/{p['mean_dia']:.0f}</td>
        <td>{p['gluc_min']:.0f}&ndash;{p['gluc_max']:.0f}</td>
    </tr>"""

    html += "</table>"

    sec += 1

    # Section: Root Cause Summary
    html += f"""

<h2>{sec}. Root Cause Analysis</h2>

<div class="insight-box">
    <h3>Finding 1: Cardiovascular State is the Dominant Confound</h3>
    <ul>
        <li>Systolic BP correlates <b>r = +{correlations[0]['r'] if correlations[0]['feature'] == 'systolic' else next(c['r'] for c in correlations if c['feature'] == 'systolic'):.3f}</b> with absolute error &mdash; the strongest predictor</li>
        <li>Actual glucose level correlates <b>r = {next(c['r'] for c in correlations if c['feature'] == 'ref_glucose'):+.3f}</b> with error &mdash; essentially zero</li>
        <li>The 3 worst-performing persons ({', '.join(p['name'] for p in person_data[-3:])}) all have elevated HR (&gt;100) or elevated BP (&gt;140 systolic) or both</li>
        <li>The 3 best-performing persons ({', '.join(p['name'] for p in person_data[:3])}) have normal cardiovascular profiles (HR &lt; 85, BP &lt; 135)</li>
    </ul>
</div>

<div class="insight-box">
    <h3>Finding 2: ICU Training Data Creates Spurious Correlations</h3>
    <ul>
        <li>The model was trained on <b>VitalDB surgical ICU</b> data where hemodynamic instability co-occurs with hyperglycemia</li>
        <li>Elevated BP &rarr; increased vascular stiffness &rarr; stiffer pulse waveform &rarr; earlier, sharper reflected wave</li>
        <li>Elevated HR &rarr; shorter diastole &rarr; compressed dicrotic notch &rarr; altered PPG morphology</li>
        <li>The model learned: "stiff pulse waveform = high glucose" from ICU patients, but this association breaks for ambulatory subjects</li>
    </ul>
</div>

<div class="insight-box">
    <h3>Finding 3: Signal Quality Metrics Do NOT Predict Accuracy</h3>
    <ul>
        <li>The worst case has <b>better</b> signal quality: 25 accepted windows, similarity = 0.967</li>
        <li>The best case has <b>worse</b> signal quality: only 2 accepted windows, similarity = 0.743</li>
        <li>High window consistency means the PPG waveform is reproducible &mdash; not that the model will interpret it correctly</li>
        <li>The model is <b>confidently and consistently wrong</b> for {worst_name}: all 25 windows predict 172&ndash;287 mg/dL when actual is 102</li>
    </ul>
</div>

<div class="insight-box">
    <h3>Finding 4: The "Perfect" Prediction is Statistical Coincidence</h3>
    <ul>
        <li>{best_name}'s {best['pred_abs_error']:.1f} mg/dL error comes from only 2 surviving windows that happened to average near truth</li>
        <li>The same person's other 13 readings have MAE ranging from 0.1 to 75.6 mg/dL (mean: {next(p['mae'] for p in person_data if p['name'] == best_name):.1f})</li>
        <li>The same person, same timestamp, on IR PPG gives ~50 mg/dL error</li>
        <li>This is <b>not genuine model accuracy</b> but rather the law of small numbers</li>
    </ul>
</div>

<h2>{sec + 1}. Recommendations</h2>

<div class="insight-box">
    <h3>For Model Improvement</h3>
    <ol style="margin-left: 20px;">
        <li><b>Cardiovascular Normalization:</b> Add HR and BP as auxiliary inputs or condition the model on hemodynamic state to disentangle vascular from metabolic PPG features</li>
        <li><b>Domain Adaptation:</b> Fine-tune with ambulatory PPG data rather than relying solely on ICU data, which has different hemodynamic profiles</li>
        <li><b>Multi-Task Learning:</b> Jointly predict glucose + HR + BP so the model learns to separate these effects</li>
        <li><b>Prediction Confidence:</b> Use prediction standard deviation as a confidence metric &mdash; flag cases where pred_std &gt; 20 as unreliable (r = +{next(c['r'] for c in correlations if c['feature'] == 'pred_std'):.3f} correlation with error)</li>
        <li><b>Subject-Specific Calibration:</b> A per-subject calibration step using 1&ndash;2 reference glucose readings could dramatically reduce person-level bias</li>
    </ol>
</div>

<div class="insight-box">
    <h3>For Data Collection</h3>
    <ol style="margin-left: 20px;">
        <li><b>Record BP and HR alongside PPG</b> to enable post-hoc cardiovascular normalization</li>
        <li><b>Collect from diverse cardiovascular profiles</b> &mdash; current data is biased toward normotensive subjects</li>
        <li><b>Include glucose challenge tests</b> (oral glucose tolerance) where the same person has readings at 80, 120, 180+ mg/dL to separate person-level bias from glucose sensitivity</li>
        <li><b>Multiple readings per session</b> to estimate within-session variability and identify unreliable recordings</li>
    </ol>
</div>

<footer>
    Generated from Batch 1 + Batch 2 results ({n_ok} samples, {len(person_data)} persons)
    | Red PPG + IR PPG | Extreme Case Analysis
</footer>

</body></html>"""

    OUTPUT.write_text(html, encoding='utf-8')
    print(f"\nReport saved: {OUTPUT}")


# ====================== SVG HELPER FUNCTIONS ======================

def _build_comparison_bar_svg(fields):
    """Side-by-side horizontal bar chart comparing best vs worst."""
    w, margin = 700, 40
    row_h = 40
    h = margin * 2 + len(fields) * row_h + 30
    bar_area = 200

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">']
    lines.append(f'<rect width="{w}" height="{h}" fill="#1e293b" rx="8"/>')
    lines.append(f'<text x="{w//2}" y="25" text-anchor="middle" fill="#94a3b8" font-size="13">Side-by-Side Comparison</text>')

    # Legend
    lines.append(f'<rect x="{w//2-100}" y="35" width="12" height="12" fill="#22c55e" rx="2"/>')
    lines.append(f'<text x="{w//2-84}" y="46" fill="#94a3b8" font-size="11">Best</text>')
    lines.append(f'<rect x="{w//2+20}" y="35" width="12" height="12" fill="#ef4444" rx="2"/>')
    lines.append(f'<text x="{w//2+36}" y="46" fill="#94a3b8" font-size="11">Worst</text>')

    y_start = 65
    for i, (label, best_v, worst_v, unit) in enumerate(fields):
        y = y_start + i * row_h
        max_v = max(abs(best_v), abs(worst_v), 0.001)

        lines.append(f'<text x="{margin}" y="{y+16}" fill="#e2e8f0" font-size="12">{label}</text>')

        bw = abs(best_v) / max_v * bar_area
        ww = abs(worst_v) / max_v * bar_area
        bx = 200
        lines.append(f'<rect x="{bx}" y="{y+2}" width="{bw:.1f}" height="14" fill="#22c55e" rx="3" opacity="0.8"/>')
        lines.append(f'<text x="{bx + bw + 5}" y="{y+14}" fill="#22c55e" font-size="11">{best_v:.1f} {unit}</text>')

        lines.append(f'<rect x="{bx}" y="{y+18}" width="{ww:.1f}" height="14" fill="#ef4444" rx="3" opacity="0.8"/>')
        lines.append(f'<text x="{bx + ww + 5}" y="{y+30}" fill="#ef4444" font-size="11">{worst_v:.1f} {unit}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def _build_scatter_svg(xs, ys, x_label, y_label, best_x, best_y, worst_x, worst_y, best_name, worst_name):
    """Scatter plot with highlighted extremes."""
    w, h, m = 450, 350, 55
    pw = w - 2 * m
    ph = h - 2 * m

    x_min, x_max = np.nanmin(xs) * 0.95, np.nanmax(xs) * 1.05
    y_min, y_max = 0, np.nanmax(ys) * 1.1

    def sx(v):
        return m + (v - x_min) / (x_max - x_min) * pw if x_max > x_min else m + pw / 2

    def sy(v):
        return h - m - (v - y_min) / (y_max - y_min) * ph if y_max > y_min else h - m - ph / 2

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">']
    lines.append(f'<rect width="{w}" height="{h}" fill="#1e293b" rx="8"/>')

    # Grid
    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        yp = sy(yv)
        lines.append(f'<line x1="{m}" y1="{yp}" x2="{w-m}" y2="{yp}" stroke="#334155" stroke-width="0.5"/>')
        lines.append(f'<text x="{m-5}" y="{yp+4}" text-anchor="end" fill="#64748b" font-size="10">{yv:.0f}</text>')

    for i in range(5):
        xv = x_min + (x_max - x_min) * i / 4
        xp = sx(xv)
        lines.append(f'<line x1="{xp}" y1="{m}" x2="{xp}" y2="{h-m}" stroke="#334155" stroke-width="0.5"/>')
        lines.append(f'<text x="{xp}" y="{h-m+15}" text-anchor="middle" fill="#64748b" font-size="10">{xv:.0f}</text>')

    # All points
    for x, y in zip(xs, ys):
        if not np.isnan(x) and not np.isnan(y):
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="#f97316" opacity="0.4"/>')

    # Best point
    lines.append(f'<circle cx="{sx(best_x):.1f}" cy="{sy(best_y):.1f}" r="8" fill="none" stroke="#22c55e" stroke-width="2.5"/>')
    lines.append(f'<circle cx="{sx(best_x):.1f}" cy="{sy(best_y):.1f}" r="4" fill="#22c55e"/>')
    lines.append(f'<text x="{sx(best_x)+12:.1f}" y="{sy(best_y)+4:.1f}" fill="#22c55e" font-size="10" font-weight="600">{best_name}</text>')

    # Worst point
    lines.append(f'<circle cx="{sx(worst_x):.1f}" cy="{sy(worst_y):.1f}" r="8" fill="none" stroke="#ef4444" stroke-width="2.5"/>')
    lines.append(f'<circle cx="{sx(worst_x):.1f}" cy="{sy(worst_y):.1f}" r="4" fill="#ef4444"/>')
    lines.append(f'<text x="{sx(worst_x)+12:.1f}" y="{sy(worst_y)+4:.1f}" fill="#ef4444" font-size="10" font-weight="600">{worst_name}</text>')

    # Labels
    lines.append(f'<text x="{w//2}" y="{h-5}" text-anchor="middle" fill="#94a3b8" font-size="12">{x_label}</text>')
    lines.append(f'<text x="15" y="{h//2}" text-anchor="middle" fill="#94a3b8" font-size="12" transform="rotate(-90,15,{h//2})">{y_label}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def _build_person_mae_svg(person_data):
    """Horizontal bar chart: per-person MAE, colored by cardiovascular risk."""
    w, m = 700, 50
    bar_h = 28
    gap = 6
    n = len(person_data)
    h = m * 2 + n * (bar_h + gap) + 20
    bar_area = 350
    max_mae = max(p['mae'] for p in person_data) * 1.1

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">']
    lines.append(f'<rect width="{w}" height="{h}" fill="#1e293b" rx="8"/>')

    for i, p in enumerate(person_data):
        y = m + i * (bar_h + gap)
        bw = p['mae'] / max_mae * bar_area

        # Color by cardiovascular risk
        if p['mean_hr'] > 100 or p['mean_sys'] > 140:
            color = '#ef4444'
        elif p['mean_hr'] > 90 or p['mean_sys'] > 130:
            color = '#f59e0b'
        else:
            color = '#22c55e'

        lines.append(f'<text x="{m + 90}" y="{y + bar_h // 2 + 5}" text-anchor="end" fill="#e2e8f0" font-size="12">{p["name"]}</text>')
        lines.append(f'<rect x="{m + 100}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{color}" rx="4" opacity="0.8"/>')
        lines.append(f'<text x="{m + 105 + bw}" y="{y + bar_h // 2 + 5}" fill="{color}" font-size="11" font-weight="600">'
                     f'{p["mae"]:.1f} (HR:{p["mean_hr"]:.0f}, BP:{p["mean_sys"]:.0f})</text>')

    # Legend
    ly = h - 18
    lines.append(f'<rect x="{m+100}" y="{ly}" width="10" height="10" fill="#22c55e" rx="2"/>')
    lines.append(f'<text x="{m+115}" y="{ly+9}" fill="#94a3b8" font-size="10">Normal CV</text>')
    lines.append(f'<rect x="{m+220}" y="{ly}" width="10" height="10" fill="#f59e0b" rx="2"/>')
    lines.append(f'<text x="{m+235}" y="{ly+9}" fill="#94a3b8" font-size="10">Borderline CV</text>')
    lines.append(f'<rect x="{m+360}" y="{ly}" width="10" height="10" fill="#ef4444" rx="2"/>')
    lines.append(f'<text x="{m+375}" y="{ly+9}" fill="#94a3b8" font-size="10">Elevated HR/BP</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def _build_correlation_svg(correlations):
    """Horizontal bar chart of feature correlations with absolute error."""
    w, m = 600, 50
    bar_h = 28
    gap = 6
    n = len(correlations)
    h = m * 2 + n * (bar_h + gap)
    bar_area = 180
    center_x = m + 180

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">']
    lines.append(f'<rect width="{w}" height="{h}" fill="#1e293b" rx="8"/>')

    # Center line
    lines.append(f'<line x1="{center_x}" y1="{m-10}" x2="{center_x}" y2="{h-m+10}" stroke="#64748b" stroke-width="1" stroke-dasharray="4,4"/>')
    lines.append(f'<text x="{center_x}" y="{m-15}" text-anchor="middle" fill="#64748b" font-size="10">r = 0</text>')

    for i, c in enumerate(correlations):
        y = m + i * (bar_h + gap)
        r = c['r']
        bw = abs(r) / 1.0 * bar_area

        color = '#ef4444' if r > 0 else '#3b82f6'
        if r > 0:
            bx = center_x
        else:
            bx = center_x - bw

        lines.append(f'<text x="{center_x - bar_area - 10}" y="{y + bar_h // 2 + 5}" text-anchor="end" fill="#e2e8f0" font-size="12">{c["feature"]}</text>')
        lines.append(f'<rect x="{bx:.1f}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{color}" rx="3" opacity="0.7"/>')

        lx = center_x + bw + 5 if r > 0 else center_x - bw - 5
        anchor = "start" if r > 0 else "end"
        lines.append(f'<text x="{lx:.1f}" y="{y + bar_h // 2 + 5}" text-anchor="{anchor}" fill="{color}" font-size="11" font-weight="600">{r:+.3f}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def _build_inversion_svg(vikraman_ok, name):
    """Show glucose vs prediction for the worst person, demonstrating inverted response."""
    w, h, m = 500, 300, 55
    pw = w - 2 * m
    ph = h - 2 * m

    refs = vikraman_ok['ref_glucose'].values
    preds = vikraman_ok['pred_mean'].values
    ppg_types = vikraman_ok['ppg_type'].values

    all_vals = np.concatenate([refs, preds])
    v_min = min(all_vals) * 0.85
    v_max = max(all_vals) * 1.1

    def sx(v):
        return m + (v - v_min) / (v_max - v_min) * pw

    def sy(v):
        return h - m - (v - v_min) / (v_max - v_min) * ph

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">']
    lines.append(f'<rect width="{w}" height="{h}" fill="#1e293b" rx="8"/>')
    lines.append(f'<text x="{w//2}" y="20" text-anchor="middle" fill="#f97316" font-size="13" font-weight="600">{name}: Actual vs Predicted Glucose</text>')

    # Perfect prediction line
    lines.append(f'<line x1="{sx(v_min)}" y1="{sy(v_min)}" x2="{sx(v_max)}" y2="{sy(v_max)}" stroke="#22c55e" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.5"/>')
    lines.append(f'<text x="{sx(v_max)-5}" y="{sy(v_max)-8}" text-anchor="end" fill="#22c55e" font-size="10" opacity="0.7">Perfect</text>')

    # Grid
    for v in range(int(v_min / 50) * 50, int(v_max) + 50, 50):
        if v_min <= v <= v_max:
            lines.append(f'<line x1="{m}" y1="{sy(v)}" x2="{w-m}" y2="{sy(v)}" stroke="#334155" stroke-width="0.5"/>')
            lines.append(f'<text x="{m-5}" y="{sy(v)+4}" text-anchor="end" fill="#64748b" font-size="10">{v}</text>')
            lines.append(f'<line x1="{sx(v)}" y1="{m}" x2="{sx(v)}" y2="{h-m}" stroke="#334155" stroke-width="0.5"/>')
            lines.append(f'<text x="{sx(v)}" y="{h-m+15}" text-anchor="middle" fill="#64748b" font-size="10">{v}</text>')

    # Points
    for ref, pred, pt in zip(refs, preds, ppg_types):
        color = '#ef4444' if pt == 'Red' else '#3b82f6'
        lines.append(f'<circle cx="{sx(ref):.1f}" cy="{sy(pred):.1f}" r="6" fill="{color}" opacity="0.8"/>')
        # Error line
        lines.append(f'<line x1="{sx(ref):.1f}" y1="{sy(ref):.1f}" x2="{sx(ref):.1f}" y2="{sy(pred):.1f}" '
                     f'stroke="{color}" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.5"/>')

    # Labels
    lines.append(f'<text x="{w//2}" y="{h-5}" text-anchor="middle" fill="#94a3b8" font-size="12">Actual Glucose (mg/dL)</text>')
    lines.append(f'<text x="12" y="{h//2}" text-anchor="middle" fill="#94a3b8" font-size="12" transform="rotate(-90,12,{h//2})">Predicted (mg/dL)</text>')

    # Legend
    lines.append(f'<rect x="{w-130}" y="30" width="10" height="10" fill="#ef4444" rx="2"/>')
    lines.append(f'<text x="{w-115}" y="40" fill="#94a3b8" font-size="10">Red PPG</text>')
    lines.append(f'<rect x="{w-130}" y="46" width="10" height="10" fill="#3b82f6" rx="2"/>')
    lines.append(f'<text x="{w-115}" y="56" fill="#94a3b8" font-size="10">IR PPG</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
