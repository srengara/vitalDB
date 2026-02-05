"""
Iteration Comparison Report
============================
Compares baseline (with single-peak check) vs Iteration 1 (skip single-peak check)
for Batch 1 Red + IR PPG to assess whether skipping the single-peak filter improves results.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Baseline: original results under LABPPG
BASELINE = [
    (Path(r"C:\IITM\vitalDB\data\LABPPG\VanillaPPG_Batch_1_500Hz\results\batch_summary.csv"), "Red"),
    (Path(r"C:\IITM\vitalDB\data\LABPPG\VanillaPPG_Batch_1_500Hz\results\iR_PPG\batch_summary.csv"), "IR"),
]

# Iteration 1: skip_single_peak_check results
ITER1 = [
    (Path(r"C:\IITM\vitalDB\data\VanillaPPG_Batch_1_500Hz\results\Iteration1\batch_summary.csv"), "Red"),
    (Path(r"C:\IITM\vitalDB\data\VanillaPPG_Batch_1_500Hz\results\Iteration1\iR_PPG\batch_summary.csv"), "IR"),
]

OUTPUT = Path(r"C:\IITM\vitalDB\data\LABPPG\ITERATION_COMPARISON_REPORT.html")


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


def load_summaries(spec, label):
    frames = []
    for path, ppg_type in spec:
        if not path.exists():
            print(f"  [SKIP] {path}")
            continue
        tmp = pd.read_csv(path)
        tmp['ppg_type'] = ppg_type
        tmp['iteration'] = label
        frames.append(tmp)
        print(f"  Loaded {label} {ppg_type}: {len(tmp)} rows")
    return pd.concat(frames, ignore_index=True)


def compute_stats(ok_df):
    if len(ok_df) == 0:
        return {}
    refs = ok_df['ref_glucose'].values
    preds = ok_df['pred_mean'].values
    errors = preds - refs
    abs_errors = np.abs(errors)
    zones = {}
    for r, p in zip(refs, preds):
        z = clarke_zone(r, p)
        zones[z] = zones.get(z, 0) + 1
    total = sum(zones.values())
    zone_pcts = {k: (v / total * 100) if total > 0 else 0 for k, v in zones.items()}
    for z in 'ABCDE':
        if z not in zone_pcts:
            zone_pcts[z] = 0.0
    return {
        'n': len(ok_df),
        'mae': np.mean(abs_errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mard': np.mean(abs_errors / refs) * 100,
        'bias': np.mean(errors),
        'r2': 1 - np.sum((refs - preds)**2) / max(np.sum((refs - refs.mean())**2), 1e-10),
        'within_20': np.mean(abs_errors <= 20) * 100,
        'within_30': np.mean(abs_errors <= 30) * 100,
        'zone_a': zone_pcts.get('A', 0),
        'zone_ab': zone_pcts.get('A', 0) + zone_pcts.get('B', 0),
        'mean_windows': ok_df['num_windows'].mean(),
        'mean_similarity': ok_df['mean_similarity'].mean(),
        'mean_pred_std': ok_df['pred_std'].mean(),
    }


def main():
    print("=" * 70)
    print("ITERATION COMPARISON: Baseline vs Iteration 1")
    print("=" * 70)

    base_df = load_summaries(BASELINE, "Baseline")
    iter1_df = load_summaries(ITER1, "Iteration 1")

    base_ok = base_df[(base_df['status'] == 'OK') & (base_df['name'] != 'Unknown')].copy()
    iter1_ok = iter1_df[(iter1_df['status'] == 'OK') & (iter1_df['name'] != 'Unknown')].copy()

    base_total = len(base_df[base_df['name'] != 'Unknown'])
    iter1_total = len(iter1_df[iter1_df['name'] != 'Unknown'])

    bs = compute_stats(base_ok)
    it = compute_stats(iter1_ok)

    print(f"\nBaseline: {bs['n']}/{base_total} OK, MAE={bs['mae']:.1f}")
    print(f"Iter1:    {it['n']}/{iter1_total} OK, MAE={it['mae']:.1f}")

    # Per-PPG comparison
    ppg_stats = {}
    for ppg in ['Red', 'IR']:
        ppg_stats[ppg] = {
            'base': compute_stats(base_ok[base_ok['ppg_type'] == ppg]),
            'iter1': compute_stats(iter1_ok[iter1_ok['ppg_type'] == ppg]),
        }

    # Per-person comparison
    all_persons = sorted(set(base_ok['name'].unique()) | set(iter1_ok['name'].unique()))
    person_compare = []
    for name in all_persons:
        b_sub = base_ok[base_ok['name'] == name]
        i_sub = iter1_ok[iter1_ok['name'] == name]
        b_st = compute_stats(b_sub) if len(b_sub) > 0 else None
        i_st = compute_stats(i_sub) if len(i_sub) > 0 else None
        person_compare.append({
            'name': name,
            'base_n': len(b_sub), 'iter1_n': len(i_sub),
            'base_mae': b_st['mae'] if b_st else float('nan'),
            'iter1_mae': i_st['mae'] if i_st else float('nan'),
            'base_mard': b_st['mard'] if b_st else float('nan'),
            'iter1_mard': i_st['mard'] if i_st else float('nan'),
            'base_windows': b_st['mean_windows'] if b_st else float('nan'),
            'iter1_windows': i_st['mean_windows'] if i_st else float('nan'),
            'base_zone_a': b_st['zone_a'] if b_st else float('nan'),
            'iter1_zone_a': i_st['zone_a'] if i_st else float('nan'),
        })

    # Per-sample matched comparison (same person/date/time/ppg_type)
    merge_keys = ['name', 'date', 'time', 'ppg_type']
    matched = base_ok[merge_keys + ['pred_mean', 'pred_abs_error', 'pred_mard', 'num_windows', 'mean_similarity',
                                     'ref_glucose', 'pred_error', 'windows_rejected_single_peak']].merge(
        iter1_ok[merge_keys + ['pred_mean', 'pred_abs_error', 'pred_mard', 'num_windows', 'mean_similarity',
                                'pred_error', 'windows_rejected_single_peak']],
        on=merge_keys, suffixes=('_base', '_iter1'),
        how='inner'
    )
    matched['mae_delta'] = matched['pred_abs_error_iter1'] - matched['pred_abs_error_base']
    matched['windows_delta'] = matched['num_windows_iter1'] - matched['num_windows_base']

    n_improved = (matched['mae_delta'] < 0).sum()
    n_worsened = (matched['mae_delta'] > 0).sum()
    n_unchanged = (matched['mae_delta'] == 0).sum()

    # Samples that are new in Iteration 1 (were FAILED in baseline, now OK)
    base_failed_keys = set()
    base_failed = base_df[(base_df['status'] != 'OK') & (base_df['name'] != 'Unknown')]
    for _, r in base_failed.iterrows():
        base_failed_keys.add((r['name'], r.get('date', ''), r.get('time', ''), r.get('ppg_type', '')))

    newly_ok = []
    for _, r in iter1_ok.iterrows():
        key = (r['name'], r['date'], r['time'], r['ppg_type'])
        if key in base_failed_keys:
            newly_ok.append(r)
    newly_ok_df = pd.DataFrame(newly_ok) if newly_ok else pd.DataFrame()

    # Samples that regressed (were OK in baseline, now FAILED)
    iter1_failed_keys = set()
    iter1_failed = iter1_df[(iter1_df['status'] != 'OK') & (iter1_df['name'] != 'Unknown')]
    for _, r in iter1_failed.iterrows():
        iter1_failed_keys.add((r['name'], r.get('date', ''), r.get('time', ''), r.get('ppg_type', '')))

    regressed = []
    for _, r in base_ok.iterrows():
        key = (r['name'], r['date'], r['time'], r['ppg_type'])
        if key in iter1_failed_keys:
            regressed.append(r)
    regressed_df = pd.DataFrame(regressed) if regressed else pd.DataFrame()

    # SVGs
    delta_svg = _build_delta_svg(matched)
    windows_svg = _build_windows_comparison_svg(matched)
    person_svg = _build_person_comparison_svg(person_compare)

    # ===== HTML =====
    mae_diff = it['mae'] - bs['mae']
    mae_css = 'good' if mae_diff < 0 else 'bad'
    mae_arrow = '&#x25BC;' if mae_diff < 0 else '&#x25B2;'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Iteration Comparison</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f23; color: #e2e8f0; line-height: 1.6; padding: 30px; }}
h1 {{ text-align: center; font-size: 2em; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px; }}
.subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 30px; font-size: 0.95em; }}
h2 {{ color: #667eea; border-bottom: 2px solid #667eea40; padding-bottom: 8px; margin: 35px 0 20px; font-size: 1.4em; }}
h3 {{ color: #818cf8; margin: 20px 0 12px; font-size: 1.1em; }}

.card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; margin: 20px 0; }}
.metric-card {{ background: #1e293b; border-radius: 10px; padding: 16px; text-align: center; }}
.metric-label {{ color: #94a3b8; font-size: 0.8em; margin-bottom: 4px; }}
.metric-value {{ font-size: 1.8em; font-weight: 700; }}
.metric-unit {{ color: #64748b; font-size: 0.75em; }}

table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #1e293b; border-radius: 8px; overflow: hidden; font-size: 0.88em; }}
th {{ background: #334155; padding: 10px 12px; text-align: center; color: #e2e8f0; font-weight: 600; }}
td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #334155; }}
tr:hover {{ background: #334155aa; }}
td.left {{ text-align: left; }}
.good {{ color: #22c55e; }}
.warn {{ color: #f59e0b; }}
.bad {{ color: #ef4444; }}
.neutral {{ color: #94a3b8; }}

.highlight-row {{ background: rgba(102,126,234,0.15) !important; font-weight: bold; }}
.insight-box {{ background: #1e293b; border: 1px solid #667eea40; border-radius: 12px; padding: 20px 24px; margin: 20px 0; }}
.insight-box h3 {{ color: #667eea; margin-top: 0; }}
.insight-box ul {{ margin-left: 20px; }}
.insight-box li {{ margin-bottom: 8px; }}
.chart-container {{ text-align: center; margin: 20px 0; overflow-x: auto; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

.delta-good {{ background: #22c55e20; color: #22c55e; padding: 2px 8px; border-radius: 6px; font-weight: 600; }}
.delta-bad {{ background: #ef444420; color: #ef4444; padding: 2px 8px; border-radius: 6px; font-weight: 600; }}
.delta-neutral {{ background: #64748b20; color: #94a3b8; padding: 2px 8px; border-radius: 6px; }}

footer {{ text-align: center; color: #475569; margin-top: 40px; font-size: 0.8em; padding-top: 20px; border-top: 1px solid #334155; }}
</style>
</head><body>

<h1>Baseline vs Iteration 1 Comparison</h1>
<p class="subtitle">
    Effect of skipping single-peak window check | Batch 1 only | Red PPG + IR PPG
</p>

<!-- ===== VERDICT ===== -->
<h2>1. Verdict</h2>
<div class="card-grid">
    <div class="metric-card">
        <div class="metric-label">MAE Change</div>
        <div class="metric-value {mae_css}">{mae_arrow} {abs(mae_diff):.1f}</div>
        <div class="metric-unit">mg/dL {'improvement' if mae_diff < 0 else 'regression'}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Success Rate</div>
        <div class="metric-value">{bs['n']}&rarr;{it['n']}</div>
        <div class="metric-unit">of {base_total} samples</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Matched Samples</div>
        <div class="metric-value">{len(matched)}</div>
        <div class="metric-unit">{n_improved} improved, {n_worsened} worsened</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Newly Recovered</div>
        <div class="metric-value">{len(newly_ok_df)}</div>
        <div class="metric-unit">were FAILED, now OK</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Newly Failed</div>
        <div class="metric-value {'bad' if len(regressed_df)>0 else ''}">{len(regressed_df)}</div>
        <div class="metric-unit">were OK, now FAILED</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Zone A</div>
        <div class="metric-value">{bs['zone_a']:.0f}%&rarr;{it['zone_a']:.0f}%</div>
        <div class="metric-unit">{'&#x25B2;' if it['zone_a'] > bs['zone_a'] else '&#x25BC;'} {abs(it['zone_a']-bs['zone_a']):.1f}pp</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Avg Windows</div>
        <div class="metric-value">{bs['mean_windows']:.0f}&rarr;{it['mean_windows']:.0f}</div>
        <div class="metric-unit">per sample</div>
    </div>
</div>

<!-- ===== OVERALL COMPARISON ===== -->
<h2>2. Overall Metrics Comparison</h2>
<table>
    <tr><th>Metric</th><th>Baseline</th><th>Iteration 1</th><th>Change</th></tr>"""

    metrics = [
        ('Succeeded / Total', f"{bs['n']} / {base_total}", f"{it['n']} / {iter1_total}", it['n'] - bs['n'], True),
        ('MAE (mg/dL)', f"{bs['mae']:.1f}", f"{it['mae']:.1f}", it['mae'] - bs['mae'], False),
        ('RMSE (mg/dL)', f"{bs['rmse']:.1f}", f"{it['rmse']:.1f}", it['rmse'] - bs['rmse'], False),
        ('MARD (%)', f"{bs['mard']:.1f}", f"{it['mard']:.1f}", it['mard'] - bs['mard'], False),
        ('Bias (mg/dL)', f"{bs['bias']:+.1f}", f"{it['bias']:+.1f}", abs(it['bias']) - abs(bs['bias']), False),
        ('R2', f"{bs['r2']:.3f}", f"{it['r2']:.3f}", it['r2'] - bs['r2'], True),
        ('Within +/-20 mg/dL (%)', f"{bs['within_20']:.1f}", f"{it['within_20']:.1f}", it['within_20'] - bs['within_20'], True),
        ('Within +/-30 mg/dL (%)', f"{bs['within_30']:.1f}", f"{it['within_30']:.1f}", it['within_30'] - bs['within_30'], True),
        ('Zone A (%)', f"{bs['zone_a']:.1f}", f"{it['zone_a']:.1f}", it['zone_a'] - bs['zone_a'], True),
        ('Zone A+B (%)', f"{bs['zone_ab']:.1f}", f"{it['zone_ab']:.1f}", it['zone_ab'] - bs['zone_ab'], True),
        ('Avg Windows/Sample', f"{bs['mean_windows']:.1f}", f"{it['mean_windows']:.1f}", it['mean_windows'] - bs['mean_windows'], True),
        ('Avg Similarity', f"{bs['mean_similarity']:.3f}", f"{it['mean_similarity']:.3f}", it['mean_similarity'] - bs['mean_similarity'], False),
        ('Avg Pred Std', f"{bs['mean_pred_std']:.1f}", f"{it['mean_pred_std']:.1f}", it['mean_pred_std'] - bs['mean_pred_std'], False),
    ]

    for name, bv, iv, delta, higher_is_better in metrics:
        if higher_is_better:
            css = 'good' if delta > 0.5 else ('bad' if delta < -0.5 else 'neutral')
        else:
            css = 'good' if delta < -0.5 else ('bad' if delta > 0.5 else 'neutral')
        arrow = '+' if delta > 0 else ''
        html += f"""
    <tr><td class="left"><b>{name}</b></td><td>{bv}</td><td>{iv}</td>
    <td class="{css}">{arrow}{delta:.1f}</td></tr>"""

    html += """
</table>

<!-- ===== PPG TYPE ===== -->
<h2>3. By PPG Type</h2>
<table>
    <tr><th>PPG</th><th colspan="3">Baseline</th><th colspan="3">Iteration 1</th><th colspan="3">Change</th></tr>
    <tr><th></th><th>N</th><th>MAE</th><th>Zone A</th><th>N</th><th>MAE</th><th>Zone A</th><th>&Delta;N</th><th>&Delta;MAE</th><th>&Delta;Zone A</th></tr>"""

    for ppg in ['Red', 'IR']:
        b = ppg_stats[ppg]['base']
        i = ppg_stats[ppg]['iter1']
        if b and i:
            d_mae = i['mae'] - b['mae']
            d_za = i['zone_a'] - b['zone_a']
            html += f"""
    <tr>
        <td><b>{ppg}</b></td>
        <td>{b['n']}</td><td>{b['mae']:.1f}</td><td>{b['zone_a']:.1f}%</td>
        <td>{i['n']}</td><td>{i['mae']:.1f}</td><td>{i['zone_a']:.1f}%</td>
        <td>{i['n']-b['n']:+d}</td>
        <td class="{'good' if d_mae<0 else 'bad'}">{d_mae:+.1f}</td>
        <td class="{'good' if d_za>0 else 'bad'}">{d_za:+.1f}pp</td>
    </tr>"""

    html += "</table>"

    # ===== Per-sample delta chart =====
    html += f"""
<h2>4. Per-Sample Impact (Matched: {len(matched)})</h2>
<div class="chart-container">{delta_svg}</div>

<div class="insight-box">
    <h3>Window Count vs Accuracy Trade-off</h3>
    <p>Skipping the single-peak check dramatically increases window count (avg {bs['mean_windows']:.0f} &rarr; {it['mean_windows']:.0f}),
    but many of the additional windows contain multi-peak artifacts that add noise to predictions.</p>
    <ul>
        <li><b>{n_improved} samples improved</b> (more windows helped averaging)</li>
        <li><b>{n_worsened} samples worsened</b> (noisy windows degraded predictions)</li>
        <li>Median MAE change: <b>{matched['mae_delta'].median():+.1f} mg/dL</b></li>
    </ul>
</div>

<div class="chart-container">{windows_svg}</div>
"""

    # ===== Newly recovered samples =====
    if len(newly_ok_df) > 0:
        html += f"""
<h2>5. Newly Recovered Samples ({len(newly_ok_df)})</h2>
<p style="color:#94a3b8;">These samples failed in baseline but succeeded in Iteration 1.</p>
<table>
    <tr><th>Person</th><th>Date</th><th>Time</th><th>PPG</th><th>Ref</th><th>Pred</th><th>Error</th><th>MARD</th><th>Windows</th></tr>"""
        for _, r in newly_ok_df.iterrows():
            err = r['pred_error']
            css = 'good' if abs(err) < 20 else 'warn' if abs(err) < 40 else 'bad'
            html += f"""
    <tr>
        <td class="left">{r['name']}</td><td>{r['date']}</td><td>{r['time']}</td><td>{r['ppg_type']}</td>
        <td>{r['ref_glucose']:.0f}</td><td>{r['pred_mean']:.1f}</td>
        <td class="{css}">{err:+.1f}</td><td>{r['pred_mard']:.1f}%</td><td>{r['num_windows']:.0f}</td>
    </tr>"""
        html += "</table>"
        newly_mae = np.mean(np.abs(newly_ok_df['pred_error'].values))
        html += f"""
<div class="insight-box">
    <p>Recovered samples have MAE = <b>{newly_mae:.1f} mg/dL</b>. """
        if newly_mae > it['mae']:
            html += "These are <b>worse than average</b>, suggesting these recordings have fundamental quality issues beyond the single-peak filter."
        else:
            html += "These are <b>better than average</b>, suggesting the single-peak filter was overly aggressive for these signals."
        html += "</p></div>"
        sec = 6
    else:
        sec = 5

    # ===== Regressed samples =====
    if len(regressed_df) > 0:
        html += f"""
<h2>{sec}. Regressed Samples ({len(regressed_df)})</h2>
<p style="color:#94a3b8;">These samples were OK in baseline but failed in Iteration 1.</p>
<table>
    <tr><th>Person</th><th>Date</th><th>Time</th><th>PPG</th><th>Ref</th><th>Baseline Pred</th><th>Baseline Error</th></tr>"""
        for _, r in regressed_df.iterrows():
            html += f"""
    <tr>
        <td class="left">{r['name']}</td><td>{r['date']}</td><td>{r['time']}</td><td>{r['ppg_type']}</td>
        <td>{r['ref_glucose']:.0f}</td><td>{r['pred_mean']:.1f}</td><td>{r['pred_error']:+.1f}</td>
    </tr>"""
        html += "</table>"
        sec += 1

    # ===== Per-person comparison =====
    html += f"""
<h2>{sec}. Per-Person Comparison</h2>
<div class="chart-container">{person_svg}</div>
<table>
    <tr><th>Person</th><th colspan="2">N Samples</th><th colspan="2">MAE</th><th>&Delta;MAE</th><th colspan="2">Avg Windows</th><th colspan="2">Zone A</th></tr>
    <tr><th></th><th>Base</th><th>Iter1</th><th>Base</th><th>Iter1</th><th></th><th>Base</th><th>Iter1</th><th>Base</th><th>Iter1</th></tr>"""

    for p in sorted(person_compare, key=lambda x: x.get('iter1_mae', 999) - x.get('base_mae', 0)):
        d = p['iter1_mae'] - p['base_mae'] if not (np.isnan(p['iter1_mae']) or np.isnan(p['base_mae'])) else float('nan')
        d_str = f"{d:+.1f}" if not np.isnan(d) else '&mdash;'
        d_css = 'good' if not np.isnan(d) and d < 0 else ('bad' if not np.isnan(d) and d > 0 else '')
        html += f"""
    <tr>
        <td class="left"><b>{p['name']}</b></td>
        <td>{p['base_n']}</td><td>{p['iter1_n']}</td>
        <td>{f"{p['base_mae']:.1f}" if not np.isnan(p['base_mae']) else '&mdash;'}</td>
        <td>{f"{p['iter1_mae']:.1f}" if not np.isnan(p['iter1_mae']) else '&mdash;'}</td>
        <td class="{d_css}">{d_str}</td>
        <td>{f"{p['base_windows']:.0f}" if not np.isnan(p['base_windows']) else '&mdash;'}</td>
        <td>{f"{p['iter1_windows']:.0f}" if not np.isnan(p['iter1_windows']) else '&mdash;'}</td>
        <td>{f"{p['base_zone_a']:.0f}" if not np.isnan(p['base_zone_a']) else '&mdash;'}%</td>
        <td>{f"{p['iter1_zone_a']:.0f}" if not np.isnan(p['iter1_zone_a']) else '&mdash;'}%</td>
    </tr>"""

    html += f"""
</table>

<h2>{sec+1}. Conclusion</h2>
<div class="insight-box">
    <h3>Impact of Skipping Single-Peak Check</h3>
    <ul>
        <li><b>Success rate:</b> {bs['n']} &rarr; {it['n']} samples ({it['n']-bs['n']:+d} change, {len(newly_ok_df)} recovered, {len(regressed_df)} regressed)</li>
        <li><b>MAE:</b> {bs['mae']:.1f} &rarr; {it['mae']:.1f} mg/dL (<span class="{mae_css}">{mae_diff:+.1f}</span>)</li>
        <li><b>Zone A:</b> {bs['zone_a']:.1f}% &rarr; {it['zone_a']:.1f}% ({it['zone_a']-bs['zone_a']:+.1f}pp)</li>
        <li><b>Windows per sample:</b> {bs['mean_windows']:.0f} &rarr; {it['mean_windows']:.0f} ({it['mean_windows']-bs['mean_windows']:+.0f})</li>
        <li>Of {len(matched)} matched samples: <span class="good">{n_improved} improved</span>, <span class="bad">{n_worsened} worsened</span>, {n_unchanged} unchanged</li>
    </ul>"""

    if mae_diff > 0:
        html += f"""
    <p style="margin-top:12px;"><b>Overall verdict:</b> Skipping the single-peak check <span class="bad">worsened</span> accuracy.
    While it recovered {len(newly_ok_df)} previously-failed samples, the additional multi-peak windows added noise that
    degraded predictions on average. The single-peak filter acts as a quality gate that, despite reducing window count,
    ensures cleaner PPG morphology for prediction.</p>"""
    else:
        html += f"""
    <p style="margin-top:12px;"><b>Overall verdict:</b> Skipping the single-peak check <span class="good">improved</span> accuracy.
    The additional windows provided better averaging, and the {len(newly_ok_df)} recovered samples expanded coverage.
    The cosine similarity filter is sufficient quality control without the single-peak check.</p>"""

    html += """
</div>

<footer>
    Batch 1 Red + IR PPG | Baseline (with single-peak) vs Iteration 1 (skip single-peak)
</footer>
</body></html>"""

    OUTPUT.write_text(html, encoding='utf-8')
    print(f"\nReport saved: {OUTPUT}")


def _build_delta_svg(matched):
    """Waterfall/bar chart of per-sample MAE delta."""
    w, m = 750, 50
    n = len(matched)
    bar_w = max(3, min(12, (w - 2 * m) // max(n, 1)))
    h = 300
    ph = h - 2 * m

    sorted_m = matched.sort_values('mae_delta')
    deltas = sorted_m['mae_delta'].values
    max_abs = max(abs(deltas.min()), abs(deltas.max()), 1)

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">']
    lines.append(f'<rect width="{w}" height="{h}" fill="#1e293b" rx="8"/>')
    lines.append(f'<text x="{w//2}" y="20" text-anchor="middle" fill="#94a3b8" font-size="12">Per-Sample MAE Change (sorted, green=improved, red=worsened)</text>')

    center_y = m + ph // 2

    # Zero line
    lines.append(f'<line x1="{m}" y1="{center_y}" x2="{w-m}" y2="{center_y}" stroke="#64748b" stroke-width="1"/>')
    lines.append(f'<text x="{m-5}" y="{center_y+4}" text-anchor="end" fill="#64748b" font-size="10">0</text>')

    # Scale labels
    for frac in [-1, -0.5, 0.5, 1]:
        yy = center_y - (frac * ph / 2)
        val = frac * max_abs
        lines.append(f'<line x1="{m}" y1="{yy}" x2="{w-m}" y2="{yy}" stroke="#334155" stroke-width="0.5"/>')
        lines.append(f'<text x="{m-5}" y="{yy+4}" text-anchor="end" fill="#64748b" font-size="9">{val:+.0f}</text>')

    for i, d in enumerate(deltas):
        x = m + i * bar_w
        bar_h = abs(d) / max_abs * (ph / 2)
        color = '#22c55e' if d < 0 else '#ef4444'
        if d < 0:
            y = center_y - bar_h
        else:
            y = center_y
        lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w-1}" height="{bar_h:.1f}" fill="{color}" opacity="0.7"/>')

    lines.append(f'<text x="{w//2}" y="{h-5}" text-anchor="middle" fill="#64748b" font-size="10">Samples (sorted by MAE change)</text>')
    lines.append('</svg>')
    return '\n'.join(lines)


def _build_windows_comparison_svg(matched):
    """Scatter: window count change vs MAE change."""
    w, h, m = 450, 300, 55
    pw, ph = w - 2 * m, h - 2 * m

    wx = matched['windows_delta'].values
    wy = matched['mae_delta'].values

    x_min, x_max = min(wx.min(), -1), max(wx.max(), 1)
    y_min, y_max = min(wy.min(), -10), max(wy.max(), 10)

    def sx(v):
        return m + (v - x_min) / (x_max - x_min) * pw
    def sy(v):
        return h - m - (v - y_min) / (y_max - y_min) * ph

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">']
    lines.append(f'<rect width="{w}" height="{h}" fill="#1e293b" rx="8"/>')
    lines.append(f'<text x="{w//2}" y="18" text-anchor="middle" fill="#94a3b8" font-size="12">Window Count Change vs MAE Change</text>')

    # Zero lines
    if x_min <= 0 <= x_max:
        lines.append(f'<line x1="{sx(0)}" y1="{m}" x2="{sx(0)}" y2="{h-m}" stroke="#64748b" stroke-width="0.5" stroke-dasharray="4,4"/>')
    if y_min <= 0 <= y_max:
        lines.append(f'<line x1="{m}" y1="{sy(0)}" x2="{w-m}" y2="{sy(0)}" stroke="#64748b" stroke-width="0.5" stroke-dasharray="4,4"/>')

    for x, y in zip(wx, wy):
        color = '#22c55e' if y < 0 else '#ef4444'
        lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}" opacity="0.6"/>')

    lines.append(f'<text x="{w//2}" y="{h-5}" text-anchor="middle" fill="#94a3b8" font-size="11">&Delta; Windows (Iter1 - Baseline)</text>')
    lines.append(f'<text x="12" y="{h//2}" text-anchor="middle" fill="#94a3b8" font-size="11" transform="rotate(-90,12,{h//2})">&Delta; MAE (mg/dL)</text>')
    lines.append('</svg>')
    return '\n'.join(lines)


def _build_person_comparison_svg(person_compare):
    """Grouped bar chart: per-person MAE baseline vs iter1."""
    w, m = 700, 50
    bar_h = 18
    gap = 8
    n = len(person_compare)
    h = m * 2 + n * (bar_h * 2 + gap) + 30
    max_mae = max(max(p['base_mae'] for p in person_compare if not np.isnan(p['base_mae'])),
                  max(p['iter1_mae'] for p in person_compare if not np.isnan(p['iter1_mae'])), 1) * 1.1
    bar_area = 350

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">']
    lines.append(f'<rect width="{w}" height="{h}" fill="#1e293b" rx="8"/>')

    # Legend
    lines.append(f'<rect x="{m+100}" y="10" width="12" height="12" fill="#94a3b8" rx="2"/>')
    lines.append(f'<text x="{m+116}" y="21" fill="#94a3b8" font-size="11">Baseline</text>')
    lines.append(f'<rect x="{m+210}" y="10" width="12" height="12" fill="#667eea" rx="2"/>')
    lines.append(f'<text x="{m+226}" y="21" fill="#667eea" font-size="11">Iteration 1</text>')

    y_start = 35
    for i, p in enumerate(person_compare):
        y = y_start + i * (bar_h * 2 + gap)
        bx = m + 90

        lines.append(f'<text x="{bx-5}" y="{y + bar_h + 4}" text-anchor="end" fill="#e2e8f0" font-size="11">{p["name"]}</text>')

        bw = (p['base_mae'] / max_mae * bar_area) if not np.isnan(p['base_mae']) else 0
        iw = (p['iter1_mae'] / max_mae * bar_area) if not np.isnan(p['iter1_mae']) else 0

        lines.append(f'<rect x="{bx}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="#94a3b8" rx="3" opacity="0.6"/>')
        if not np.isnan(p['base_mae']):
            lines.append(f'<text x="{bx+bw+4}" y="{y+13}" fill="#94a3b8" font-size="10">{p["base_mae"]:.1f}</text>')

        lines.append(f'<rect x="{bx}" y="{y+bar_h}" width="{iw:.1f}" height="{bar_h}" fill="#667eea" rx="3" opacity="0.8"/>')
        if not np.isnan(p['iter1_mae']):
            lines.append(f'<text x="{bx+iw+4}" y="{y+bar_h+13}" fill="#667eea" font-size="10">{p["iter1_mae"]:.1f}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
