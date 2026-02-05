"""
Compare Lab PPG Inference Results from Two Models
Generates HTML report comparing model 174012 vs 163632
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Reference glucose values
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

def load_predictions(sample_dir, model_id):
    """Load predictions for a specific model"""
    pred_file = sample_dir / f'predictions_{model_id}' / 'glucose_predictions.csv'
    if not pred_file.exists():
        return None
    df = pd.read_csv(pred_file)
    return df['predicted_glucose_mg_dl'].values

def analyze_model(data_dir, model_id):
    """Analyze predictions for a model across all samples"""
    results = []

    for sample_name, ref_glucose in REFERENCE.items():
        sample_dir = data_dir / sample_name
        predictions = load_predictions(sample_dir, model_id)

        if predictions is None:
            continue

        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        pred_median = np.median(predictions)
        error = pred_mean - ref_glucose
        abs_error = abs(error)
        rel_error = abs_error / ref_glucose * 100

        results.append({
            'sample': sample_name,
            'reference': ref_glucose,
            'predicted_mean': pred_mean,
            'predicted_std': pred_std,
            'predicted_median': pred_median,
            'error': error,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'num_windows': len(predictions)
        })

    return pd.DataFrame(results)

def classify_glucose(g):
    """Classify glucose into clinical categories"""
    if g < 70: return 'Hypoglycemic'
    elif g < 100: return 'Normal'
    elif g < 126: return 'Prediabetes'
    else: return 'Diabetic'

def main():
    data_dir = Path(r"C:\IITM\vitalDB\data\LABPPG\ppg_windows_dir")

    # Analyze both models
    print("Analyzing Model 174012...")
    df_174012 = analyze_model(data_dir, '174012')

    print("Analyzing Model 163632...")
    df_163632 = analyze_model(data_dir, '163632')

    # Calculate summary metrics
    def calc_metrics(df):
        return {
            'mae': df['abs_error'].mean(),
            'rmse': np.sqrt((df['error']**2).mean()),
            'mard': df['rel_error'].mean(),
            'max_error': df['abs_error'].max(),
            'min_error': df['abs_error'].min(),
        }

    metrics_174012 = calc_metrics(df_174012)
    metrics_163632 = calc_metrics(df_163632)

    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    print(f"\nModel 174012:")
    print(f"  MAE: {metrics_174012['mae']:.2f} mg/dL")
    print(f"  RMSE: {metrics_174012['rmse']:.2f} mg/dL")
    print(f"  MARD: {metrics_174012['mard']:.2f}%")

    print(f"\nModel 163632:")
    print(f"  MAE: {metrics_163632['mae']:.2f} mg/dL")
    print(f"  RMSE: {metrics_163632['rmse']:.2f} mg/dL")
    print(f"  MARD: {metrics_163632['mard']:.2f}%")

    # Determine better model
    better_mae = '174012' if metrics_174012['mae'] < metrics_163632['mae'] else '163632'
    better_mard = '174012' if metrics_174012['mard'] < metrics_163632['mard'] else '163632'

    # Generate HTML Report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Lab PPG Model Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8;
            padding: 30px;
            max-width: 1200px;
            margin: 0 auto;
            line-height: 1.6;
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            font-size: 2.2em;
            margin-bottom: 10px;
        }}
        h2 {{
            color: #764ba2;
            border-bottom: 2px solid #764ba2;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        .subtitle {{
            text-align: center;
            color: #bdc3c7;
            margin-bottom: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: rgba(255,255,255,0.02);
        }}
        tr:hover {{
            background: rgba(102,126,234,0.1);
        }}
        .good {{ color: #2ecc71; font-weight: bold; }}
        .warn {{ color: #f39c12; font-weight: bold; }}
        .bad {{ color: #e74c3c; font-weight: bold; }}
        .better {{ background: rgba(46,204,113,0.2) !important; }}
        .metric-container {{
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin: 30px 0;
        }}
        .metric-card {{
            background: rgba(102,126,234,0.15);
            border-radius: 15px;
            padding: 25px 35px;
            text-align: center;
            min-width: 200px;
            border: 1px solid rgba(102,126,234,0.3);
        }}
        .metric-card.winner {{
            border: 2px solid #2ecc71;
            box-shadow: 0 0 20px rgba(46,204,113,0.3);
        }}
        .metric-label {{
            color: #bdc3c7;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-model {{
            font-size: 0.85em;
            color: #95a5a6;
            margin-top: 5px;
        }}
        .comparison-row {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .model-box {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            width: 45%;
            text-align: center;
        }}
        .model-box h3 {{
            color: #667eea;
            margin-top: 0;
        }}
        .insight-box {{
            background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 10px 10px 0;
        }}
        .insight-box h3 {{
            color: #667eea;
            margin-top: 0;
        }}
        .bar-container {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .bar-label {{
            width: 100px;
            text-align: right;
            padding-right: 10px;
            font-size: 0.9em;
        }}
        .bar-wrapper {{
            flex: 1;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            height: 25px;
            position: relative;
        }}
        .bar {{
            height: 100%;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .bar-174012 {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }}
        .bar-163632 {{
            background: linear-gradient(90deg, #f39c12 0%, #e74c3c 100%);
        }}
    </style>
</head>
<body>
    <h1>Lab PPG Model Comparison Report</h1>
    <p class="subtitle">Model 174012 vs Model 163632 | 8 Lab Samples</p>

    <h2>Summary Metrics</h2>
    <div class="metric-container">
        <div class="metric-card {'winner' if better_mae == '174012' else ''}">
            <div class="metric-label">MAE (Model 174012)</div>
            <div class="metric-value">{metrics_174012['mae']:.1f}</div>
            <div class="metric-model">mg/dL</div>
        </div>
        <div class="metric-card {'winner' if better_mae == '163632' else ''}">
            <div class="metric-label">MAE (Model 163632)</div>
            <div class="metric-value">{metrics_163632['mae']:.1f}</div>
            <div class="metric-model">mg/dL</div>
        </div>
    </div>

    <div class="metric-container">
        <div class="metric-card {'winner' if better_mard == '174012' else ''}">
            <div class="metric-label">MARD (Model 174012)</div>
            <div class="metric-value">{metrics_174012['mard']:.1f}%</div>
            <div class="metric-model">Mean Abs Rel Diff</div>
        </div>
        <div class="metric-card {'winner' if better_mard == '163632' else ''}">
            <div class="metric-label">MARD (Model 163632)</div>
            <div class="metric-value">{metrics_163632['mard']:.1f}%</div>
            <div class="metric-model">Mean Abs Rel Diff</div>
        </div>
    </div>

    <h2>Comparison Table</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Model 174012</th>
            <th>Model 163632</th>
            <th>Better</th>
            <th>Difference</th>
        </tr>
        <tr>
            <td><b>MAE</b></td>
            <td class="{'good' if better_mae == '174012' else ''}">{metrics_174012['mae']:.2f} mg/dL</td>
            <td class="{'good' if better_mae == '163632' else ''}">{metrics_163632['mae']:.2f} mg/dL</td>
            <td class="good">{better_mae}</td>
            <td>{abs(metrics_174012['mae'] - metrics_163632['mae']):.2f} mg/dL</td>
        </tr>
        <tr>
            <td><b>RMSE</b></td>
            <td class="{'good' if metrics_174012['rmse'] < metrics_163632['rmse'] else ''}">{metrics_174012['rmse']:.2f} mg/dL</td>
            <td class="{'good' if metrics_163632['rmse'] < metrics_174012['rmse'] else ''}">{metrics_163632['rmse']:.2f} mg/dL</td>
            <td class="good">{'174012' if metrics_174012['rmse'] < metrics_163632['rmse'] else '163632'}</td>
            <td>{abs(metrics_174012['rmse'] - metrics_163632['rmse']):.2f} mg/dL</td>
        </tr>
        <tr>
            <td><b>MARD</b></td>
            <td class="{'good' if better_mard == '174012' else ''}">{metrics_174012['mard']:.2f}%</td>
            <td class="{'good' if better_mard == '163632' else ''}">{metrics_163632['mard']:.2f}%</td>
            <td class="good">{better_mard}</td>
            <td>{abs(metrics_174012['mard'] - metrics_163632['mard']):.2f}%</td>
        </tr>
        <tr>
            <td><b>Max Error</b></td>
            <td>{metrics_174012['max_error']:.2f} mg/dL</td>
            <td>{metrics_163632['max_error']:.2f} mg/dL</td>
            <td class="good">{'174012' if metrics_174012['max_error'] < metrics_163632['max_error'] else '163632'}</td>
            <td>{abs(metrics_174012['max_error'] - metrics_163632['max_error']):.2f} mg/dL</td>
        </tr>
    </table>

    <h2>Sample-wise Comparison</h2>
    <table>
        <tr>
            <th>Sample</th>
            <th>Reference</th>
            <th>Model 174012</th>
            <th>Error 174012</th>
            <th>Model 163632</th>
            <th>Error 163632</th>
            <th>Better</th>
        </tr>"""

    # Add sample rows
    for i, sample in enumerate(REFERENCE.keys()):
        ref = REFERENCE[sample]

        row_174012 = df_174012[df_174012['sample'] == sample].iloc[0] if len(df_174012[df_174012['sample'] == sample]) > 0 else None
        row_163632 = df_163632[df_163632['sample'] == sample].iloc[0] if len(df_163632[df_163632['sample'] == sample]) > 0 else None

        if row_174012 is None or row_163632 is None:
            continue

        pred_174012 = row_174012['predicted_mean']
        err_174012 = row_174012['error']
        abs_err_174012 = row_174012['abs_error']

        pred_163632 = row_163632['predicted_mean']
        err_163632 = row_163632['error']
        abs_err_163632 = row_163632['abs_error']

        better = '174012' if abs_err_174012 < abs_err_163632 else '163632'

        err_class_174012 = 'good' if abs_err_174012 < 15 else ('warn' if abs_err_174012 < 30 else 'bad')
        err_class_163632 = 'good' if abs_err_163632 < 15 else ('warn' if abs_err_163632 < 30 else 'bad')

        html += f"""
        <tr class="{'better' if better == '174012' else ''}">
            <td><b>{sample}</b></td>
            <td>{ref}</td>
            <td>{pred_174012:.1f}</td>
            <td class="{err_class_174012}">{err_174012:+.1f}</td>
            <td>{pred_163632:.1f}</td>
            <td class="{err_class_163632}">{err_163632:+.1f}</td>
            <td class="good">{better}</td>
        </tr>"""

    html += """
    </table>

    <h2>Visual Comparison: Error by Sample</h2>
    <div style="margin: 20px 0;">"""

    # Add bar chart for each sample
    max_error = max(df_174012['abs_error'].max(), df_163632['abs_error'].max())

    for sample in REFERENCE.keys():
        row_174012 = df_174012[df_174012['sample'] == sample].iloc[0] if len(df_174012[df_174012['sample'] == sample]) > 0 else None
        row_163632 = df_163632[df_163632['sample'] == sample].iloc[0] if len(df_163632[df_163632['sample'] == sample]) > 0 else None

        if row_174012 is None or row_163632 is None:
            continue

        err_174012 = row_174012['abs_error']
        err_163632 = row_163632['abs_error']

        width_174012 = (err_174012 / max_error) * 100
        width_163632 = (err_163632 / max_error) * 100

        html += f"""
        <div style="margin-bottom: 15px;">
            <div style="font-weight: bold; margin-bottom: 5px;">{sample} (Ref: {REFERENCE[sample]} mg/dL)</div>
            <div class="bar-container">
                <div class="bar-label">174012</div>
                <div class="bar-wrapper">
                    <div class="bar bar-174012" style="width: {width_174012}%">{err_174012:.1f}</div>
                </div>
            </div>
            <div class="bar-container">
                <div class="bar-label">163632</div>
                <div class="bar-wrapper">
                    <div class="bar bar-163632" style="width: {width_163632}%">{err_163632:.1f}</div>
                </div>
            </div>
        </div>"""

    html += """
    </div>

    <h2>Clinical Classification Accuracy</h2>
    <table>
        <tr>
            <th>Sample</th>
            <th>Reference Class</th>
            <th>Model 174012 Class</th>
            <th>Match 174012</th>
            <th>Model 163632 Class</th>
            <th>Match 163632</th>
        </tr>"""

    correct_174012 = 0
    correct_163632 = 0
    total = 0

    for sample in REFERENCE.keys():
        ref = REFERENCE[sample]
        ref_class = classify_glucose(ref)

        row_174012 = df_174012[df_174012['sample'] == sample].iloc[0] if len(df_174012[df_174012['sample'] == sample]) > 0 else None
        row_163632 = df_163632[df_163632['sample'] == sample].iloc[0] if len(df_163632[df_163632['sample'] == sample]) > 0 else None

        if row_174012 is None or row_163632 is None:
            continue

        total += 1
        pred_class_174012 = classify_glucose(row_174012['predicted_mean'])
        pred_class_163632 = classify_glucose(row_163632['predicted_mean'])

        match_174012 = 'Yes' if ref_class == pred_class_174012 else 'No'
        match_163632 = 'Yes' if ref_class == pred_class_163632 else 'No'

        if match_174012 == 'Yes': correct_174012 += 1
        if match_163632 == 'Yes': correct_163632 += 1

        html += f"""
        <tr>
            <td>{sample}</td>
            <td>{ref_class}</td>
            <td>{pred_class_174012}</td>
            <td class="{'good' if match_174012 == 'Yes' else 'bad'}">{match_174012}</td>
            <td>{pred_class_163632}</td>
            <td class="{'good' if match_163632 == 'Yes' else 'bad'}">{match_163632}</td>
        </tr>"""

    acc_174012 = (correct_174012 / total) * 100 if total > 0 else 0
    acc_163632 = (correct_163632 / total) * 100 if total > 0 else 0

    html += f"""
        <tr style="background: rgba(102,126,234,0.2);">
            <td colspan="2"><b>Classification Accuracy</b></td>
            <td colspan="2" class="{'good' if acc_174012 >= acc_163632 else ''}">{correct_174012}/{total} ({acc_174012:.1f}%)</td>
            <td colspan="2" class="{'good' if acc_163632 >= acc_174012 else ''}">{correct_163632}/{total} ({acc_163632:.1f}%)</td>
        </tr>
    </table>

    <h2>Key Insights</h2>
    <div class="insight-box">
        <h3>Model Performance Analysis</h3>
        <ul>
            <li><b>Overall Winner:</b> Model <span class="good">{better_mae}</span> has lower MAE ({min(metrics_174012['mae'], metrics_163632['mae']):.1f} vs {max(metrics_174012['mae'], metrics_163632['mae']):.1f} mg/dL)</li>
            <li><b>MAE Difference:</b> {abs(metrics_174012['mae'] - metrics_163632['mae']):.1f} mg/dL between models</li>
            <li><b>MARD Difference:</b> {abs(metrics_174012['mard'] - metrics_163632['mard']):.1f}% between models</li>
            <li><b>Clinical Classification:</b> Model 174012 = {acc_174012:.0f}%, Model 163632 = {acc_163632:.0f}%</li>
        </ul>
    </div>

    <div class="insight-box">
        <h3>Prediction Behavior</h3>
        <ul>
            <li><b>Model 174012:</b> Mean prediction = {df_174012['predicted_mean'].mean():.1f} mg/dL (Range: {df_174012['predicted_mean'].min():.1f} - {df_174012['predicted_mean'].max():.1f})</li>
            <li><b>Model 163632:</b> Mean prediction = {df_163632['predicted_mean'].mean():.1f} mg/dL (Range: {df_163632['predicted_mean'].min():.1f} - {df_163632['predicted_mean'].max():.1f})</li>
            <li><b>Reference Range:</b> {min(REFERENCE.values())} - {max(REFERENCE.values())} mg/dL</li>
        </ul>
    </div>

    <div class="insight-box">
        <h3>Recommendations</h3>
        <ul>"""

    # Add specific recommendations based on analysis
    if metrics_174012['mae'] < metrics_163632['mae']:
        html += f"""
            <li>Model 174012 shows better overall accuracy with {metrics_174012['mae'] - metrics_163632['mae']:.1f} mg/dL lower MAE</li>"""
    else:
        html += f"""
            <li>Model 163632 shows better overall accuracy with {metrics_163632['mae'] - metrics_174012['mae']:.1f} mg/dL lower MAE</li>"""

    # Check prediction spread
    spread_174012 = df_174012['predicted_mean'].max() - df_174012['predicted_mean'].min()
    spread_163632 = df_163632['predicted_mean'].max() - df_163632['predicted_mean'].min()
    ref_spread = max(REFERENCE.values()) - min(REFERENCE.values())

    html += f"""
            <li>Reference glucose spread: {ref_spread} mg/dL</li>
            <li>Model 174012 prediction spread: {spread_174012:.1f} mg/dL</li>
            <li>Model 163632 prediction spread: {spread_163632:.1f} mg/dL</li>"""

    if spread_163632 > spread_174012:
        html += f"""
            <li>Model 163632 shows more variation in predictions, which may indicate better feature learning</li>"""

    html += """
        </ul>
    </div>

</body>
</html>"""

    # Save report
    report_path = data_dir / "MODEL_COMPARISON_174012_vs_163632.html"
    report_path.write_text(html, encoding='utf-8')
    print(f"\nReport saved: {report_path}")

if __name__ == "__main__":
    main()
