"""
Lab PPG Inference Report Generator
Runs inference on all lab samples and compares with reference glucose values
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, r"C:\IITM\vitalDB\src")
from training.resnet34_glucose_predictor import ResidualBlock1D

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

def build_model_from_checkpoint(state_dict):
    """Auto-detect architecture and build model"""
    conv1_kernel = state_dict['conv1.weight'].shape[2]
    has_multi_fc = 'fc1.weight' in state_dict

    class ResNet34_1D_Flex(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=conv1_kernel, stride=2, padding=conv1_kernel // 2, bias=False)
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
                self._use_multi_fc = True
            else:
                self.fc = nn.Linear(512, 1)
                self._use_multi_fc = False

        def _make_layer(self, in_ch, out_ch, blocks, stride=1):
            downsample = None
            if stride != 1 or in_ch != out_ch:
                downsample = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            layers = [ResidualBlock1D(in_ch, out_ch, stride=stride, downsample=downsample)]
            for _ in range(1, blocks):
                layers.append(ResidualBlock1D(out_ch, out_ch))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self._use_multi_fc:
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc_out(x)
            else:
                x = self.fc(x)
            return x

    return ResNet34_1D_Flex()

def load_ppg_windows(data_dir):
    """Load PPG windows from CSV"""
    ppg_file = os.path.join(data_dir, 'ppg_windows.csv')
    if not os.path.exists(ppg_file):
        return None

    ppg_df = pd.read_csv(ppg_file)
    windows = []
    window_lengths = []

    for window_idx in sorted(ppg_df['window_index'].unique()):
        window_df = ppg_df[ppg_df['window_index'] == window_idx].sort_values('sample_index')
        window = window_df['amplitude'].values
        windows.append(window)
        window_lengths.append(len(window))

    if not windows:
        return None

    length_counts = Counter(window_lengths)
    target_length = length_counts.most_common(1)[0][0]

    normalized_windows = []
    for window in windows:
        if len(window) == target_length:
            normalized_windows.append(window)
        elif len(window) > target_length:
            normalized_windows.append(window[:target_length])
        else:
            padded = np.zeros(target_length)
            padded[:len(window)] = window
            normalized_windows.append(padded)

    return np.array(normalized_windows)

def run_inference(model, ppg_data, device, glucose_mean, glucose_std, ppg_global_mean=None, ppg_global_std=None):
    """Run inference and return predictions in mg/dL"""
    # Normalize PPG - use GLOBAL normalization if available (same as training)
    if ppg_global_mean is not None and ppg_global_std is not None:
        # Global normalization (matches training)
        ppg_normalized = (ppg_data - ppg_global_mean) / ppg_global_std
    else:
        # Per-window normalization (fallback)
        ppg_mean = np.mean(ppg_data, axis=1, keepdims=True)
        ppg_std = np.std(ppg_data, axis=1, keepdims=True)
        ppg_std[ppg_std == 0] = 1.0
        ppg_normalized = (ppg_data - ppg_mean) / ppg_std

    predictions = []
    batch_size = 32

    model.eval()
    with torch.no_grad():
        for i in range(0, len(ppg_normalized), batch_size):
            batch = torch.tensor(ppg_normalized[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(device)
            preds = model(batch).cpu().numpy().flatten()
            predictions.extend(preds)

    # Denormalize
    predictions = np.array(predictions) * glucose_std + glucose_mean
    return predictions

def main():
    data_dir = Path(r"C:\IITM\vitalDB\data\LABPPG\ppg_windows_dir")
    model_path = Path(r"C:\IITM\vitalDB\model\latest_model_174012\best_model.pth")

    print("="*80)
    print("LAB PPG INFERENCE REPORT")
    print("="*80)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    norm = checkpoint.get('normalization', {})
    glucose_mean = norm.get('glucose_mean', 121.64)
    glucose_std = norm.get('glucose_std', 37.77)
    ppg_norm_type = norm.get('ppg_normalization', 'per_window')
    ppg_global_mean = norm.get('ppg_mean', None)
    ppg_global_std = norm.get('ppg_std', None)

    print(f"Model loaded: epoch {checkpoint['epoch']}")
    print(f"Glucose normalization: mean={glucose_mean:.2f}, std={glucose_std:.2f}")
    print(f"PPG normalization: {ppg_norm_type}")
    if ppg_global_mean is not None:
        print(f"  PPG global mean: {ppg_global_mean:.4f}, std: {ppg_global_std:.4f}")

    # Run inference on all samples
    results = []

    for sample_name, ref_glucose in REFERENCE.items():
        sample_dir = data_dir / sample_name
        ppg_data = load_ppg_windows(sample_dir)

        if ppg_data is None:
            print(f"  {sample_name}: No data found")
            continue

        predictions = run_inference(model, ppg_data, device, glucose_mean, glucose_std,
                                    ppg_global_mean, ppg_global_std)

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

        print(f"  {sample_name}: Ref={ref_glucose}, Pred={pred_mean:.1f} (+/-{pred_std:.1f}), Error={error:+.1f}")

    # Summary statistics
    df = pd.DataFrame(results)

    mae = df['abs_error'].mean()
    mard = df['rel_error'].mean()
    rmse = np.sqrt((df['error']**2).mean())

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Samples analyzed: {len(results)}")
    print(f"MAE: {mae:.2f} mg/dL")
    print(f"RMSE: {rmse:.2f} mg/dL")
    print(f"MARD: {mard:.2f}%")

    # Save results
    df.to_csv(data_dir / "inference_results.csv", index=False)

    # Generate HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Lab PPG Inference Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #fff; padding: 30px; max-width: 1000px; margin: 0 auto; }}
        h1, h2 {{ color: #667eea; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: center; border: 1px solid #333; }}
        th {{ background: #667eea; }}
        tr:nth-child(even) {{ background: rgba(255,255,255,0.05); }}
        .good {{ color: #2ecc71; }}
        .warn {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        .metric-box {{ display: inline-block; background: rgba(102,126,234,0.15); padding: 20px 30px; margin: 10px; border-radius: 10px; text-align: center; }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; color: #667eea; }}
        .metric-label {{ font-size: 0.9em; color: #bdc3c7; }}
    </style>
</head>
<body>
    <h1>Lab PPG Inference Report</h1>
    <p>Model: latest_model_174012 | Date: January 2026</p>

    <div style="text-align: center; margin: 30px 0;">
        <div class="metric-box">
            <div class="metric-value">{len(results)}</div>
            <div class="metric-label">Samples</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{mae:.1f}</div>
            <div class="metric-label">MAE (mg/dL)</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{mard:.1f}%</div>
            <div class="metric-label">MARD</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{rmse:.1f}</div>
            <div class="metric-label">RMSE</div>
        </div>
    </div>

    <h2>Sample-wise Results</h2>
    <table>
        <tr>
            <th>Sample</th>
            <th>Reference (mg/dL)</th>
            <th>Predicted (mg/dL)</th>
            <th>Std Dev</th>
            <th>Error</th>
            <th>MARD</th>
            <th>Windows</th>
        </tr>"""

    for _, r in df.iterrows():
        err_class = 'good' if r['abs_error'] < 15 else ('warn' if r['abs_error'] < 30 else 'bad')
        html += f"""
        <tr>
            <td><b>{r['sample']}</b></td>
            <td>{r['reference']}</td>
            <td>{r['predicted_mean']:.1f}</td>
            <td>{r['predicted_std']:.1f}</td>
            <td class="{err_class}">{r['error']:+.1f}</td>
            <td class="{err_class}">{r['rel_error']:.1f}%</td>
            <td>{r['num_windows']}</td>
        </tr>"""

    html += f"""
    </table>

    <h2>Clinical Interpretation</h2>
    <table>
        <tr>
            <th>Sample</th>
            <th>Reference Range</th>
            <th>Predicted Range</th>
            <th>Match</th>
        </tr>"""

    def classify(g):
        if g < 70: return 'Hypoglycemic'
        elif g < 100: return 'Normal'
        elif g < 126: return 'Prediabetes'
        else: return 'Diabetic'

    for _, r in df.iterrows():
        ref_class = classify(r['reference'])
        pred_class = classify(r['predicted_mean'])
        match = 'Yes' if ref_class == pred_class else 'No'
        match_class = 'good' if match == 'Yes' else 'bad'
        html += f"""
        <tr>
            <td>{r['sample']}</td>
            <td>{ref_class}</td>
            <td>{pred_class}</td>
            <td class="{match_class}">{match}</td>
        </tr>"""

    html += """
    </table>
</body>
</html>"""

    report_path = data_dir / "LAB_PPG_INFERENCE_REPORT.html"
    report_path.write_text(html, encoding='utf-8')
    print(f"\nReport saved: {report_path}")

if __name__ == "__main__":
    main()
