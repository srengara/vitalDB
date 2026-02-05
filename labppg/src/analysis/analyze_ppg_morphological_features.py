#!/usr/bin/env python3
"""
Analyze PPG Morphological Features in 697-Cases Model
Extract and analyze features like kurtosis, dicrotic notch, pulse width, etc.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.resnet34_glucose_predictor import ResNet34_1D

class PPGFeatureExtractor:
    """Extract comprehensive PPG features"""

    @staticmethod
    def extract_morphological_features(ppg_window):
        """Extract morphological features from PPG window"""
        features = {}

        # Normalize window for analysis
        ppg = (ppg_window - np.mean(ppg_window)) / (np.std(ppg_window) + 1e-8)

        # 1. PEAK DETECTION AND MORPHOLOGICAL FEATURES
        peaks, properties = signal.find_peaks(ppg, prominence=0.5, distance=20)

        if len(peaks) > 0:
            # Systolic peak amplitude
            features['systolic_peak_amplitude'] = np.mean(ppg[peaks])
            features['peak_to_peak_amplitude'] = np.max(ppg) - np.min(ppg)

            # Pulse width (duration between peaks)
            if len(peaks) > 1:
                features['pulse_width'] = np.mean(np.diff(peaks))
            else:
                features['pulse_width'] = 0

            # Dicrotic notch detection (valley after systolic peak)
            dicrotic_notches = []
            for peak in peaks:
                # Search for valley after peak (within 20-40% of pulse cycle)
                search_start = peak + int(0.2 * 30)  # Approx 20% of typical pulse
                search_end = min(peak + int(0.4 * 30), len(ppg))
                if search_end > search_start:
                    window_after_peak = ppg[search_start:search_end]
                    if len(window_after_peak) > 0:
                        notch_idx = search_start + np.argmin(window_after_peak)
                        dicrotic_notches.append(notch_idx)

            if len(dicrotic_notches) > 0:
                # Dicrotic notch timing (relative to peak)
                notch_times = []
                for i, peak in enumerate(peaks[:len(dicrotic_notches)]):
                    notch_times.append(dicrotic_notches[i] - peak)
                features['dicrotic_notch_timing'] = np.mean(notch_times)

                # Dicrotic notch depth
                notch_depths = []
                for i, notch in enumerate(dicrotic_notches):
                    if i < len(peaks):
                        notch_depths.append(ppg[peaks[i]] - ppg[notch])
                features['dicrotic_notch_depth'] = np.mean(notch_depths) if notch_depths else 0
            else:
                features['dicrotic_notch_timing'] = 0
                features['dicrotic_notch_depth'] = 0
        else:
            features['systolic_peak_amplitude'] = 0
            features['peak_to_peak_amplitude'] = np.max(ppg) - np.min(ppg)
            features['pulse_width'] = 0
            features['dicrotic_notch_timing'] = 0
            features['dicrotic_notch_depth'] = 0

        # 2. SLOPE ANALYSIS
        derivative = np.diff(ppg)
        if len(derivative) > 0:
            features['rising_edge_slope'] = np.mean(derivative[derivative > 0]) if np.any(derivative > 0) else 0
            features['falling_edge_slope'] = np.mean(np.abs(derivative[derivative < 0])) if np.any(derivative < 0) else 0
            features['max_slope'] = np.max(np.abs(derivative))
        else:
            features['rising_edge_slope'] = 0
            features['falling_edge_slope'] = 0
            features['max_slope'] = 0

        # 3. STATISTICAL FEATURES
        features['mean_amplitude'] = np.mean(ppg_window)
        features['std_amplitude'] = np.std(ppg_window)
        features['skewness'] = float(stats.skew(ppg_window))
        features['kurtosis'] = float(stats.kurtosis(ppg_window))

        # 4. AREA FEATURES
        features['area_under_curve'] = np.trapezoid(ppg_window)
        features['signal_energy'] = np.sum(ppg_window ** 2)

        # 5. TEMPORAL FEATURES
        zero_crossings = np.where(np.diff(np.sign(ppg)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(ppg_window)

        return features

    @staticmethod
    def extract_spectral_features(ppg_window):
        """Extract spectral (frequency domain) features"""
        features = {}

        # FFT
        fft = np.fft.fft(ppg_window)
        power_spectrum = np.abs(fft[:len(fft)//2])**2
        freqs = np.fft.fftfreq(len(ppg_window), d=1/100)[:len(fft)//2]  # Assuming 100Hz

        if len(power_spectrum) > 0 and np.sum(power_spectrum) > 0:
            # Spectral peak power
            features['spectral_peak_power'] = np.max(power_spectrum)

            # Spectral centroid
            features['spectral_centroid'] = np.sum(power_spectrum * freqs) / np.sum(power_spectrum)

            # Spectral spread
            centroid = features['spectral_centroid']
            features['spectral_spread'] = np.sqrt(np.sum(((freqs - centroid)**2) * power_spectrum) / np.sum(power_spectrum))

            # Spectral entropy
            power_norm = power_spectrum / np.sum(power_spectrum)
            power_norm = power_norm[power_norm > 0]  # Remove zeros
            features['spectral_entropy'] = -np.sum(power_norm * np.log2(power_norm))

            # Dominant frequency
            features['dominant_frequency'] = freqs[np.argmax(power_spectrum)]
        else:
            features['spectral_peak_power'] = 0
            features['spectral_centroid'] = 0
            features['spectral_spread'] = 0
            features['spectral_entropy'] = 0
            features['dominant_frequency'] = 0

        return features

def load_test_data(case_dir, max_samples=500):
    """Load PPG windows from test case"""
    ppg_file = os.path.join(case_dir, 'ppg_windows.csv')
    glucose_file = os.path.join(case_dir, 'glucose_labels.csv')

    if not os.path.exists(ppg_file) or not os.path.exists(glucose_file):
        return None, None

    ppg_df = pd.read_csv(ppg_file)

    # Handle both formats
    if 'sample_index' in ppg_df.columns:
        # Long format
        windows = []
        for window_idx in sorted(ppg_df['window_index'].unique())[:max_samples]:
            window_df = ppg_df[ppg_df['window_index'] == window_idx].sort_values('sample_index')
            windows.append(window_df['amplitude'].values)
        ppg_data = np.array(windows)
    else:
        # Wide format
        amplitude_cols = sorted([col for col in ppg_df.columns if col.startswith('amplitude_sample_')],
                               key=lambda x: int(x.split('_')[-1]))
        ppg_data = ppg_df[amplitude_cols].values[:max_samples]

    # Load glucose
    glucose_df = pd.read_csv(glucose_file)
    glucose_data = glucose_df['glucose_mg_dl'].values[:len(ppg_data)]

    return ppg_data, glucose_data

def compute_model_sensitivity_to_features(model, device, ppg_windows, feature_name, perturbation=0.1):
    """
    Compute how sensitive the model is to a specific feature
    by perturbing that feature and measuring prediction change
    """
    model.eval()

    # Sample windows
    num_samples = min(100, len(ppg_windows))
    indices = np.random.choice(len(ppg_windows), num_samples, replace=False)
    sample_windows = ppg_windows[indices].copy()

    # Normalize
    ppg_mean = np.mean(sample_windows, axis=1, keepdims=True)
    ppg_std = np.std(sample_windows, axis=1, keepdims=True)
    ppg_std[ppg_std == 0] = 1.0
    normalized = (sample_windows - ppg_mean) / ppg_std

    # Original predictions
    with torch.no_grad():
        inputs = torch.tensor(normalized, dtype=torch.float32).unsqueeze(1).to(device)
        original_preds = model(inputs).cpu().numpy().flatten()

    # Perturb feature and get new predictions
    perturbed_windows = sample_windows.copy()

    # Apply perturbation based on feature type
    if feature_name == 'kurtosis':
        # Add noise to increase/decrease kurtosis
        for i in range(len(perturbed_windows)):
            perturbed_windows[i] += np.random.normal(0, perturbation * np.std(perturbed_windows[i]), len(perturbed_windows[i]))

    elif feature_name == 'skewness':
        # Shift distribution to change skewness
        for i in range(len(perturbed_windows)):
            perturbed_windows[i] = perturbed_windows[i] ** (1 + perturbation)

    elif feature_name == 'pulse_width':
        # Time-stretch signal
        from scipy.interpolate import interp1d
        for i in range(len(perturbed_windows)):
            x_old = np.linspace(0, 1, len(perturbed_windows[i]))
            x_new = np.linspace(0, 1, int(len(perturbed_windows[i]) * (1 + perturbation)))
            f = interp1d(x_old, perturbed_windows[i], kind='cubic', fill_value='extrapolate')
            stretched = f(x_new)
            # Resample back to original length
            perturbed_windows[i] = signal.resample(stretched, len(perturbed_windows[i]))

    elif feature_name == 'peak_amplitude':
        # Scale amplitude
        for i in range(len(perturbed_windows)):
            perturbed_windows[i] *= (1 + perturbation)

    elif feature_name == 'spectral_peak':
        # Add high-frequency noise
        for i in range(len(perturbed_windows)):
            noise = np.sin(2 * np.pi * 5 * np.linspace(0, 1, len(perturbed_windows[i])))
            perturbed_windows[i] += perturbation * np.std(perturbed_windows[i]) * noise

    # Normalize perturbed windows
    perturbed_mean = np.mean(perturbed_windows, axis=1, keepdims=True)
    perturbed_std = np.std(perturbed_windows, axis=1, keepdims=True)
    perturbed_std[perturbed_std == 0] = 1.0
    perturbed_normalized = (perturbed_windows - perturbed_mean) / perturbed_std

    # Perturbed predictions
    with torch.no_grad():
        perturbed_inputs = torch.tensor(perturbed_normalized, dtype=torch.float32).unsqueeze(1).to(device)
        perturbed_preds = model(perturbed_inputs).cpu().numpy().flatten()

    # Compute sensitivity (how much predictions changed)
    sensitivity = np.mean(np.abs(perturbed_preds - original_preds))

    return sensitivity

def analyze_feature_importance_via_correlation(ppg_windows, glucose_labels):
    """
    Analyze which features correlate most with glucose
    This shows what features SHOULD be important
    """
    print("\n" + "=" * 80)
    print("Feature-Glucose Correlation Analysis")
    print("=" * 80)

    extractor = PPGFeatureExtractor()

    # Extract features for all windows
    all_features = []
    for window in ppg_windows[:1000]:  # Sample 1000 windows
        morph_features = extractor.extract_morphological_features(window)
        spectral_features = extractor.extract_spectral_features(window)
        combined = {**morph_features, **spectral_features}
        all_features.append(combined)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['glucose'] = glucose_labels[:len(all_features)]

    # Compute correlations
    correlations = {}
    for col in features_df.columns:
        if col != 'glucose':
            valid_idx = ~(np.isnan(features_df[col]) | np.isinf(features_df[col]) |
                         np.isnan(features_df['glucose']) | np.isinf(features_df['glucose']))
            if valid_idx.sum() > 10:
                corr = np.corrcoef(features_df[col][valid_idx], features_df['glucose'][valid_idx])[0, 1]
                correlations[col] = abs(corr)  # Use absolute correlation
            else:
                correlations[col] = 0

    # Sort by correlation
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 15 Features by Glucose Correlation:")
    print(f"{'Feature':<35} {'|Correlation|':<15} {'Category'}")
    print("-" * 70)

    for feature, corr in sorted_features[:15]:
        # Categorize feature
        if 'spectral' in feature or 'frequency' in feature:
            category = "Spectral"
        elif any(x in feature for x in ['peak', 'notch', 'pulse', 'slope', 'area']):
            category = "Morphological"
        elif any(x in feature for x in ['kurtosis', 'skewness', 'mean', 'std']):
            category = "Statistical"
        else:
            category = "Other"

        print(f"{feature:<35} {corr:<15.4f} {category}")

    return sorted_features, features_df

def main():
    """Main analysis function"""
    print("\n" + "=" * 80)
    print("PPG MORPHOLOGICAL FEATURE ANALYSIS - 697 CASES MODEL")
    print("=" * 80)

    # Paths
    model_path = r"C:\IITM\vitalDB\model\697-cases-model\best_model.pth"
    output_dir = r"C:\IITM\vitalDB\model\697-cases-model"

    # Test cases
    test_cases = {
        'case_101': r"C:\IITM\vitalDB\model\697-cases-model\inference_data_set\case_101",
        'case_104': r"C:\IITM\vitalDB\model\697-cases-model\inference_data_set\case_104",
    }

    # Load model
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    input_length = checkpoint.get('input_length', 100)
    model = ResNet34_1D(input_length=input_length, num_classes=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded (Epoch {checkpoint['epoch']})")

    # Load test data
    print("\n" + "=" * 80)
    print("Loading Test Data")
    print("=" * 80)

    all_windows = []
    all_glucose = []

    for case_name, case_dir in test_cases.items():
        windows, glucose = load_test_data(case_dir, max_samples=500)
        if windows is not None:
            print(f"{case_name}: {len(windows)} windows, glucose={glucose[0]:.0f} mg/dL")
            all_windows.append(windows)
            all_glucose.append(glucose)

    if not all_windows:
        print("ERROR: No test data loaded")
        return

    # Combine data
    ppg_windows = np.vstack(all_windows)
    glucose_labels = np.hstack(all_glucose)

    print(f"\nTotal: {len(ppg_windows)} windows")

    # 1. Analyze feature-glucose correlations (what SHOULD be important)
    sorted_features, features_df = analyze_feature_importance_via_correlation(ppg_windows, glucose_labels)

    # 2. Analyze model sensitivity to features (what IS important to model)
    print("\n" + "=" * 80)
    print("Model Sensitivity to Feature Perturbations")
    print("=" * 80)
    print("Testing how model predictions change when features are perturbed...")

    feature_tests = [
        'kurtosis',
        'pulse_width',
        'peak_amplitude',
        'spectral_peak',
        'skewness'
    ]

    sensitivities = {}
    for feature_name in feature_tests:
        print(f"\nTesting {feature_name}...")
        sensitivity = compute_model_sensitivity_to_features(
            model, device, ppg_windows, feature_name, perturbation=0.2
        )
        sensitivities[feature_name] = sensitivity
        print(f"  Sensitivity: {sensitivity:.4f} (normalized glucose units)")

    # Sort by sensitivity
    sorted_sensitivities = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 80)

    print("\nModel Sensitivity Ranking (What model USES):")
    print(f"{'Feature':<25} {'Sensitivity':<15}")
    print("-" * 40)
    for feature, sens in sorted_sensitivities:
        print(f"{feature:<25} {sens:<15.4f}")

    print("\n\nFeature-Glucose Correlation (What SHOULD be used):")
    print(f"{'Feature':<35} {'|Correlation|':<15}")
    print("-" * 50)
    for feature, corr in sorted_features[:5]:
        print(f"{feature:<35} {corr:<15.4f}")

    # 3. Categorize feature usage
    print("\n" + "=" * 80)
    print("CATEGORY ANALYSIS")
    print("=" * 80)

    # From correlation analysis
    morphological_corr = []
    statistical_corr = []
    spectral_corr = []

    for feature, corr in sorted_features:
        if 'spectral' in feature or 'frequency' in feature:
            spectral_corr.append(corr)
        elif any(x in feature for x in ['peak', 'notch', 'pulse', 'slope', 'area']):
            morphological_corr.append(corr)
        elif any(x in feature for x in ['kurtosis', 'skewness', 'mean', 'std']):
            statistical_corr.append(corr)

    print("\nAverage Absolute Correlation by Category:")
    print(f"  Morphological features: {np.mean(morphological_corr):.4f}")
    print(f"  Statistical features:   {np.mean(statistical_corr):.4f}")
    print(f"  Spectral features:      {np.mean(spectral_corr):.4f}")

    # 4. Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Feature correlations
    ax1 = axes[0, 0]
    top_features = sorted_features[:15]
    feature_names = [f[0] for f in top_features]
    correlations = [f[1] for f in top_features]

    colors = ['red' if 'spectral' in f else 'green' if any(x in f for x in ['peak', 'notch', 'pulse']) else 'blue'
              for f in feature_names]

    ax1.barh(range(len(feature_names)), correlations, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels([f.replace('_', ' ').title()[:25] for f in feature_names], fontsize=9)
    ax1.set_xlabel('|Correlation with Glucose|')
    ax1.set_title('Top 15 Features by Glucose Correlation\n(Red=Spectral, Green=Morphological, Blue=Statistical)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Model sensitivity
    ax2 = axes[0, 1]
    sens_features = [f[0] for f in sorted_sensitivities]
    sens_values = [f[1] for f in sorted_sensitivities]

    ax2.bar(range(len(sens_features)), sens_values, color='purple', alpha=0.7)
    ax2.set_xticks(range(len(sens_features)))
    ax2.set_xticklabels([f.replace('_', ' ').title() for f in sens_features], rotation=45, ha='right')
    ax2.set_ylabel('Prediction Change (normalized)')
    ax2.set_title('Model Sensitivity to Feature Perturbations')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Category comparison
    ax3 = axes[1, 0]
    categories = ['Morphological', 'Statistical', 'Spectral']
    avg_corrs = [np.mean(morphological_corr), np.mean(statistical_corr), np.mean(spectral_corr)]

    ax3.bar(categories, avg_corrs, color=['green', 'blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Average |Correlation|')
    ax3.set_title('Feature Category Importance (by Correlation)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sample PPG with features
    ax4 = axes[1, 1]
    sample_ppg = ppg_windows[0]
    sample_ppg_norm = (sample_ppg - np.mean(sample_ppg)) / np.std(sample_ppg)

    ax4.plot(sample_ppg_norm, linewidth=2, label='PPG Signal')

    # Mark peaks
    peaks, _ = signal.find_peaks(sample_ppg_norm, prominence=0.5, distance=20)
    if len(peaks) > 0:
        ax4.plot(peaks, sample_ppg_norm[peaks], 'ro', markersize=10, label='Systolic Peaks')

    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Normalized Amplitude')
    ax4.set_title('Sample PPG Window with Detected Features')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'morphological_feature_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()

    # 5. Save results
    # Convert all numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            # Handle NaN values
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native(obj.tolist())
        elif isinstance(obj, (int, float)):
            # Handle regular Python float NaN
            if obj != obj:  # NaN check
                return None
            return obj
        else:
            return obj

    results = {
        'model_path': str(model_path),
        'feature_correlations': {k: float(v) for k, v in sorted_features},
        'model_sensitivities': {k: (None if np.isnan(v) or v != v else float(v)) for k, v in sensitivities.items()},
        'category_analysis': {
            'morphological_avg_correlation': float(np.mean(morphological_corr)),
            'statistical_avg_correlation': float(np.mean(statistical_corr)),
            'spectral_avg_correlation': float(np.mean(spectral_corr))
        },
        'top_5_features_by_correlation': [(k, float(v)) for k, v in sorted_features[:5]],
        'feature_sensitivity_ranking': [(k, (None if np.isnan(v) or v != v else float(v))) for k, v in sorted_sensitivities]
    }

    # Convert all remaining numpy types recursively
    results = convert_to_native(results)

    output_json = os.path.join(output_dir, 'morphological_feature_analysis_results.json')
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_json}")

    # Final comparison with Epoch 85
    print("\n" + "=" * 80)
    print("COMPARISON WITH EPOCH 85 ISSUE")
    print("=" * 80)

    spectral_corr_avg = np.mean(spectral_corr)
    morph_corr_avg = np.mean(morphological_corr)

    print("\nEpoch 85 Known Issue:")
    print("  - 100% reliance on spectral_peak_power")
    print("  - 99.97% loss of morphological features (pulse width, kurtosis, peak amplitude)")
    print()

    print("Current Model (697-cases):")
    print(f"  - Spectral features avg correlation: {spectral_corr_avg:.4f}")
    print(f"  - Morphological features avg correlation: {morph_corr_avg:.4f}")
    print(f"  - Ratio (Spectral/Morphological): {spectral_corr_avg/morph_corr_avg if morph_corr_avg > 0 else 'inf':.2f}x")
    print()

    if spectral_corr_avg > morph_corr_avg * 2:
        print("  [WARNING] Spectral features dominate - similar to Epoch 85 issue")
        print("  -> Feature regularization STRONGLY recommended")
    elif spectral_corr_avg > morph_corr_avg:
        print("  [CAUTION] Spectral features slightly elevated")
        print("  -> Feature regularization recommended")
    else:
        print("  [OK] Morphological features properly utilized")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
