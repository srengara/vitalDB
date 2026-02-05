"""
Multi-Channel PPG Analysis Web Application
===========================================
Flask web application for visualizing multi-channel processing pipeline outputs.

Features:
- Browse processed files
- View all intermediate processing steps
- Visualize signal transformations
- Compare windows before/after filtering
- Analyze template matching
- Extract and display key features
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def create_app(data_directory: str = './output'):
    """Create and configure Flask application."""
    app = Flask(__name__)
    app.config['DATA_DIR'] = data_directory

    def get_available_cases() -> List[Dict]:
        """Scan data directory for processed case folders."""
        data_dir = Path(app.config['DATA_DIR'])

        if not data_dir.exists():
            return []

        cases = []
        for case_folder in sorted(data_dir.iterdir()):
            if not case_folder.is_dir():
                continue

            # Look for metadata file
            metadata_file = case_folder / f"{case_folder.name}-metadata.json"

            case_info = {
                'name': case_folder.name,
                'path': str(case_folder),
                'has_metadata': metadata_file.exists()
            }

            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    case_info['metadata'] = metadata
                    case_info['glucose'] = metadata.get('metadata', {}).get('glucose')
                    case_info['channel'] = metadata.get('metadata', {}).get('channel')
                    case_info['status'] = metadata.get('status')
                except Exception as e:
                    case_info['error'] = str(e)

            cases.append(case_info)

        return cases

    def load_csv_safe(filepath: Path) -> Optional[pd.DataFrame]:
        """Safely load CSV file."""
        if not filepath.exists():
            return None
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def load_case_data(case_name: str) -> Dict:
        """Load all intermediate files for a case."""
        case_folder = Path(app.config['DATA_DIR']) / case_name

        if not case_folder.exists():
            return {'error': 'Case folder not found'}

        data = {
            'case_name': case_name,
            'files': {}
        }

        # Load metadata
        metadata_file = case_folder / f"{case_name}-metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                data['metadata'] = json.load(f)

        # Load all intermediate files
        file_types = [
            ('raw', f"{case_name}-01-raw.csv"),
            ('cleaned', f"{case_name}-02-cleaned.csv"),
            ('downsampled', f"{case_name}-03-downsampled.csv"),
            ('preprocessed', f"{case_name}-04-preprocessed.csv"),
            ('peaks', f"{case_name}-05-peaks.csv"),
            ('windows', f"{case_name}-06-windows.csv"),
            ('filtered', f"{case_name}-07-filtered.csv"),
            ('template', f"{case_name}-08-template.csv"),
            ('output', f"{case_name}-output.csv")
        ]

        for file_type, filename in file_types:
            df = load_csv_safe(case_folder / filename)
            if df is not None:
                data['files'][file_type] = df

        return data

    def create_signal_comparison_plot(case_data: Dict) -> str:
        """Create plot comparing signal at different processing stages."""
        files = case_data['files']

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('1. Raw Signal', '2. Cleaned Signal',
                          '3. Downsampled (100Hz)', '4. Preprocessed (Bandpass Filtered)'),
            vertical_spacing=0.08
        )

        # Limit to first 10 seconds for visualization
        max_samples = 1000

        # Plot 1: Raw
        if 'raw' in files:
            df = files['raw'].head(max_samples)
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['ppg'], name='Raw', line=dict(color='blue')),
                row=1, col=1
            )

        # Plot 2: Cleaned
        if 'cleaned' in files:
            df = files['cleaned'].head(max_samples)
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['ppg'], name='Cleaned', line=dict(color='green')),
                row=2, col=1
            )

        # Plot 3: Downsampled
        if 'downsampled' in files:
            df = files['downsampled'].head(max_samples)
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['ppg'], name='Downsampled', line=dict(color='orange')),
                row=3, col=1
            )

        # Plot 4: Preprocessed
        if 'preprocessed' in files:
            df = files['preprocessed'].head(max_samples)
            if 'ppg_preprocessed' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['time'], y=df['ppg_preprocessed'],
                             name='Preprocessed', line=dict(color='red')),
                    row=4, col=1
                )

        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude")
        fig.update_layout(height=1000, showlegend=False, title_text="Signal Processing Pipeline")

        return fig.to_html(full_html=False)

    def create_peak_detection_plot(case_data: Dict) -> str:
        """Create plot showing peak detection on preprocessed signal."""
        files = case_data['files']

        if 'preprocessed' not in files or 'peaks' not in files:
            return "<p>Peak data not available</p>"

        df_signal = files['preprocessed']
        df_peaks = files['peaks']

        # Limit to first 10 seconds
        max_time = df_signal['time'].min() + 10
        df_signal_plot = df_signal[df_signal['time'] <= max_time]

        fig = go.Figure()

        # Plot preprocessed signal
        if 'ppg_preprocessed' in df_signal_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_signal_plot['time'],
                y=df_signal_plot['ppg_preprocessed'],
                mode='lines',
                name='Preprocessed Signal',
                line=dict(color='blue')
            ))

        # Plot detected peaks
        df_peaks_plot = df_peaks[df_peaks['peak_time'] <= max_time]
        fig.add_trace(go.Scatter(
            x=df_peaks_plot['peak_time'],
            y=df_peaks_plot['peak_amplitude'],
            mode='markers',
            name='Detected Peaks',
            marker=dict(color='red', size=10, symbol='x')
        ))

        fig.update_layout(
            title=f"Peak Detection (Total: {len(df_peaks)} peaks)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=500
        )

        return fig.to_html(full_html=False)

    def create_windows_plot(case_data: Dict) -> str:
        """Create plot comparing all windows vs filtered windows."""
        files = case_data['files']

        if 'windows' not in files or 'filtered' not in files:
            return "<p>Window data not available</p>"

        df_all = files['windows']
        df_filtered = files['filtered']

        # Count windows
        num_all = df_all['window_number'].nunique() if 'window_number' in df_all.columns else 0
        num_filtered = df_filtered['window_number'].nunique() if 'window_number' in df_filtered.columns else 0

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'All Windows ({num_all})', f'Filtered Windows ({num_filtered})'),
            horizontal_spacing=0.15
        )

        # Plot all windows (sample first 50)
        if 'window_number' in df_all.columns:
            for window_num in df_all['window_number'].unique()[:50]:
                window_data = df_all[df_all['window_number'] == window_num]
                fig.add_trace(
                    go.Scatter(x=window_data['sample_index'], y=window_data['amplitude'],
                             mode='lines', line=dict(width=0.5), showlegend=False,
                             opacity=0.3),
                    row=1, col=1
                )

        # Plot filtered windows (sample first 50)
        if 'window_number' in df_filtered.columns:
            for window_num in df_filtered['window_number'].unique()[:50]:
                window_data = df_filtered[df_filtered['window_number'] == window_num]
                fig.add_trace(
                    go.Scatter(x=window_data['sample_index'], y=window_data['amplitude'],
                             mode='lines', line=dict(width=0.5), showlegend=False,
                             opacity=0.5),
                    row=1, col=2
                )

        fig.update_xaxes(title_text="Sample Index")
        fig.update_yaxes(title_text="Amplitude")
        fig.update_layout(height=500, title_text="Window Extraction and Filtering")

        return fig.to_html(full_html=False)

    def create_template_plot(case_data: Dict) -> str:
        """Create plot showing template and sample windows."""
        files = case_data['files']

        if 'template' not in files:
            return "<p>Template data not available</p>"

        df_template = files['template']

        fig = go.Figure()

        # Plot template
        fig.add_trace(go.Scatter(
            x=df_template['sample_index'],
            y=df_template['amplitude'],
            mode='lines',
            name='Template',
            line=dict(color='red', width=3)
        ))

        # Overlay sample filtered windows
        if 'filtered' in files:
            df_filtered = files['filtered']
            if 'window_number' in df_filtered.columns:
                for window_num in df_filtered['window_number'].unique()[:10]:
                    window_data = df_filtered[df_filtered['window_number'] == window_num]
                    fig.add_trace(go.Scatter(
                        x=window_data['sample_index'],
                        y=window_data['amplitude'],
                        mode='lines',
                        name=f'Window {window_num}',
                        line=dict(width=1),
                        opacity=0.3
                    ))

        fig.update_layout(
            title="Template Signal (Average of All Windows)",
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            height=500
        )

        return fig.to_html(full_html=False)

    def extract_features(case_data: Dict) -> Dict:
        """Extract key features from processed data."""
        files = case_data['files']
        metadata = case_data.get('metadata', {})

        features = {
            'metadata': metadata.get('metadata', {}),
            'processing': metadata.get('processing', {}),
            'signal_features': {},
            'window_features': {}
        }

        # Signal features
        if 'preprocessed' in files:
            df = files['preprocessed']
            if 'ppg_preprocessed' in df.columns:
                signal = df['ppg_preprocessed'].values
                features['signal_features'] = {
                    'mean': float(np.mean(signal)),
                    'std': float(np.std(signal)),
                    'min': float(np.min(signal)),
                    'max': float(np.max(signal)),
                    'range': float(np.max(signal) - np.min(signal))
                }

        # Window features
        if 'filtered' in files:
            df = files['filtered']
            if 'window_number' in df.columns:
                # Calculate per-window statistics
                window_amplitudes = []
                for window_num in df['window_number'].unique():
                    window_data = df[df['window_number'] == window_num]['amplitude'].values
                    if len(window_data) > 0:
                        window_amplitudes.append({
                            'mean': np.mean(window_data),
                            'std': np.std(window_data),
                            'max': np.max(window_data),
                            'min': np.min(window_data)
                        })

                if window_amplitudes:
                    features['window_features'] = {
                        'mean_amplitude': float(np.mean([w['mean'] for w in window_amplitudes])),
                        'std_amplitude': float(np.std([w['mean'] for w in window_amplitudes])),
                        'mean_range': float(np.mean([w['max'] - w['min'] for w in window_amplitudes])),
                        'std_range': float(np.std([w['max'] - w['min'] for w in window_amplitudes]))
                    }

        # Peak features
        if 'peaks' in files:
            df_peaks = files['peaks']
            if len(df_peaks) > 1:
                peak_intervals = np.diff(df_peaks['peak_time'].values)
                heart_rates = 60.0 / peak_intervals
                features['peak_features'] = {
                    'num_peaks': int(len(df_peaks)),
                    'mean_heart_rate': float(np.mean(heart_rates)),
                    'std_heart_rate': float(np.std(heart_rates)),
                    'min_heart_rate': float(np.min(heart_rates)),
                    'max_heart_rate': float(np.max(heart_rates))
                }

        return features

    # HTML Templates
    INDEX_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Channel PPG Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            .case-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .case-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .case-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .case-card h3 {
                margin-top: 0;
                color: #4CAF50;
            }
            .case-info {
                margin: 10px 0;
                font-size: 14px;
            }
            .case-info strong {
                color: #666;
            }
            .status-success {
                color: #4CAF50;
                font-weight: bold;
            }
            .status-failed {
                color: #f44336;
                font-weight: bold;
            }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 10px;
            }
            .btn:hover {
                background-color: #45a049;
            }
            .no-cases {
                text-align: center;
                padding: 40px;
                background: white;
                border-radius: 8px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Multi-Channel PPG Analysis Dashboard</h1>
        <p><strong>Data Directory:</strong> {{ data_dir }}</p>
        <p><strong>Cases Found:</strong> {{ num_cases }}</p>

        {% if cases %}
        <div class="case-grid">
            {% for case in cases %}
            <div class="case-card">
                <h3>{{ case.name }}</h3>
                <div class="case-info">
                    {% if case.channel %}
                    <strong>Channel:</strong> {{ case.channel }}<br>
                    {% endif %}
                    {% if case.glucose %}
                    <strong>Glucose:</strong> {{ case.glucose }} mg/dL<br>
                    {% endif %}
                    {% if case.status %}
                    <strong>Status:</strong>
                    <span class="status-{{ 'success' if case.status == 'SUCCESS' else 'failed' }}">
                        {{ case.status }}
                    </span>
                    {% endif %}
                </div>
                <a href="/case/{{ case.name }}" class="btn">View Analysis</a>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-cases">
            <h2>No Cases Found</h2>
            <p>No processed data found in the data directory.</p>
            <p>Run the processing pipeline first to generate data.</p>
        </div>
        {% endif %}
    </body>
    </html>
    """

    CASE_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Case: {{ case_name }}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1, h2 {
                color: #333;
            }
            .back-link {
                display: inline-block;
                padding: 10px 20px;
                background-color: #666;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                margin-bottom: 20px;
            }
            .back-link:hover {
                background-color: #555;
            }
            .section {
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metadata-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }
            .metadata-item {
                padding: 10px;
                background: #f9f9f9;
                border-left: 3px solid #4CAF50;
            }
            .metadata-item strong {
                display: block;
                color: #666;
                font-size: 12px;
            }
            .metadata-item span {
                font-size: 18px;
                color: #333;
            }
            .plot-container {
                margin: 20px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← Back to Dashboard</a>

        <h1>Case Analysis: {{ case_name }}</h1>

        <div class="section">
            <h2>Metadata & Processing Info</h2>
            <div class="metadata-grid">
                {% if features.metadata %}
                <div class="metadata-item">
                    <strong>Channel</strong>
                    <span>{{ features.metadata.channel }}</span>
                </div>
                <div class="metadata-item">
                    <strong>Glucose</strong>
                    <span>{{ features.metadata.glucose }} mg/dL</span>
                </div>
                {% if features.metadata.systolic %}
                <div class="metadata-item">
                    <strong>Systolic BP</strong>
                    <span>{{ features.metadata.systolic }} mmHg</span>
                </div>
                {% endif %}
                {% if features.metadata.diastolic %}
                <div class="metadata-item">
                    <strong>Diastolic BP</strong>
                    <span>{{ features.metadata.diastolic }} mmHg</span>
                </div>
                {% endif %}
                {% endif %}

                {% if features.processing %}
                <div class="metadata-item">
                    <strong>Sampling Rate</strong>
                    <span>{{ "%.1f"|format(features.processing.sampling_rate) }} Hz</span>
                </div>
                <div class="metadata-item">
                    <strong>Duration</strong>
                    <span>{{ "%.1f"|format(features.processing.duration_seconds) }} sec</span>
                </div>
                <div class="metadata-item">
                    <strong>Peaks Detected</strong>
                    <span>{{ features.processing.peaks_detected }}</span>
                </div>
                <div class="metadata-item">
                    <strong>Windows Extracted</strong>
                    <span>{{ features.processing.windows_extracted }}</span>
                </div>
                <div class="metadata-item">
                    <strong>Windows Filtered</strong>
                    <span>{{ features.processing.windows_filtered }}</span>
                </div>
                <div class="metadata-item">
                    <strong>Filtering Rate</strong>
                    <span>{{ "%.1f"|format(features.processing.filtering_rate) }}%</span>
                </div>
                <div class="metadata-item">
                    <strong>Data Quality</strong>
                    <span>{{ features.processing.data_quality }}</span>
                </div>
                <div class="metadata-item">
                    <strong>Final Output Rows</strong>
                    <span>{{ features.processing.final_output_rows }}</span>
                </div>
                {% endif %}
            </div>
        </div>

        <div class="section">
            <h2>Signal Processing Pipeline</h2>
            <div class="plot-container">
                {{ signal_plot|safe }}
            </div>
        </div>

        <div class="section">
            <h2>Peak Detection</h2>
            <div class="plot-container">
                {{ peak_plot|safe }}
            </div>
        </div>

        <div class="section">
            <h2>Window Extraction & Filtering</h2>
            <div class="plot-container">
                {{ windows_plot|safe }}
            </div>
        </div>

        <div class="section">
            <h2>Template Matching</h2>
            <div class="plot-container">
                {{ template_plot|safe }}
            </div>
        </div>

        <div class="section">
            <h2>Extracted Features</h2>

            {% if features.signal_features %}
            <h3>Signal Features</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                </tr>
                {% for key, value in features.signal_features.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ "%.4f"|format(value) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}

            {% if features.window_features %}
            <h3>Window Features</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                </tr>
                {% for key, value in features.window_features.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ "%.4f"|format(value) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}

            {% if features.peak_features %}
            <h3>Peak/Heart Rate Features</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                </tr>
                {% for key, value in features.peak_features.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ "%.2f"|format(value) if value is number else value }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </div>
    </body>
    </html>
    """

    # Routes
    @app.route('/')
    def index():
        """Main dashboard showing all cases."""
        cases = get_available_cases()
        return render_template_string(
            INDEX_TEMPLATE,
            data_dir=app.config['DATA_DIR'],
            num_cases=len(cases),
            cases=cases
        )

    @app.route('/case/<case_name>')
    def view_case(case_name: str):
        """View detailed analysis for a specific case."""
        case_data = load_case_data(case_name)

        if 'error' in case_data:
            return f"<h1>Error</h1><p>{case_data['error']}</p>"

        # Generate plots
        signal_plot = create_signal_comparison_plot(case_data)
        peak_plot = create_peak_detection_plot(case_data)
        windows_plot = create_windows_plot(case_data)
        template_plot = create_template_plot(case_data)

        # Extract features
        features = extract_features(case_data)

        return render_template_string(
            CASE_TEMPLATE,
            case_name=case_name,
            signal_plot=signal_plot,
            peak_plot=peak_plot,
            windows_plot=windows_plot,
            template_plot=template_plot,
            features=features
        )

    @app.route('/api/cases')
    def api_cases():
        """API endpoint to get list of cases."""
        cases = get_available_cases()
        return jsonify(cases)

    @app.route('/api/case/<case_name>/features')
    def api_case_features(case_name: str):
        """API endpoint to get features for a case."""
        case_data = load_case_data(case_name)
        if 'error' in case_data:
            return jsonify({'error': case_data['error']}), 404

        features = extract_features(case_data)
        return jsonify(features)

    return app


if __name__ == '__main__':
    # For testing
    app = create_app('./output')
    app.run(debug=True, port=5001)
