import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Help message
HELP_MESSAGE = """
Usage: python process_waveform.py [START_THRESHOLD] [END_THRESHOLD]

Parameters:
  START_THRESHOLD     Force value where extraction begins (default: 3750)
  END_THRESHOLD       Force value where extraction ends (default: 36752)

Examples:
  # Use default values
  python process_waveform.py

  # Custom start threshold (extraction starts at 4000, ends at 36752)
  python process_waveform.py 4000

  # Custom start and end thresholds
  python process_waveform.py 4000 37000

  # With help
  python process_waveform.py --help
"""

if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
    print(HELP_MESSAGE)
    sys.exit(0)

# Configuration Parameters
input_file = r'C:\SenzrTech\Multi-channel\Multi-channel-input.xlsx'
output_file = r'C:\SenzrTech\Multi-channel\Multi-channel-output.csv'
plot_file = r'C:\SenzrTech\Multi-channel\waveform_analysis.png'

# Force value thresholds - MODIFY THESE or pass as command line arguments
START_FORCE_THRESHOLD = 3750
END_FORCE_THRESHOLD = 36752

# Parse command line arguments if provided
if len(sys.argv) > 1:
    START_FORCE_THRESHOLD = float(sys.argv[1])
    print(f"Using command-line START threshold: {START_FORCE_THRESHOLD}")

if len(sys.argv) > 2:
    END_FORCE_THRESHOLD = float(sys.argv[2])
    print(f"Using command-line END threshold: {END_FORCE_THRESHOLD}")

print(f"\n{'='*60}")
print(f"Configuration:")
print(f"  START_FORCE_THRESHOLD: {START_FORCE_THRESHOLD}")
print(f"  END_FORCE_THRESHOLD: {END_FORCE_THRESHOLD}")
print(f"{'='*60}\n")

print("Loading data from Excel...")
df = pd.read_excel(input_file)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Find force waveform column (adjust name if different)
force_col = 'Force_pulse'  # Change this if column name is different
if force_col not in df.columns:
    # Try to find it automatically
    possible_cols = [col for col in df.columns if 'force' in col.lower() or 'pulse' in col.lower()]
    if possible_cols:
        force_col = possible_cols[0]
        print(f"\nForce column auto-detected: {force_col}")
    else:
        print(f"\nAvailable columns: {df.columns.tolist()}")
        raise ValueError(f"Cannot find force column. Please check column names.")

# Get force values
force_signal = df[force_col].values
timestamps = df.iloc[:, 0].values  # Assume first column is timestamp

print(f"\nForce signal statistics:")
print(f"  Min: {np.min(force_signal):.0f}")
print(f"  Max: {np.max(force_signal):.0f}")
print(f"  Mean: {np.mean(force_signal):.0f}")

# Find start point: first value >= START_FORCE_THRESHOLD
start_idx = np.where(force_signal >= START_FORCE_THRESHOLD)[0]
if len(start_idx) > 0:
    start_idx = start_idx[0]
    start_value = force_signal[start_idx]
    print(f"\nStart point (first value >= {START_FORCE_THRESHOLD}):")
    print(f"  Index: {start_idx}")
    print(f"  Value: {start_value:.0f}")
    print(f"  Timestamp: {timestamps[start_idx]}")
else:
    print(f"\nWarning: No value >= {START_FORCE_THRESHOLD} found!")
    start_idx = 0
    start_value = force_signal[0]

# Find end point: first value >= END_FORCE_THRESHOLD
end_idx = np.where(force_signal >= END_FORCE_THRESHOLD)[0]
if len(end_idx) > 0:
    end_idx = end_idx[0]
    end_value = force_signal[end_idx]
    print(f"\nEnd point (first value >= {END_FORCE_THRESHOLD}):")
    print(f"  Index: {end_idx}")
    print(f"  Value: {end_value:.0f}")
    print(f"  Timestamp: {timestamps[end_idx]}")
else:
    print(f"\nWarning: No value >= {END_FORCE_THRESHOLD} found!")
    end_idx = len(force_signal) - 1
    end_value = force_signal[end_idx]

start_timestamp = timestamps[start_idx]
end_timestamp = timestamps[end_idx]

print(f"\nBoundary Detection Results:")
print(f"  Start index: {start_idx}, Start timestamp: {start_timestamp}")
print(f"  End index: {end_idx}, End timestamp: {end_timestamp}")
print(f"  Original data points: {len(df)}")
print(f"  Extracted data points: {end_idx - start_idx + 1}")

print(f"\nBoundary Detection Results:")
print(f"  Start index: {start_idx}, Start timestamp: {start_timestamp}")
print(f"  End index: {end_idx}, End timestamp: {end_timestamp}")
print(f"  Original data points: {len(df)}")
print(f"  Extracted data points: {end_idx - start_idx + 1}")

# Extract data within boundaries
df_extracted = df.iloc[start_idx:end_idx+1].copy()

# Save to CSV
df_extracted.to_csv(output_file, index=False)
print(f"\nOutput saved to: {output_file}")
print(f"Extracted data shape: {df_extracted.shape}")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Force signal with boundaries
axes[0].plot(timestamps, force_signal, 'b-', linewidth=1.5, label='Force Signal')
axes[0].axvline(start_timestamp, color='g', linestyle='--', linewidth=2, label=f'Start: {start_timestamp} (value={start_value:.0f})')
axes[0].axvline(end_timestamp, color='r', linestyle='--', linewidth=2, label=f'End: {end_timestamp} (value={end_value:.0f})')
axes[0].fill_between(timestamps[start_idx:end_idx+1], 
                      force_signal[start_idx:end_idx+1].min(), 
                      force_signal[start_idx:end_idx+1].max(),
                      alpha=0.2, color='yellow', label='Extracted region')
axes[0].axhline(3750, color='g', linestyle=':', alpha=0.5, label='Start threshold (3750)')
axes[0].axhline(36752, color='r', linestyle=':', alpha=0.5, label='End threshold (36752)')
axes[0].set_xlabel('Timestamp')
axes[0].set_ylabel('Force Value')
axes[0].set_title(f'{force_col} - Signal with Extraction Boundaries')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Plot 2: Extracted region zoomed in
if end_idx > start_idx:
    axes[1].plot(timestamps[start_idx:end_idx+1], force_signal[start_idx:end_idx+1], 'b-', linewidth=2, label='Extracted Data')
    axes[1].fill_between(timestamps[start_idx:end_idx+1], 
                          force_signal[start_idx:end_idx+1].min(), 
                          force_signal[start_idx:end_idx+1].max(),
                          alpha=0.3, color='yellow')
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('Force Value')
    axes[1].set_title('Extracted Region (Zoomed In)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No data between boundaries', 
                ha='center', va='center', transform=axes[1].transAxes)

plt.tight_layout()
plot_file = r'C:\SenzrTech\Multi-channel\waveform_analysis.png'
plt.savefig(plot_file, dpi=150)
print(f"\nVisualization saved to: {plot_file}")
plt.show()

print("\n" + "="*60)
print("Processing Complete!")
print("="*60)
